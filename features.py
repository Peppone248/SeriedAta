"""Feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level football features.

    This function mutates the given DataFrame and returns it.
    """
    df["points"] = np.select(
        [df["result"] == "W", df["result"] == "D"],
        [3, 1],
        default=0,
    ).astype("int8")

    df["goal_diff"] = df["gf"] - df["ga"]
    df["shot_accuracy"] = np.where(df["sh"] > 0, df["sot"] / df["sh"], np.nan)

    df["is_home"] = (df["venue"] == "Home").astype("int8")
    df["win_flag"] = (df["result"] == "W").astype("int8")
    df["draw_flag"] = (df["result"] == "D").astype("int8")
    df["loss_flag"] = (df["result"] == "L").astype("int8")
    df["clean_sheet"] = (df["ga"] == 0).astype("int8")

    df["xg_diff"] = df["xg"] - df["xga"]
    df["xg_ratio"] = df["xg"] / (df["xga"] + 1e-6)
    df["conversion_rate"] = df["gf"] / (df["sh"] + 1e-6)
    df["shots_allowed_efficiency"] = df["ga"] / (df["sot"] + 1e-6)
    df["points_per_xg"] = np.where(df["xg"] > 0, df["points"] / df["xg"], np.nan)
    df["low_scoring_match"] = ((df["gf"] + df["ga"]) <= 2).astype("int8")
    # ------ if finishing_eff is less than 1, means attacking was wasteful ------
    df["finishing_efficiency"] = np.where(df["xg"] > 0, df["gf"] / df["xg"], np.nan)
    df["defensive_efficiency"] = np.where(
        df["xga"] > 0,
        df["ga"] / df["xga"],
        np.nan
    )

    if "round" in df.columns:
        df["matchweek"] = (
            df["round"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
        )

    df = df.sort_values(["team", "date"])

    df["last_5_points"] = (
        df.groupby("team")["points"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
    )

    df["last_5_goal_diff"] = (
        df.groupby("team")["goal_diff"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
    )

    df["last_5_xg"] = (
        df.groupby("team")["xg"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
    )

    df["xg_trend"] = df.groupby("team")["xg"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().diff()
    )

    df["points_trend"] = df.groupby("team")["points"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().diff()
    )

    # ----- fatigue features ------
    df["date"] = pd.to_datetime(df["date"])

    df["days_rest"] = df.groupby("team")["date"].diff().dt.days

    df["days_rest"] = df["days_rest"].fillna(df["days_rest"].median())

    # Forma ponderata: le ultime partite contano di più
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # window 5
    df["weighted_form"] = (
        df.groupby("team")["points"]
            .transform(lambda x: x.shift(1)
                       .rolling(5, min_periods=5)
                       .apply(lambda v: np.dot(v, weights), raw=True))
    )

    # Consistenza: deviazione standard dei punti ultimi 5
    df["form_consistency"] = (
        df.groupby("team")["points"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
    )

    # Head-to-head storico tra le due squadre specifiche
    df_sorted = df.sort_values(["team", "opponent", "date"])
    df["h2h_win_rate"] = (
        df_sorted.groupby(["team", "opponent"])["win_flag"]
            .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Stanchezza da calendario: numero di partite negli ultimi 14 giorni
    """df["matches_last_14d"] = (
        df.groupby("team")["date"]
            .transform(lambda x: x.expanding()
                       .apply(lambda dates: ((x.iloc[len(dates) - 1] - dates[:-1])
                                             .dt.days <= 14).sum()))
    )"""

    return df


def add_match_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add match-level identifiers for de-duplicating team-perspective rows."""
    df["home_team"] = np.where(df["venue"] == "Home", df["team"], df["opponent"])
    df["away_team"] = np.where(df["venue"] == "Away", df["team"], df["opponent"])
    df["total_goals"] = df["gf"] + df["ga"]
    return df


def build_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global team strength metrics.
    """

    team_strength = (
        df.groupby("team", observed=True)
        .agg(
            avg_points=("points", "mean"),
            avg_goals_for=("gf", "mean"),
            avg_goals_against=("ga", "mean"),
            avg_xg=("xg", "mean"),
            avg_xga=("xga", "mean"),
            win_rate=("win_flag", "mean"),
        )
        .reset_index()
    )

    return team_strength


def build_cumulative_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le feature di forza squadra senza data leakage.

    Problema:
        la media globale include partite future rispetto a quella da predire,
        gonfiando artificialmente le correlazioni con il risultato.

    Soluzione: expanding mean con shift(1) — ogni riga vede solo
    la media delle partite precedenti della stessa squadra.

    Nuove colonne aggiunte:
        cum_avg_points   media cumulativa punti (storico precedente)
        cum_avg_xg       media cumulativa xG
        cum_avg_xga      media cumulativa xGA
        strength_points_diff   differenza team vs avversario (punti)
        strength_xg_diff       differenza team vs avversario (xG)
        strength_xga_diff      differenza team vs avversario (xGA)

    Le prime partite di ogni squadra avranno NaN (nessuno storico disponibile).
    Gestite con fillna(0) alla fine — neutrali per il modello.

    """
    df = df.copy()
    df = df.sort_values(["team", "date"])

    # ── medie cumulative per il team (shift(1): esclude la partita corrente) ──
    for col, src in [("cum_avg_points", "points"), ("cum_avg_xg", "xg"), ("cum_avg_xga", "xga")]:
        df[col] = (
            df.groupby("team")[src]
                .transform(lambda x: x.shift(1).expanding().mean())
        )

    # ── stesse medie per l'avversario: merge su (opponent, date) ──
    opp_stats = (
        df[["team", "date", "cum_avg_points", "cum_avg_xg", "cum_avg_xga"]]
            .rename(columns={
            "team": "opponent",
            "cum_avg_points": "opp_cum_avg_points",
            "cum_avg_xg": "opp_cum_avg_xg",
            "cum_avg_xga": "opp_cum_avg_xga",
        })
    )

    df = df.merge(opp_stats, on=["opponent", "date"], how="left")

    # ── differenze team vs avversario ──
    df["strength_points_diff"] = df["cum_avg_points"] - df["opp_cum_avg_points"]
    df["strength_xg_diff"] = df["cum_avg_xg"] - df["opp_cum_avg_xg"]
    df["strength_xga_diff"] = df["cum_avg_xga"] - df["opp_cum_avg_xga"]

    # ── le prime partite di ogni team non hanno storico → 0 (neutro) ──
    strength_cols = [
        "cum_avg_points", "cum_avg_xg", "cum_avg_xga",
        "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    ]
    df[strength_cols] = df[strength_cols].fillna(0)

    return df


def add_team_strength_to_matches(df: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    """
    Add home/away team strength features to match-level dataframe.
    """

    df = df.copy()

    # home team = team when venue is Home
    # away team = opponent when venue is Home
    df["home_team"] = df["team"]
    df["away_team"] = df["opponent"]

    home = team_strength.add_prefix("home_")
    away = team_strength.add_prefix("away_")

    df = df.merge(
        home,
        left_on="home_team",
        right_on="home_team",
        how="left"
    )

    df = df.merge(
        away,
        left_on="away_team",
        right_on="away_team",
        how="left"
    )

    return df


def add_strength_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparative strength features.
    """

    df = df.copy()

    df["strength_points_diff"] = df["home_avg_points"] - df["away_avg_points"]
    df["strength_xg_diff"] = df["home_avg_xg"] - df["away_avg_xg"]
    df["strength_xga_diff"] = df["home_avg_xga"] - df["away_avg_xga"]
    df["strength_goal_diff"] = df["home_avg_goals_for"] - df["away_avg_goals_for"]

    return df


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le nuove feature al DataFrame."""
    df = df.copy()
    df = df.sort_values(["team", "date"])

    # overperformance rispetto alle attese
    df["finishing_over_xg"] = df["gf"] - df["xg"]

    # qualità media per conclusione
    df["xg_per_shot"] = np.where(df["sh"] > 0, df["xg"] / df["sh"], np.nan)

    # cambio formazione rispetto alla partita precedente
    df["prev_formation"] = df.groupby("team")["formation"].shift(1)
    df["formation_changed"] = (
        df["formation"] != df["prev_formation"]
    ).astype("int8")
    df["formation_changed"] = df["formation_changed"].fillna(0)

    return df


def add_rolling_team_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["team", "date"])

    # rolling features per team
    df["roll_points"] = (
        df.groupby("team")["points"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df["roll_xg"] = (
        df.groupby("team")["xg"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df["roll_xga"] = (
        df.groupby("team")["xga"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df["roll_goal_diff"] = (
        df.groupby("team")["goal_diff"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    return df
