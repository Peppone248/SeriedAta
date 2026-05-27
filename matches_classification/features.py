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
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["last_5_goal_diff"] = (
        df.groupby("team")["goal_diff"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["last_5_xg"] = (
        df.groupby("team")["xg"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["xg_trend"] = df.groupby("team")["xg"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean().diff()
    )

    df["points_trend"] = df.groupby("team")["points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean().diff()
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


def add_parity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature di equilibrio tra le due squadre.
    Valori bassi → partita equilibrata → pareggio più probabile.

    Richiedono che build_cumulative_team_strength() sia già stato eseguito
    (usa cum_avg_points, opp_cum_avg_points, ecc.).
    """
    df = df.copy()

    # ── parità di forza ───────────────────────────────────────────────────
    # distanza assoluta nella forza cumulativa — 0 = squadre identiche
    df["strength_parity"] = (
        df["cum_avg_points"] - df["opp_cum_avg_points"]
    ).abs()

    df["xg_parity"] = (
        df["cum_avg_xg"] - df["opp_cum_avg_xg"]
    ).abs()

    # ── parità di forma recente ───────────────────────────────────────────
    # richiede weighted_form dell'avversario — merge su (opponent, date)
    form_opp = (
        df[["team", "date", "weighted_form"]]
        .rename(columns={"team": "opponent", "weighted_form": "opp_weighted_form"})
    )
    df = df.merge(form_opp, on=["opponent", "date"], how="left")

    df["form_parity"] = (
        df["weighted_form"] - df["opp_weighted_form"]
    ).abs()

    # ── storico pareggi head-to-head ─────────────────────────────────────
    # quante volte queste due squadre hanno pareggiato in passato
    df_sorted = df.sort_values(["team", "opponent", "date"])
    df["h2h_draw_rate"] = (
        df_sorted
        .groupby(["team", "opponent"])["draw_flag"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # ── entrambe le squadre difensive ────────────────────────────────────
    # bassa xG attesa per entrambe → più probabile 0-0 o 1-0
    xga_opp = (
        df[["team", "date", "cum_avg_xga"]]
        .rename(columns={"team": "opponent", "cum_avg_xga": "opp_cum_avg_xga_def"})
    )
    df = df.merge(xga_opp, on=["opponent", "date"], how="left")

    df["both_defensive"] = (
        df["cum_avg_xga"] + df["opp_cum_avg_xga_def"]
    )  # più basso = entrambe difendono bene → partita bloccata

    # ── fase stagionale ───────────────────────────────────────────────────
    # i pareggi aumentano nella fase centrale della stagione
    if "matchweek" in df.columns:
        df["season_phase"] = pd.cut(
            df["matchweek"],
            bins    = [0, 10, 28, 38],
            labels  = [0, 1, 2],    # inizio / metà / fine
            ordered = False,
        ).astype("float32")

    # fillna conservativo: parità neutra = 0 per le diff, mediana per il resto
    parity_cols = [
        "strength_parity", "xg_parity", "form_parity",
        "h2h_draw_rate", "both_defensive",
    ]
    for col in parity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ─── STANDINGS CORE ──────────────────────────────────────────────────────────

def _build_cumulative_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i punti cumulativi di ogni squadra prima di ogni partita.

    Usa groupby(season, team) + shift(1).expanding().sum() per garantire
    che la partita corrente non venga mai inclusa nel calcolo.

    Aggiunge:
        cum_points_before  → punti totali prima di questa partita
        cum_gd_before      → differenza reti cumulativa (tiebreaker)
        cum_gf_before      → gol segnati cumulativi (tiebreaker secondario)
    """
    df = df.copy()
    df = df.sort_values(["season", "team", "matchweek", "date"])

    for col, src in [
        ("cum_points_before", "points"),
        ("cum_gd_before", "goal_diff"),
        ("cum_gf_before", "gf"),
    ]:
        df[col] = (
            df.groupby(["season", "team"])[src]
                .transform(lambda x: x.shift(1).expanding().sum())
                .fillna(0)
        )

    return df


def _assign_league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assegna la posizione in classifica prima di ogni partita.
    Tiebreaker: punti → differenza reti → gol segnati (regola Serie A).

    Aggiunge:
        league_position     → 1 (primo) a 20 (ultimo)
        opp_league_position → posizione dell'avversario
    """
    df = df.copy()

    # rank all'interno di ogni (season, matchweek) — più punti = posizione più bassa
    df["league_position"] = (
        df.groupby(["season", "matchweek"])
            .apply(lambda g: (
            g[["cum_points_before", "cum_gd_before", "cum_gf_before"]]
                .apply(tuple, axis=1)
                .rank(ascending=False, method="min")
                .astype(int)
        ))
            .reset_index(level=[0, 1], drop=True)
    )

    # posizione dell'avversario — merge su (season, matchweek, team=opponent)
    pos_lookup = (
        df[["season", "matchweek", "team", "league_position"]]
            .rename(columns={"team": "opponent", "league_position": "opp_league_position"})
    )
    df = df.merge(pos_lookup, on=["season", "matchweek", "opponent"], how="left")

    return df


# ─── PRESSURE FEATURES ───────────────────────────────────────────────────────

def _add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature di pressione contestuale derivate dalla classifica pre-match.

    Champions League: top 4
    Europa League:    top 6 (per completezza)
    Retrocessione:    ultimi 3 (posizioni 18-20)

    Aggiunge:
        points_gap_top4         → punti da guadagnare per entrare in top 4
                                  (0 se già in top 4)
        points_gap_relegation   → punti di vantaggio sulla zona retrocessione
                                  (negativo = in zona retrocessione)
        is_top_half             → 1 se posizione <= 10
        is_relegation_zone      → 1 se posizione >= 18
        position_diff           → league_position - opp_league_position
                                  (negativo = team meglio posizionato)
        season_progress         → matchweek / 38 ∈ [0, 1]
                                  (quanto è avanzata la stagione)
    """
    df = df.copy()

    def _compute_gaps(group):
        pts = group["cum_points_before"]
        mw = group.name  # (season, matchweek)

        sorted_pts = pts.sort_values(ascending=False).values

        # punti del 4° — se meno di 4 squadre usa l'ultimo
        top4_pts = sorted_pts[3] if len(sorted_pts) > 3 else sorted_pts[-1]
        # punti del 18° — soglia retrocessione
        relegation_pts = sorted_pts[17] if len(sorted_pts) > 17 else sorted_pts[0]

        group = group.copy()
        group["points_gap_top4"] = (top4_pts - pts).clip(lower=0)
        group["points_gap_relegation"] = pts - relegation_pts
        return group

    df = (
        df.groupby(["season", "matchweek"], group_keys=False)
            .apply(_compute_gaps)
    )

    df["is_top_half"] = (df["league_position"] <= 10).astype("int8")
    df["is_relegation_zone"] = (df["league_position"] >= 18).astype("int8")

    # differenza di posizione tra le due squadre
    # negativo = team meglio classificato dell'avversario
    df["position_diff"] = df["league_position"] - df["opp_league_position"]

    # avanzamento stagionale — cattura pressione crescente a fine stagione
    df["season_progress"] = (df["matchweek"] / 38.0).clip(upper=1.0)

    # fillna conservativo: inizio stagione → posizione neutra, gap = 0
    pressure_cols = [
        "league_position", "opp_league_position",
        "points_gap_top4", "points_gap_relegation",
        "is_top_half", "is_relegation_zone",
        "position_diff", "season_progress",
    ]
    for col in pressure_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def add_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa: standings → posizioni → pressure features.

    Richiede che il DataFrame abbia già:
        - colonna "matchweek" (estratta da "round" in add_match_features)
        - colonne "points", "goal_diff", "gf" (da add_match_features)
        - colonne "season", "team", "opponent", "date"

    Returns:
        DataFrame con le nuove colonne aggiunte.
    """
    required = ["matchweek", "points", "goal_diff", "gf",
                "season", "team", "opponent", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonne mancanti per add_standings_features(): {missing}\n"
            f"Assicurati che add_match_features() sia stato chiamato prima."
        )

    df = _build_cumulative_standings(df)
    df = _assign_league_positions(df)
    df = _add_pressure_features(df)

    print(f"  Standings features aggiunte: "
          f"league_position, opp_league_position, points_gap_top4, "
          f"points_gap_relegation, position_diff, season_progress, "
          f"is_top_half, is_relegation_zone")

    return df


# ─── DIAGNOSTICA ─────────────────────────────────────────────────────────────

def print_standings_sample(df: pd.DataFrame, season=None, matchweek=10) -> None:
    """
    Stampa la classifica ricostruita per una giornata specifica.
    Utile per verificare che i valori siano corretti.
    """
    if season is None:
        season = df["season"].max()

    subset = (
        df[(df["season"] == season) & (df["matchweek"] == matchweek)]
        [["team", "league_position", "cum_points_before",
          "cum_gd_before", "points_gap_top4", "points_gap_relegation"]]
            .drop_duplicates("team")
            .sort_values("league_position")
    )

    print(f"\n  Classifica ricostruita — stagione {season}, "
          f"prima della giornata {matchweek}:")
    print(subset.to_string(index=False))


# ─── OPPONENT-ADJUSTED FEATURES ──────────────────────────────────────────────

def add_opponent_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature di forma aggiustate per la qualità dell'avversario.

    Divide le partite storiche in due categorie basandosi su
    opp_league_position (calcolata da add_standings_features):
        strong → avversario nella metà alta (opp_league_position ≤ 10)
        weak   → avversario nella metà bassa (opp_league_position > 10)

    Per ogni categoria calcola medie cumulative con shift(1) per
    garantire che la partita corrente non venga inclusa (no leakage).

    Richiede: opp_league_position, points, xg (da add_standings_features
              e add_match_features).

    Nuove colonne:
        form_vs_strong   → media punti nelle partite storiche vs top-half
        form_vs_weak     → media punti nelle partite storiche vs bottom-half
        xg_vs_strong     → media xG vs top-half
        xg_vs_weak       → media xG vs bottom-half
        big_game_delta   → form_vs_strong - form_vs_weak
                           positivo = performa meglio vs squadre forti
                           negativo = squadra da trasferta facile
    """
    required = ["opp_league_position", "points", "xg", "team", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonne mancanti per add_opponent_adjusted_features(): {missing}\n"
            f"Chiama add_standings_features() prima di questa funzione."
        )

    df = df.copy()
    df = df.sort_values(["team", "date"])

    def _expanding_conditional_mean(
            values: pd.Series,
            mask: pd.Series,
    ) -> pd.Series:
        """
        Expanding mean di `values` solo dove `mask` è True.
        Entrambi i vettori devono essere già shiftati (shift(1) applicato
        esternamente nel gruppo) per garantire no leakage.

        Implementazione efficiente con cumsum:
            mean(t) = cumsum(values * mask)[t] / cumsum(mask)[t]
        """
        weighted_cum = (values * mask).expanding().sum()
        count_cum = mask.expanding().sum()
        return weighted_cum / count_cum.replace(0, np.nan)

    def _compute_group(group: pd.DataFrame) -> pd.DataFrame:
        # shift(1): esclude la partita corrente
        pts = group["points"].shift(1)
        xg = group["xg"].shift(1)
        opp_pos = group["opp_league_position"].shift(1)

        # mask float per moltiplicazione diretta
        strong_mask = (opp_pos <= 10).astype(float)
        weak_mask = (opp_pos > 10).astype(float)

        return pd.DataFrame({
            "form_vs_strong": _expanding_conditional_mean(pts, strong_mask),
            "form_vs_weak": _expanding_conditional_mean(pts, weak_mask),
            "xg_vs_strong": _expanding_conditional_mean(xg, strong_mask),
            "xg_vs_weak": _expanding_conditional_mean(xg, weak_mask),
        }, index=group.index)

    adj = (
        df.groupby("team", group_keys=False)
            .apply(_compute_group)
    )

    df = pd.concat([df, adj], axis=1)

    # feature derivata: delta di performance vs qualità avversario
    df["big_game_delta"] = df["form_vs_strong"] - df["form_vs_weak"]

    # fillna: prime partite senza storico → mediana (valore neutro)
    new_cols = [
        "form_vs_strong", "form_vs_weak",
        "xg_vs_strong", "xg_vs_weak",
        "big_game_delta",
    ]
    for col in new_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"  Opponent-adjusted features aggiunte: {', '.join(new_cols)}")
    return df
