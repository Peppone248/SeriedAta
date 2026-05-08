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
    df["xg_overperformance"] = df["gf"] - df["xg"]
    df["xg_underperformance_def"] = df["xga"] - df["ga"]
    df["points_per_xg"] = np.where(df["xg"] > 0, df["points"] / df["xg"], np.nan)
    df["low_scoring_match"] = ((df["gf"] + df["ga"]) <= 2).astype("int8")

    if "round" in df.columns:
        df["matchweek"] = (
            df["round"].astype("string").str.extract(r"(\d+)", expand=False).astype("Int64")
        )

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
