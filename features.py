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
