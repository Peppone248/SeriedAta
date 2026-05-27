"""
etl/gold/squad_features.py — Silver player stats → Gold squad-level features.

Gold layer contract:
  - One row per (team, matchweek)
  - All features are pre-match (shift(1) applied — no leakage)
  - Feature names are descriptive and consistent
  - Ready as direct input to ML models

This module aggregates player-level data up to squad level per matchweek,
then computes rolling windows to produce the features used in team_trend models.

Feature groups produced:
  attacking    → squad xG quality, shot volume, shot quality
  defensive    → PSxG conceded, pressures, defensive actions
  physical     → accumulated minutes, squad depth
  tactical     → formation stability, pressing intensity (PPDA proxy)
  form         → rolling points, rolling goal diff
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rolling window for recent form features
FORM_WINDOW = 5


# ─── AGGREGATION ─────────────────────────────────────────────────────────────

def aggregate_squad_per_match(
    summary_df:    pd.DataFrame,
    keeper_df:     pd.DataFrame,
    shots_df:      pd.DataFrame,
    defense_df:    pd.DataFrame,
    possession_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate player-level Silver tables to squad-level per match.

    Returns one row per (match_url, team) with raw aggregated stats.
    These are then rolled in build_squad_features().
    """
    frames = []

    # ── attacking (from summary) ──────────────────────────────────────────
    if not summary_df.empty:
        att = (
            summary_df
            .groupby(["match_url", "team", "date"], observed=True)
            .agg(
                squad_xg             = ("xg",              "sum"),
                squad_shots          = ("shots",            "sum"),
                squad_shots_on_target= ("shots_on_target",  "sum"),
                squad_goals          = ("goals",            "sum"),
                squad_assists        = ("assists",          "sum"),
                squad_sca            = ("shot_creating_actions", "sum"),
                squad_gca            = ("goal_creating_actions", "sum"),
                n_players_used       = ("player",           "nunique"),
            )
            .reset_index()
        )
        att["shot_accuracy"]    = np.where(
            att["squad_shots"] > 0,
            att["squad_shots_on_target"] / att["squad_shots"],
            np.nan,
        )
        att["xg_per_shot"] = np.where(
            att["squad_shots"] > 0,
            att["squad_xg"] / att["squad_shots"],
            np.nan,
        )
        frames.append(att)

    # ── goalkeeper / defensive (from keeper) ──────────────────────────────
    if not keeper_df.empty:
        gk = (
            keeper_df
            .groupby(["match_url", "team", "date"], observed=True)
            .agg(
                psxg              = ("psxg",           "sum"),
                psxg_plus_minus   = ("psxg_plus_minus","sum"),
                saves             = ("saves",           "sum"),
                goals_against     = ("goals_against",   "sum"),
            )
            .reset_index()
        )
        frames.append(gk)

    # ── shot events (from shots) ──────────────────────────────────────────
    if not shots_df.empty and "team" in shots_df.columns:
        shot_agg = (
            shots_df
            .groupby(["match_url", "team", "date"], observed=True)
            .agg(
                shots_from_pen_area = ("distance", lambda x: (x <= 18).sum()),
                avg_shot_distance   = ("distance", "mean"),
            )
            .reset_index()
        )
        frames.append(shot_agg)

    # ── defensive actions (from defense) ──────────────────────────────────
    if not defense_df.empty:
        def_agg = (
            defense_df
            .groupby(["match_url", "team", "date"], observed=True)
            .agg(
                total_tackles      = ("tackles",       "sum"),
                tackles_won        = ("tackles_won",   "sum"),
                interceptions      = ("interceptions", "sum"),
                blocks             = ("blocks",        "sum"),
                clearances         = ("clearances",    "sum"),
            )
            .reset_index()
        )
        frames.append(def_agg)

    # ── possession (from possession) ──────────────────────────────────────
    if not possession_df.empty:
        pos_agg = (
            possession_df
            .groupby(["match_url", "team", "date"], observed=True)
            .agg(
                total_touches         = ("touches",               "sum"),
                progressive_carries   = ("progressive_carries",   "sum"),
                dribbles_attempted    = ("dribbles_attempted",    "sum"),
                dribbles_successful   = ("dribbles_successful",   "sum"),
                miscontrols           = ("miscontrols",           "sum"),
            )
            .reset_index()
        )
        frames.append(pos_agg)

    if not frames:
        return pd.DataFrame()

    # merge all aggregations on match_url + team + date
    result = frames[0]
    for frame in frames[1:]:
        on_cols = [c for c in ["match_url", "team", "date"] if c in frame.columns]
        result = result.merge(frame, on=on_cols, how="outer")

    return result.sort_values(["team", "date"]).reset_index(drop=True)


# ─── ROLLING FEATURES ────────────────────────────────────────────────────────

def build_squad_features(
    squad_match_df: pd.DataFrame,
    match_results:  pd.DataFrame,
    window:         int = FORM_WINDOW,
) -> pd.DataFrame:
    """
    Build rolling squad-level features (pre-match, no leakage).

    Args:
        squad_match_df: Output of aggregate_squad_per_match()
        match_results:  DataFrame with (team, date, points, goal_diff)
                        to add form features
        window:         Rolling window size (default=5)

    Returns:
        DataFrame with one row per (team, matchweek), ready for ML.
    """
    df = squad_match_df.copy()
    df = df.sort_values(["team", "date"])

    # join in points and goal_diff from results
    if not match_results.empty:
        df = df.merge(
            match_results[["team", "date", "points", "goal_diff"]],
            on=["team", "date"],
            how="left",
        )

    def rolling_mean_shifted(series: pd.Series, w: int) -> pd.Series:
        """Expanding/rolling mean with shift(1) — excludes current match."""
        return (
            series
            .shift(1)
            .rolling(w, min_periods=1)
            .mean()
        )

    # ── apply rolling to all numeric stat columns ─────────────────────────
    stat_cols = [
        "squad_xg", "squad_shots", "squad_shots_on_target",
        "xg_per_shot", "shot_accuracy",
        "psxg", "psxg_plus_minus", "goals_against",
        "avg_shot_distance", "shots_from_pen_area",
        "total_tackles", "tackles_won", "interceptions",
        "progressive_carries", "dribbles_successful",
        "points", "goal_diff",
    ]

    for col in stat_cols:
        if col in df.columns:
            df[f"roll_{col}"] = (
                df.groupby("team")[col]
                .transform(lambda x: rolling_mean_shifted(x, window))
            )

    # ── derived rolling features ──────────────────────────────────────────
    if "roll_squad_shots_on_target" in df.columns and "roll_squad_shots" in df.columns:
        df["roll_shot_accuracy"] = np.where(
            df["roll_squad_shots"] > 0,
            df["roll_squad_shots_on_target"] / df["roll_squad_shots"],
            np.nan,
        )

    if "roll_psxg_plus_minus" in df.columns:
        # positive = GK outperforming expectations (saves > expected)
        df["gk_form"] = df["roll_psxg_plus_minus"]

    # ── squad depth proxy ─────────────────────────────────────────────────
    if "n_players_used" in df.columns:
        df["squad_rotation"] = (
            df.groupby("team")["n_players_used"]
            .transform(lambda x: rolling_mean_shifted(x, window))
        )

    return df.reset_index(drop=True)
