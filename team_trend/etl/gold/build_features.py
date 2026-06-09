"""
etl/gold/build_features.py — Gold layer: model-ready squad-momentum features.

Gold contract:
  - Input: Silver tables (team_match + player_match_agg), one row per (team, game)
  - Join them, sort chronologically per (team, season)
  - Build PRE-MATCH features only (everything uses shift(1) or earlier)
  - Build the prediction TARGET (points over the next N matchweeks)
  - Output: one row per (team, matchweek), ready for a regression model

The critical principle here is temporal integrity:
  - Rolling form features look BACKWARD with shift(1): they describe the team
    as it was BEFORE kickoff, never including the current match.
  - The target looks FORWARD: it sums points in the matches AFTER the current
    one. The current row's own result is in neither the features nor the target.
  - All temporal ops are grouped by (team, season) so they never cross a
    season boundary or bleed between teams.

Output: data/gold/squad_momentum.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JOIN_KEYS    = ["team", "game"]
ROLL_WINDOW  = 5     # matches in the backward form window
TARGET_HORIZON = 5   # matchweeks summed into the forward target
TARGET_COL   = f"next_{TARGET_HORIZON}_matchweek_points"

# raw per-match columns that get rolled into backward form features
ROLL_COLS = [
    # team-match attacking / result
    "points", "goal_diff", "goals_for", "goals_against",
    "possession", "shots", "shots_on_target", "shots_on_target_pct",
    "goals_per_shot", "save_pct", "saves",
    # player-derived squad signals
    "players_used", "starters_used", "minutes_std", "squad_age_mean",
    "sum_tackles_won", "sum_interceptions", "sum_yellow_cards", "sum_fouls",
]


# ─── join ──────────────────────────────────────────────────────────────────────

def _merge_silver(team_match: pd.DataFrame, player_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Join the two Silver tables on (team, game). team_match is the spine
    (it carries date, matchweek, season, venue, result, points).

    Both tables expose keeper_goals_against (team_match from the keeper table,
    player_agg from the keeper aggregation). They are the same quantity, so we
    keep the team_match version and drop the player-side duplicate to avoid
    a _x / _y suffix collision.
    """
    player_agg = player_agg.copy()
    dup_cols = [c for c in player_agg.columns
                if c in team_match.columns and c not in JOIN_KEYS]
    if dup_cols:
        logger.info("Dropping duplicate columns from player_agg before merge: %s",
                    dup_cols)
        player_agg = player_agg.drop(columns=dup_cols)

    merged = team_match.merge(player_agg, on=JOIN_KEYS, how="left")
    logger.info("Merged Silver: team_match=%s + player_agg=%s -> %s",
                team_match.shape, player_agg.shape, merged.shape)
    return merged


# ─── context features (pre-match) ──────────────────────────────────────────────

def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contextual pre-match features that don't need a rolling window.

    is_home          → 1 if playing at home
    days_rest        → days since this team's previous match (fatigue proxy)
    cum_points       → points accumulated BEFORE this match (season to date)
    cum_goal_diff    → goal difference accumulated before this match
    league_position  → standing entering this matchweek (1 = top)
    season_progress  → matchweek / 38, pressure proxy
    """
    df = df.copy()
    df = df.sort_values(["season", "team", "matchweek"]).reset_index(drop=True)

    grp = df.groupby(["team", "season"], observed=True)

    # home/away from venue
    df["is_home"] = (df["venue"].astype(str).str.title() == "Home").astype("Int64")

    # days since previous match (first match of season -> NaN -> filled with 7)
    df["days_rest"] = grp["date"].diff().dt.days
    df["days_rest"] = df["days_rest"].fillna(7).clip(lower=0, upper=30).astype("Int64")

    # cumulative points/goal_diff BEFORE the current match (shift(1) then cumsum)
    df["cum_points"] = (
        grp["points"].transform(lambda x: x.shift(1).cumsum()).fillna(0).astype("Int64")
    )
    df["cum_goal_diff"] = (
        grp["goal_diff"].transform(lambda x: x.shift(1).cumsum()).fillna(0).astype("Int64")
    )
    df["cum_goals_for"] = (
        grp["goals_for"].transform(lambda x: x.shift(1).cumsum()).fillna(0).astype("Int64")
    )

    # league position entering the matchweek: rank by (points, gd, gf) desc
    sort_key = (
        df["cum_points"].astype(float) * 10_000
        + df["cum_goal_diff"].astype(float) * 100
        + df["cum_goals_for"].astype(float)
    )
    df["_sort_key"] = sort_key
    df["league_position"] = (
        df.groupby(["season", "matchweek"], observed=True)["_sort_key"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    df = df.drop(columns=["_sort_key"])

    # season progress in [0, 1]
    df["season_progress"] = (df["matchweek"] / 38.0).clip(upper=1.0)

    return df


# ─── rolling form features (pre-match, shift(1)) ───────────────────────────────

def _build_rolling(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Backward rolling means with shift(1).

    shift(1) drops the current match from its own window, so each feature
    describes the team's form ENTERING the match — no leakage.
    Grouped by (team, season) so windows never cross seasons or teams.

    min_periods=1: early-season matches still get a value (mean of whatever
    is available). The tradeoff is that matchweek 2's "form" is just
    matchweek 1 — noisy, but we keep the row rather than discard it.
    """
    df = df.copy()
    grp = df.groupby(["team", "season"], observed=True)

    for col in ROLL_COLS:
        if col not in df.columns:
            logger.warning("ROLL_COLS column missing, skipped: %s", col)
            continue
        df[f"roll{window}_{col}"] = (
            grp[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        )

    return df


# ─── forward target ────────────────────────────────────────────────────────────

def _build_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Target = sum of points in the NEXT `horizon` matches (not incl. current).

    Implemented as the sum of points shifted back by 1..horizon within
    (team, season). If any of the next `horizon` matches is missing
    (end of season), the sum is NaN and the row is dropped later — exactly
    the rows where a full forward horizon doesn't exist.
    """
    df = df.copy()
    grp = df.groupby(["team", "season"], observed=True)["points"]

    target = None
    for k in range(1, horizon + 1):
        shifted = grp.shift(-k)
        target = shifted if target is None else target + shifted

    df[TARGET_COL] = target
    return df


# ─── orchestrator ──────────────────────────────────────────────────────────────

def build_gold(
    team_match: pd.DataFrame,
    player_agg: pd.DataFrame,
    drop_incomplete_target: bool = True,
) -> pd.DataFrame:
    """
    Full Gold pipeline for one or more seasons of Silver data.

    Args:
        team_match: Silver team_match (can span multiple seasons)
        player_agg: Silver player_match_agg
        drop_incomplete_target: drop rows where the forward target is NaN
                                (the last `horizon` matchweeks of each season)

    Returns:
        Model-ready DataFrame, one row per (team, matchweek).
    """
    df = _merge_silver(team_match, player_agg)
    df = _build_context(df)
    df = _build_rolling(df)
    df = _build_target(df)

    n_before = len(df)
    if drop_incomplete_target:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info("Target built: %d rows, %d dropped (incomplete forward horizon)",
                len(df), n_before - len(df))

    df[TARGET_COL] = df[TARGET_COL].astype("Int64")
    return df


def run_gold(
    silver_dir: str,
    seasons:    list[str],
    gold_dir:   str = "data/gold",
    save:       bool = True,
) -> pd.DataFrame:
    """
    Load Silver for the given seasons, build Gold, optionally persist.

    Rolling/target/standings are all within (team, season), so loading
    multiple seasons and processing together is safe — group boundaries
    prevent any cross-season leakage.
    """
    team_frames, player_frames = [], []

    for season in seasons:
        sp = Path(silver_dir) / season
        tm = pd.read_parquet(sp / "team_match.parquet")
        pa = pd.read_parquet(sp / "player_match_agg.parquet")
        team_frames.append(tm)
        player_frames.append(pa)

    team_match = pd.concat(team_frames, ignore_index=True)
    player_agg = pd.concat(player_frames, ignore_index=True)

    gold = build_gold(team_match, player_agg)

    if save:
        out_dir = Path(gold_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "squad_momentum.parquet"
        gold.to_parquet(out_path, index=False)
        logger.info("Saved Gold -> %s", out_path)

    return gold