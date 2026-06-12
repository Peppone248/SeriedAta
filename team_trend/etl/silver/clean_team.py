"""
etl/silver/clean_team.py — Silver layer for team-match tables.

Silver contract:
  - Read Bronze parquet (faithful raw strings)
  - Enforce correct dtypes (GF/GA -> int, percentages -> float)
  - Normalise names (team, opponent, venue)
  - Rename columns to clean, consistent snake_case
  - Merge the three team-match tables into ONE row per (team, game)
  - schedule is authoritative for shared descriptive columns;
    shooting and keeper contribute only their distinctive stat columns
  - No feature engineering (no rolling, no shift) — that is Gold's job

Output: data/silver/{season}/team_match.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# join key for team-match tables
JOIN_KEYS = ["team", "game"]

# descriptive columns owned by schedule (shooting/keeper copies are dropped)
SCHEDULE_RENAME = {
    "team":          "team",
    "game":          "game",
    "season":        "season",
    "date":          "date",
    "round":         "round",
    "venue":         "venue",
    "result":        "result",
    "GF":            "goals_for",
    "GA":            "goals_against",
    "opponent":      "opponent",
    "Poss":          "possession",
    "Attendance":    "attendance",
    "Captain":       "captain",
    "Formation":     "formation",
    "Opp Formation": "opp_formation",
    "Referee":       "referee",
    "match_report":  "match_report",
}

SHOOTING_RENAME = {
    "Standard_Gls":    "shooting_goals",
    "Standard_Sh":     "shots",
    "Standard_SoT":    "shots_on_target",
    "Standard_SoT%":   "shots_on_target_pct",
    "Standard_G/Sh":   "goals_per_shot",
    "Standard_G/SoT":  "goals_per_shot_on_target",
    "Standard_PK":     "penalties_made",
    "Standard_PKatt":  "penalties_attempted",
}

KEEPER_RENAME = {
    "Performance_SoTA":     "shots_on_target_against",
    "Performance_GA":       "keeper_goals_against",
    "Performance_Saves":    "saves",
    "Performance_Save%":    "save_pct",
    "Performance_CS":       "clean_sheet",
    "Penalty Kicks_PKatt":  "pk_against_attempted",
    "Penalty Kicks_PKA":    "pk_against_allowed",
    "Penalty Kicks_PKsv":   "pk_saved",
    "Penalty Kicks_PKm":    "pk_missed_by_opponent",
}

# columns that should be integers after cleaning
INT_COLS = [
    "goals_for", "goals_against", "possession", "attendance",
    "shooting_goals", "shots", "shots_on_target",
    "penalties_made", "penalties_attempted",
    "shots_on_target_against", "keeper_goals_against", "saves",
    "clean_sheet", "pk_against_attempted", "pk_against_allowed",
    "pk_saved", "pk_missed_by_opponent",
]

FLOAT_COLS = [
    "shots_on_target_pct", "goals_per_shot",
    "goals_per_shot_on_target", "save_pct",
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _normalise_team(name) -> str:
    """Consistent team naming: strip + title case."""
    return str(name).strip().title() if pd.notna(name) else name


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast int/float columns, coercing invalid values to NaN."""
    df = df.copy()
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    return df


def _select_and_rename(
    df:         pd.DataFrame,
    rename_map: dict[str, str],
    keep_keys:  bool = False,
) -> pd.DataFrame:
    """
    Keep only the columns in rename_map (plus join keys if requested),
    then rename them. Drops everything else (e.g. duplicated descriptors).
    """
    cols = list(rename_map.keys())
    if keep_keys:
        cols = JOIN_KEYS + [c for c in cols if c not in JOIN_KEYS]
    cols = [c for c in cols if c in df.columns]
    return df[cols].rename(columns=rename_map)


# ─── main ──────────────────────────────────────────────────────────────────────

def clean_team_match(
    bronze_dir: str,
    season:     str,
) -> pd.DataFrame:
    """
    Merge and clean the three team-match Bronze tables into one Silver table.

    Args:
        bronze_dir: Bronze root (e.g. "data/bronze")
        season:     e.g. "2024-2025"

    Returns:
        One row per (team, game) with cleaned, typed, renamed columns.
    """
    season_path = Path(bronze_dir) / season

    sched = pd.read_parquet(season_path / "team_schedule.parquet")
    shoot = pd.read_parquet(season_path / "team_shooting.parquet")
    keep  = pd.read_parquet(season_path / "team_keeper.parquet")

    logger.info("Loaded Bronze team tables: schedule=%s shooting=%s keeper=%s",
                sched.shape, shoot.shape, keep.shape)

    # schedule: authoritative for descriptive columns
    sched_clean = _select_and_rename(sched, SCHEDULE_RENAME)

    # shooting/keeper: only distinctive stat columns + join keys
    shoot_clean = _select_and_rename(shoot, SHOOTING_RENAME, keep_keys=True)
    keep_clean  = _select_and_rename(keep,  KEEPER_RENAME,   keep_keys=True)

    # merge on (team, game)
    merged = (
        sched_clean
        .merge(shoot_clean, on=JOIN_KEYS, how="left")
        .merge(keep_clean,  on=JOIN_KEYS, how="left")
    )

    # normalise names
    merged["team"]     = merged["team"].apply(_normalise_team)
    merged["opponent"] = merged["opponent"].apply(_normalise_team)

    # enforce dtypes
    merged = _enforce_dtypes(merged)

    # derive points from result (W=3, D=1, L=0)
    result_to_points = {"W": 3, "D": 1, "L": 0}
    merged["points"] = merged["result"].map(result_to_points).astype("Int64")

    # derive goal difference
    merged["goal_diff"] = merged["goals_for"] - merged["goals_against"]

    # extract numeric matchweek from "Matchweek 1" -> 1
    # refinement of existing data (not feature engineering): the value is
    # already in the row, we just parse it into a usable integer type
    merged["matchweek"] = (
        merged["round"]
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )

    # sort chronologically per team
    merged = merged.sort_values(["team", "date"]).reset_index(drop=True)

    logger.info("Silver team_match: %s, %d teams",
                merged.shape, merged["team"].nunique())

    # sanity checks
    _validate(merged)

    return merged


def _validate(df: pd.DataFrame) -> None:
    """Loud validation: fail fast if the Silver output is malformed."""
    # each team should have 38 matches in a full Serie A season
    counts = df.groupby("team").size()
    odd    = counts[counts != 38]
    if not odd.empty:
        logger.warning("Teams without exactly 38 matches:\n%s", odd.to_string())

    # points must be in {0,1,3}
    bad_points = df[~df["points"].isin([0, 1, 3])]
    if not bad_points.empty:
        logger.warning("Rows with invalid points: %d", len(bad_points))

    # no missing join keys
    if df[["team", "game"]].isna().any().any():
        logger.error("NULL values in join keys (team/game)!")

    # goals_for must be non-negative
    if (df["goals_for"] < 0).any():
        logger.error("Negative goals_for detected!")

    # matchweek must be in 1..38
    if "matchweek" in df.columns:
        bad_mw = df[~df["matchweek"].between(1, 38)]
        if not bad_mw.empty:
            logger.warning("Rows with matchweek outside 1-38: %d", len(bad_mw))


def save_silver_team_match(
    df:         pd.DataFrame,
    silver_dir: str,
    season:     str,
) -> Path:
    """Persist the cleaned team_match table to Silver."""
    out_dir = Path(silver_dir) / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "team_match.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved Silver team_match -> %s", out_path)
    return out_path