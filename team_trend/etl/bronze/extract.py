"""
etl/bronze/extract.py — Bronze layer: persist raw FBref tables to parquet.

Bronze contract:
  - Faithful to the source: no cleaning, no renaming beyond flattening
  - MultiIndex columns are flattened to single strings (parquet requirement)
  - The MultiIndex row index is reset to columns (league/season/team/game)
  - Each table saved as data/bronze/{season}/{table}.parquet
  - Idempotent: a .done flag marks completed seasons; re-runs skip them

Tables persisted for Serie A (the confirmed availability set):
  team-match:    schedule, shooting, keeper
  player-match:  summary, keepers

This is the only layer that talks to soccerdata. Everything downstream
(Silver, Gold) reads from the parquet files written here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from team_trend.scrapers.fbref_source import FBrefSource

logger = logging.getLogger(__name__)

# confirmed available stat types for Serie A
TEAM_MATCH_STAT_TYPES = ["schedule", "shooting", "keeper"]
PLAYER_MATCH_STAT_TYPES = ["summary", "keepers"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a MultiIndex column header into single underscore-joined strings.

    ('Performance', 'Gls') -> 'Performance_Gls'
    ('date', '')           -> 'date'
    'already_flat'         -> 'already_flat'

    Faithful to source: we preserve the full hierarchy, just make it a
    valid single-level column name that parquet can store.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    df = df.copy()
    df.columns = [
        "_".join(part for part in map(str, col) if part and part != "nan").strip("_")
        for col in df.columns
    ]
    return df


def _reset_index_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move the MultiIndex row index (league/season/team/game) into columns.
    Parquet cannot store a MultiIndex on rows, and we want these as join keys.
    """
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    return df


def _coerce_object_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force object (mixed-type) columns to pure strings so parquet can store them.

    soccerdata returns some columns with mixed types — e.g. 'GF' holds both
    the int 0 and the str '0'. pyarrow infers int64 for the column, then fails
    on the string value.

    This is serialization, NOT cleaning: we preserve the exact value as text
    without interpreting it. Semantic typing (GF -> int, xG -> float) is the
    job of the Silver layer. Keeping Bronze as strings stays faithful to the
    source — no value is altered, only its storage representation.

    NaN is preserved as NaN (not the string 'nan') so Silver can detect nulls.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def _save(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("  saved %-22s %s  ->  %s", label, str(df.shape), path.name)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard Bronze preparation:
      1. reset MultiIndex rows to columns (join keys)
      2. flatten MultiIndex column headers
      3. coerce mixed-type object columns to strings (parquet-safe)
    """
    df = _reset_index_to_columns(df)
    df = _flatten_columns(df)
    df = _coerce_object_columns_to_str(df)
    return df


# ─── team-match extraction ─────────────────────────────────────────────────────

def _extract_team_match(source: FBrefSource, season_dir: Path) -> None:
    """Pull and persist the three team-match tables."""
    logger.info("Team-match tables: %s", TEAM_MATCH_STAT_TYPES)

    for stat_type in TEAM_MATCH_STAT_TYPES:
        try:
            raw = source.read_team_match_stats(stat_type=stat_type)
            df = _prepare(raw)
            _save(df, season_dir / f"team_{stat_type}.parquet", f"team_{stat_type}")
        except Exception as exc:
            logger.error("  failed team_%s: %s", stat_type, exc)


# ─── player-match extraction ───────────────────────────────────────────────────

def _extract_player_match(
        source: FBrefSource,
        season_dir: Path,
        game_ids: list[str],
) -> None:
    """
    Pull and persist the two player-match tables.

    read_player_match_stats works per match, so we pass the full list of
    game_ids at once — soccerdata handles the iteration internally and
    returns a single concatenated DataFrame.
    """
    logger.info("Player-match tables: %s (%d games)",
                PLAYER_MATCH_STAT_TYPES, len(game_ids))

    for stat_type in PLAYER_MATCH_STAT_TYPES:
        try:
            raw = source.read_player_match_stats(
                stat_type=stat_type, match_id=game_ids
            )
            df = _prepare(raw)
            _save(df, season_dir / f"player_{stat_type}.parquet", f"player_{stat_type}")
        except Exception as exc:
            logger.error("  failed player_%s: %s", stat_type, exc)


# ─── orchestrator ──────────────────────────────────────────────────────────────

def run_bronze(
        season: str,
        bronze_dir: str = "data/bronze",
        force: bool = False,
) -> Path:
    """
    Extract all available raw tables for one season and persist to Bronze.

    Args:
        season:     e.g. "2024-2025"
        bronze_dir: root directory for Bronze parquet files
        force:      re-extract even if the season is already marked done

    Returns:
        Path to the season's Bronze directory.
    """
    season_dir = Path(bronze_dir) / season
    done_flag = season_dir / ".done"

    if done_flag.exists() and not force:
        logger.info("Bronze season %s already done - skipping (use force=True)", season)
        return season_dir

    logger.info("=" * 60)
    logger.info("BRONZE extraction: season %s", season)
    logger.info("=" * 60)

    source = FBrefSource(seasons=season)

    # 1. schedule first — it gives us the game_ids needed for player stats
    schedule = source.read_schedule()
    schedule_prepared = _prepare(schedule)
    _save(schedule_prepared, season_dir / "schedule.parquet", "schedule")

    game_ids = schedule["game_id"].dropna().unique().tolist()
    logger.info("Found %d game_ids in schedule", len(game_ids))

    # 2. team-match tables
    _extract_team_match(source, season_dir)

    # 3. player-match tables (slow on first run, cached afterwards)
    _extract_player_match(source, season_dir, game_ids)

    done_flag.touch()
    logger.info("Bronze season %s complete -> %s", season, season_dir)
    return season_dir


def run_bronze_seasons(
        seasons: list[str],
        bronze_dir: str = "data/bronze",
        force: bool = False,
) -> None:
    """Run Bronze extraction for multiple seasons."""
    for season in seasons:
        run_bronze(season, bronze_dir=bronze_dir, force=force)
