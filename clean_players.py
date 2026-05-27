"""
etl/silver/clean_players.py — Bronze → Silver cleaning for player stat tables.

Silver layer contract:
  - All nulls are handled explicitly (filled, flagged, or dropped with reason)
  - dtypes are enforced
  - Duplicates are removed
  - Player names are normalised (strip accents, consistent casing)
  - Outliers are flagged (not removed — let the Gold layer decide)
  - No feature engineering — only cleaning and validation
"""

from __future__ import annotations

import logging
import unicodedata

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── NORMALISATION ────────────────────────────────────────────────────────────

def normalise_player_name(name: str) -> str:
    """
    Normalise a player name:
      - Strip leading/trailing whitespace
      - Remove accents (e.g. 'Müller' → 'Muller')
      - Title case

    Used as a join key across tables that may have inconsistent encoding.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.title()


def normalise_team_name(name: str) -> str:
    """Consistent team name: strip whitespace + title case."""
    return name.strip().title() if isinstance(name, str) else ""


# ─── DTYPE ENFORCEMENT ────────────────────────────────────────────────────────

INT_COLS    = ["goals", "assists", "shots", "shots_on_target",
               "yellow_cards", "red_cards", "pk", "pk_att"]
FLOAT_COLS  = ["xg", "xag", "minutes", "psxg", "psxg_plus_minus",
               "pass_completion_pct", "dribble_success_pct",
               "aerial_duel_win_pct", "save_pct"]
STRING_COLS = ["player", "team", "position", "match_url"]


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their expected types, coercing errors to NaN/0."""
    df = df.copy()

    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# ─── CLEANING PIPELINE ────────────────────────────────────────────────────────

def clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the player summary table.

    Steps:
      1. Normalise player and team names
      2. Enforce dtypes
      3. Fill null stats with 0 (player appeared but had no recorded action)
      4. Flag rows with missing minutes (substitutes with unknown play time)
      5. Remove exact duplicates
    """
    df = df.copy()

    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalise_team_name)

    df = enforce_dtypes(df)

    # zero-fill stat columns — a player listed without a goal scored 0 goals
    stat_cols = [c for c in INT_COLS if c in df.columns]
    df[stat_cols] = df[stat_cols].fillna(0)

    # flag rows where minutes is null
    if "minutes" in df.columns:
        df["minutes_missing"] = df["minutes"].isna().astype(int)
        df["minutes"] = df["minutes"].fillna(0)

    # remove exact duplicates (same player + match_url)
    before = len(df)
    df = df.drop_duplicates(subset=["player", "match_url", "team"], keep="first")
    after  = len(df)
    if before != after:
        logger.debug("Removed %d duplicate rows from summary", before - after)

    return df.reset_index(drop=True)


def clean_keeper(df: pd.DataFrame) -> pd.DataFrame:
    """Clean goalkeeper stats."""
    df = df.copy()

    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalise_team_name)

    df = enforce_dtypes(df)

    float_keeper = ["psxg", "psxg_plus_minus", "save_pct"]
    for col in float_keeper:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["player", "match_url"], keep="first")
    return df.reset_index(drop=True)


def clean_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the shots table.
    Each row is a shot event — keep only rows with a valid minute.
    """
    df = df.copy()

    if "minute" in df.columns:
        df["minute"] = pd.to_numeric(
            df["minute"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        df = df.dropna(subset=["minute"])
        df["minute"] = df["minute"].astype(int)

    if "xg" in df.columns:
        df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalise_team_name)

    return df.reset_index(drop=True)


def clean_passing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean passing stats."""
    df = df.copy()
    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    df = enforce_dtypes(df)
    return df.drop_duplicates(subset=["player", "match_url", "team"], keep="first")


def clean_defense(df: pd.DataFrame) -> pd.DataFrame:
    """Clean defensive stats."""
    df = df.copy()
    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    df = enforce_dtypes(df)
    return df.drop_duplicates(subset=["player", "match_url", "team"], keep="first")


def clean_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Clean possession stats."""
    df = df.copy()
    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    df = enforce_dtypes(df)
    return df.drop_duplicates(subset=["player", "match_url", "team"], keep="first")


def clean_misc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean miscellaneous stats (fouls, aerial duels, offsides)."""
    df = df.copy()
    if "player" in df.columns:
        df["player"] = df["player"].apply(normalise_player_name)
    df = enforce_dtypes(df)
    return df.drop_duplicates(subset=["player", "match_url", "team"], keep="first")
