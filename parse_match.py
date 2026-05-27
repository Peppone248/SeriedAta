"""
etl/bronze/parse_match.py — Raw HTML → structured Bronze DataFrames.

Bronze layer contract:
  - Data is structured (typed columns, consistent names)
  - Data is NOT yet clean (nulls, anomalies, duplicates may exist)
  - Every row carries match_url + date as join keys
  - No feature engineering — only parsing and renaming

One function per FBref table type, all returning a standardised DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── COLUMN RENAME MAPS ───────────────────────────────────────────────────────
# FBref column names may change across seasons — adjust here if needed.

SUMMARY_COLS = {
    "Player":  "player",
    "Pos":     "position",
    "Age":     "age",
    "Min":     "minutes",
    "Gls":     "goals",
    "Ast":     "assists",
    "PK":      "pk",
    "PKatt":   "pk_att",
    "Sh":      "shots",
    "SoT":     "shots_on_target",
    "CrdY":    "yellow_cards",
    "CrdR":    "red_cards",
    "xG":      "xg",
    "xAG":     "xag",
    "SCA":     "shot_creating_actions",
    "GCA":     "goal_creating_actions",
}

KEEPER_COLS = {
    "Player":   "player",
    "SoTA":     "shots_on_target_against",
    "GA":       "goals_against",
    "Saves":    "saves",
    "Save%":    "save_pct",
    "CS":       "clean_sheet",
    "PSxG":     "psxg",
    "PSxG+/-":  "psxg_plus_minus",
}

SHOTS_COLS = {
    "Minute":   "minute",
    "Player":   "player",
    "Squad":    "team",
    "xG":       "xg",
    "Outcome":  "outcome",
    "Distance": "distance",
    "Body Part":"body_part",
    "Notes":    "notes",
}

PASSING_COLS = {
    "Player":   "player",
    "Pos":      "position",
    "Min":      "minutes",
    "Cmp":      "passes_completed",
    "Att":      "passes_attempted",
    "Cmp%":     "pass_completion_pct",
    "TotDist":  "total_pass_distance",
    "PrgDist":  "progressive_pass_distance",
    "KP":       "key_passes",
    "1/3":      "passes_final_third",
    "PPA":      "passes_penalty_area",
    "CrsPA":    "crosses_penalty_area",
    "PrgP":     "progressive_passes",
}

DEFENSE_COLS = {
    "Player":   "player",
    "Pos":      "position",
    "Min":      "minutes",
    "Tkl":      "tackles",
    "TklW":     "tackles_won",
    "Int":      "interceptions",
    "Blocks":   "blocks",
    "Sh":       "shots_blocked",
    "Pass":     "passes_blocked",
    "Clr":      "clearances",
    "Err":      "errors_leading_to_shot",
}

POSSESSION_COLS = {
    "Player":   "player",
    "Pos":      "position",
    "Min":      "minutes",
    "Touches":  "touches",
    "Att 3rd":  "touches_att_third",
    "Att Pen":  "touches_pen_area",
    "Att":      "dribbles_attempted",
    "Succ":     "dribbles_successful",
    "Succ%":    "dribble_success_pct",
    "Carries":  "carries",
    "TotDist":  "carry_distance",
    "PrgDist":  "progressive_carry_distance",
    "PrgC":     "progressive_carries",
    "Mis":      "miscontrols",
    "Dis":      "dispossessed",
}

MISC_COLS = {
    "Player":   "player",
    "Pos":      "position",
    "Min":      "minutes",
    "CrdY":     "yellow_cards",
    "CrdR":     "red_cards",
    "Fls":      "fouls_committed",
    "Fld":      "fouls_drawn",
    "Off":      "offsides",
    "Won":      "aerial_duels_won",
    "Lost":     "aerial_duels_lost",
    "Won%":     "aerial_duel_win_pct",
}


# ─── PARSERS ─────────────────────────────────────────────────────────────────

def _base_parse(
    df:         pd.DataFrame,
    rename_map: dict[str, str],
    match_url:  str,
    date:       pd.Timestamp,
    team:       str,
    is_home:    int,
) -> pd.DataFrame:
    """
    Common operations applied to every raw FBref player stat table:
      1. Flatten multi-level columns
      2. Rename to standard names
      3. Drop summary/total rows (player == "Squad")
      4. Add join keys: match_url, date, team, is_home
    """
    df = df.copy()

    # flatten multi-level headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(filter(None, map(str, c))).strip()
            for c in df.columns
        ]

    # rename available columns
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # drop aggregate rows
    if "player" in df.columns:
        df = df[~df["player"].isin(["Squad", "Opponent", ""])]
        df = df.dropna(subset=["player"])

    # add join keys
    df["match_url"] = match_url
    df["date"]      = date
    df["team"]      = team
    df["is_home"]   = is_home

    return df.reset_index(drop=True)


def parse_summary(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    """Standard stats + xG for each player in the match."""
    df = _base_parse(raw, SUMMARY_COLS, match_url, date, team, is_home)
    for col in ["goals", "assists", "shots", "shots_on_target",
                "yellow_cards", "red_cards", "xg", "xag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(
            df["minutes"].astype(str).str.replace("+", "", regex=False),
            errors="coerce"
        )
    return df


def parse_keeper(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    """Goalkeeper statistics including PSxG."""
    return _base_parse(raw, KEEPER_COLS, match_url, date, team, is_home)


def parse_shots(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Individual shot events (both teams combined).
    Each row = one shot attempt.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, c))) for c in df.columns]

    df = df.rename(columns={k: v for k, v in SHOTS_COLS.items() if k in df.columns})
    df = df.dropna(subset=["minute"] if "minute" in df.columns else [])
    df["match_url"] = match_url
    df["date"]      = date

    if "xg" in df.columns:
        df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    return df.reset_index(drop=True)


def parse_passing(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    return _base_parse(raw, PASSING_COLS, match_url, date, team, is_home)


def parse_defense(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    return _base_parse(raw, DEFENSE_COLS, match_url, date, team, is_home)


def parse_possession(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    return _base_parse(raw, POSSESSION_COLS, match_url, date, team, is_home)


def parse_misc(
    raw: pd.DataFrame, match_url: str, date: pd.Timestamp,
    team: str, is_home: int,
) -> pd.DataFrame:
    return _base_parse(raw, MISC_COLS, match_url, date, team, is_home)
