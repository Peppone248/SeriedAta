"""
etl/silver/clean_players.py — Silver layer for player-match tables.

Two responsibilities:
  1. Clean the raw player rows (dtypes, name normalisation)
  2. Aggregate player-level rows UP to one row per (team, game)

The aggregation is the key value-add of the player data: it produces
squad-level signals that team-match data cannot give us —
  - squad rotation (how many players used, how concentrated minutes are)
  - aggregate defensive workload (tackles won, interceptions summed)
  - discipline (cards, fouls summed)
  - attacking volume from players (shots, shots on target summed)

Why aggregate in Silver and not Gold?
  The aggregation here is a deterministic roll-UP within a single match
  (sum/count over players in the same game). It uses only the current
  match's rows, so it is still "one row per (team, game)" — the same grain
  as team_match. Gold will do the temporal roll-FORWARD (shift + rolling
  across matchweeks), which is the leakage-sensitive part.

Output: data/silver/{season}/player_match_agg.parquet
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JOIN_KEYS = ["team", "game"]


# ─── name normalisation ────────────────────────────────────────────────────────

def _strip_accents(name) -> str:
    """'Müller' -> 'Muller'. Used so player keys join across tables."""
    if not isinstance(name, str):
        return name
    decomposed = unicodedata.normalize("NFD", name.strip())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _normalise_team(name) -> str:
    return str(name).strip().title() if pd.notna(name) else name


# ─── cleaning (row-level) ──────────────────────────────────────────────────────

def clean_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw player_summary rows.
      - normalise team and player names
      - ensure stat columns are numeric (Bronze stored some as strings)
      - drop rows with no player name (malformed)
    """
    df = df.copy()

    df["team"]   = df["team"].apply(_normalise_team)
    df["player"] = df["player"].apply(_strip_accents)

    df = df[df["player"].notna() & (df["player"].astype(str).str.len() > 0)]

    stat_cols = [c for c in df.columns if c.startswith("Performance_")]
    for col in stat_cols + ["min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")

    return df.reset_index(drop=True)


def clean_player_keepers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw player_keepers rows."""
    df = df.copy()

    df["team"]   = df["team"].apply(_normalise_team)
    df["player"] = df["player"].apply(_strip_accents)
    df = df[df["player"].notna() & (df["player"].astype(str).str.len() > 0)]

    stat_cols = [c for c in df.columns if c.startswith("Shot Stopping_")]
    for col in stat_cols + ["min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ─── aggregation (player -> squad) ─────────────────────────────────────────────

def aggregate_summary_to_squad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll player_summary up to one row per (team, game).

    Produces squad-level signals not available from team-match data:
        players_used        → count of players who appeared
        starters_used       → count with >= 60 minutes (proxy for starters)
        minutes_std         → spread of minutes (rotation concentration)
        squad_age_mean      → mean age of players used (needs numeric age)
        sum_*               → summed attacking & defensive contributions
    """
    df = df.copy()

    # numeric age from "27-322" (years-days) -> 27.88
    if "age" in df.columns:
        age_parts = df["age"].astype(str).str.extract(r"(\d+)-(\d+)")
        df["age_numeric"] = (
            pd.to_numeric(age_parts[0], errors="coerce")
            + pd.to_numeric(age_parts[1], errors="coerce") / 365.0
        )

    grouped = df.groupby(JOIN_KEYS, observed=True)

    agg = grouped.agg(
        players_used      = ("player",              "nunique"),
        squad_minutes     = ("min",                 "sum"),
        minutes_std       = ("min",                 "std"),
        squad_age_mean    = ("age_numeric",         "mean"),
        sum_goals         = ("Performance_Gls",     "sum"),
        sum_assists       = ("Performance_Ast",     "sum"),
        sum_shots         = ("Performance_Sh",      "sum"),
        sum_shots_on_tgt  = ("Performance_SoT",     "sum"),
        sum_yellow_cards  = ("Performance_CrdY",    "sum"),
        sum_red_cards     = ("Performance_CrdR",    "sum"),
        sum_fouls         = ("Performance_Fls",     "sum"),
        sum_fouls_drawn   = ("Performance_Fld",     "sum"),
        sum_tackles_won   = ("Performance_TklW",    "sum"),
        sum_interceptions = ("Performance_Int",     "sum"),
    ).reset_index()

    # starters proxy: count of players with >= 60 minutes, computed separately
    starters = (
        df[df["min"] >= 60]
        .groupby(JOIN_KEYS, observed=True)["player"]
        .nunique()
        .reset_index(name="starters_used")
    )
    agg = agg.merge(starters, on=JOIN_KEYS, how="left")

    return agg


def aggregate_keepers_to_squad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll player_keepers up to one row per (team, game).

    A team usually has one keeper per match; if two appear (substitution),
    we sum saves/GA and keep the primary keeper's save% via weighted mean.
    """
    df = df.copy()
    grouped = df.groupby(JOIN_KEYS, observed=True)

    agg = grouped.agg(
        keeper_saves      = ("Shot Stopping_Saves", "sum"),
        keeper_goals_against = ("Shot Stopping_GA",  "sum"),
        keeper_sota       = ("Shot Stopping_SoTA",  "sum"),
        keepers_used      = ("player",              "nunique"),
    ).reset_index()

    # save% recomputed from totals (more robust than averaging per-keeper %)
    agg["keeper_save_pct"] = np.where(
        agg["keeper_sota"] > 0,
        (agg["keeper_saves"] / agg["keeper_sota"] * 100).round(1),
        np.nan,
    )

    return agg


# ─── orchestrator ──────────────────────────────────────────────────────────────

def clean_player_match(bronze_dir: str, season: str) -> pd.DataFrame:
    """
    Full Silver player pipeline:
        load Bronze -> clean rows -> aggregate to squad -> merge summary+keepers

    Returns one row per (team, game) with squad-level player-derived signals.
    """
    season_path = Path(bronze_dir) / season

    raw_summary = pd.read_parquet(season_path / "player_summary.parquet")
    raw_keepers = pd.read_parquet(season_path / "player_keepers.parquet")

    logger.info("Loaded Bronze player tables: summary=%s keepers=%s",
                raw_summary.shape, raw_keepers.shape)

    summary_clean = clean_player_summary(raw_summary)
    keepers_clean = clean_player_keepers(raw_keepers)

    summary_agg = aggregate_summary_to_squad(summary_clean)
    keepers_agg = aggregate_keepers_to_squad(keepers_clean)

    merged = summary_agg.merge(keepers_agg, on=JOIN_KEYS, how="left")

    merged = merged.sort_values(["team", "game"]).reset_index(drop=True)

    logger.info("Silver player_match_agg: %s, %d teams",
                merged.shape, merged["team"].nunique())

    _validate(merged)
    return merged


def _validate(df: pd.DataFrame) -> None:
    """Loud validation for the aggregated player table."""
    # players_used should be plausible (11 starters + subs, ~13-20)
    odd = df[~df["players_used"].between(11, 25)]
    if not odd.empty:
        logger.warning("Rows with implausible players_used: %d", len(odd))

    # squad_minutes should be ~990-1050 (11 players * 90 + stoppage subs)
    if "squad_minutes" in df.columns:
        odd_min = df[~df["squad_minutes"].between(900, 1200)]
        if not odd_min.empty:
            logger.warning("Rows with implausible squad_minutes: %d", len(odd_min))

    if df[JOIN_KEYS].isna().any().any():
        logger.error("NULL values in join keys!")


def save_silver_player_match(df: pd.DataFrame, silver_dir: str, season: str) -> Path:
    out_dir = Path(silver_dir) / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "player_match_agg.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved Silver player_match_agg -> %s", out_path)
    return out_path


# ─── per-player historical stats (for Gold squad-quality features) ────────────

def build_player_history(bronze_dir: str, season: str) -> pd.DataFrame:
    """
    Per-player, per-match table with HISTORICAL stats (no leakage).

    For each (player, match) row, computes the player's stats accumulated
    BEFORE this match using shift(1).expanding(), grouped by (team, player).
    This is the per-player analogue of the rolling team features.

    Output schema:
        team, match_url, player, position, minutes_this_match, date,
        hist_goals_per_90, hist_assists_per_90, hist_shots_per_90,
        hist_xg_proxy_per_90, hist_minutes_total

    The hist_*_per_90 stats describe each player's productivity BEFORE the
    current match. They are the building blocks for squad-quality features
    in Gold: "what is the historical quality of the starting XI?"

    Why we don't have real xG: as discovered earlier, Serie A per-match xG
    isn't exposed by soccerdata. We use SoT/shots ratios as a quality proxy.
    """
    season_path = Path(bronze_dir) / season
    raw = pd.read_parquet(season_path / "player_summary.parquet")

    df = clean_player_summary(raw)

    # extract date from the 'game' field, which has format "YYYY-MM-DD Home-Away"
    # player_summary Bronze has no standalone 'date' column
    if "date" not in df.columns and "game" in df.columns:
        df["date"] = pd.to_datetime(
            df["game"].astype(str).str.extract(r"^(\d{4}-\d{2}-\d{2})")[0],
            errors="coerce",
        )

    if "date" not in df.columns or df["date"].isna().all():
        raise ValueError("Could not derive 'date' from player_summary Bronze.")

    df = df.sort_values(["player", "team", "date"]).reset_index(drop=True)

    grp = df.groupby(["player", "team"], observed=True)

    # cumulative stats BEFORE the current match (shift(1) then expanding sum)
    df["hist_goals_total"]   = grp["Performance_Gls"].transform(lambda x: x.shift(1).expanding().sum())
    df["hist_assists_total"] = grp["Performance_Ast"].transform(lambda x: x.shift(1).expanding().sum())
    df["hist_shots_total"]   = grp["Performance_Sh"].transform(lambda x: x.shift(1).expanding().sum())
    df["hist_sot_total"]     = grp["Performance_SoT"].transform(lambda x: x.shift(1).expanding().sum())
    df["hist_minutes_total"] = grp["min"].transform(lambda x: x.shift(1).expanding().sum())

    # per-90 normalisation (the standard football comparability metric)
    # avoid division by zero: replace 0 minutes with NaN -> per_90 becomes NaN
    safe_minutes = df["hist_minutes_total"].where(df["hist_minutes_total"] > 0)
    df["hist_goals_per_90"]   = (df["hist_goals_total"]   * 90 / safe_minutes).fillna(0)
    df["hist_assists_per_90"] = (df["hist_assists_total"] * 90 / safe_minutes).fillna(0)
    df["hist_shots_per_90"]   = (df["hist_shots_total"]   * 90 / safe_minutes).fillna(0)
    df["hist_sot_per_90"]     = (df["hist_sot_total"]     * 90 / safe_minutes).fillna(0)

    # xG proxy: shots-on-target rate (since real xG is unavailable for Serie A)
    df["hist_xg_proxy_per_90"] = df["hist_sot_per_90"] * 0.3   # ~30% conversion benchmark

    cols = [
        "team", "match_url", "player", "pos", "min", "date",
        "hist_minutes_total",
        "hist_goals_per_90", "hist_assists_per_90",
        "hist_shots_per_90", "hist_sot_per_90", "hist_xg_proxy_per_90",
    ]
    cols = [c for c in cols if c in df.columns]
    out  = df[cols].rename(columns={"min": "minutes_this_match"})

    logger.info("Built player_history: %s, %d unique players",
                out.shape, out["player"].nunique())
    return out


def save_silver_player_history(df: pd.DataFrame, silver_dir: str, season: str) -> Path:
    out_dir  = Path(silver_dir) / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "player_history.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved Silver player_history -> %s", out_path)
    return out_path