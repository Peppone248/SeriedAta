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

JOIN_KEYS = ["team", "game"]
ROLL_WINDOW = 5  # matches in the backward form window
TARGET_HORIZON = 5  # matchweeks summed into the forward target
TARGET_COL = f"next_{TARGET_HORIZON}_matchweek_points"

# alternative target: deviation from current rolling form
# momentum_change > 0 means the team OVER-performs its recent baseline
# momentum_change < 0 means the team UNDER-performs (form crash)
MOMENTUM_TARGET_COL = f"momentum_change_{TARGET_HORIZON}"

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

# columns we additionally roll as STANDARD DEVIATION (volatility signal)
# A team with {3,0,3,0,3} and one with {2,1,2,1,3} have the same mean points
# but very different volatility — and very different predictability.
# This directly addresses the mid-table-teams-are-hard pattern in residuals.
VOLATILITY_COLS = ["points", "goal_diff"]


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

    # volatility features: rolling std of points/goal_diff entering the match
    # min_periods=2 because std of one observation is undefined (NaN by design)
    for col in VOLATILITY_COLS:
        if col not in df.columns:
            continue
        df[f"roll{window}_{col}_std"] = (
            grp[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
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


# ─── squad quality from player history (Phase A) ───────────────────────────────

def build_squad_quality_features(
        player_history: pd.DataFrame,
        team_match: pd.DataFrame,
        starter_min: int = 60,
) -> pd.DataFrame:
    """
    For each (team, match), aggregate the HISTORICAL per-90 stats of the
    starting XI (players with minutes >= starter_min in that match).

    The output joins onto Gold via (team, date). Each row produces:

      squad_quality_goals_per_90       sum  of starters' hist_goals_per_90
      squad_quality_assists_per_90     sum  of starters' hist_assists_per_90
      squad_quality_xg_proxy_per_90    sum  of starters' hist_xg_proxy_per_90
      top_scorer_present               1 if team's all-time top scorer is in XI
      top_assister_present             1 if team's top assister is in XI
      starter_avg_experience           mean of hist_minutes_total across starters

    Why "sum" rather than "mean" for the per-90 stats: a team with 3 strikers
    each at 0.5 goals/90 produces more attacking threat than one with 3 at
    0.2/90. Summing reflects cumulative threat; meaning would normalise that
    away.
    """
    p = player_history.copy()

    # join match date from team_match so we can attribute history correctly
    if "date" not in p.columns:
        raise ValueError("player_history must carry 'date' column")
    p["date"] = pd.to_datetime(p["date"])

    # restrict to starters (minutes >= threshold this match)
    starters = p[p["minutes_this_match"] >= starter_min].copy()
    logger.info("Squad quality: %d starter-rows across %d (team, date) groups",
                len(starters),
                starters.groupby(["team", "date"]).ngroups)

    grouped = starters.groupby(["team", "date"], observed=True)

    quality = grouped.agg(
        squad_quality_goals_per_90=("hist_goals_per_90", "sum"),
        squad_quality_assists_per_90=("hist_assists_per_90", "sum"),
        squad_quality_xg_proxy_per_90=("hist_xg_proxy_per_90", "sum"),
        squad_quality_shots_per_90=("hist_shots_per_90", "sum"),
        starter_avg_experience=("hist_minutes_total", "mean"),
        n_starters=("player", "nunique"),
    ).reset_index()

    # key player flags: top scorer / assister of the SEASON SO FAR (no leakage)
    # For each (team, date) we find the player on the team with the highest
    # cumulative goals/assists BEFORE this date, then check if they started.
    season_rank = (
        p.sort_values(["team", "date"])
            .groupby(["team", "date"], observed=True)
            .apply(_top_player_flags, starter_min=starter_min, include_groups=False)
            .reset_index()
    )

    quality = quality.merge(season_rank, on=["team", "date"], how="left")

    # fillna for flags (early season may have ties / no clear leader)
    for col in ["top_scorer_present", "top_assister_present"]:
        if col in quality.columns:
            quality[col] = quality[col].fillna(0).astype("Int64")

    return quality


def _top_player_flags(group: pd.DataFrame, starter_min: int) -> pd.Series:
    """
    Within one (team, date) group, identify:
      - the player with the highest hist_goals_per_90 weighted by minutes
        (= 'top scorer up to now')
      - the player with the highest hist_assists_per_90 weighted by minutes
      - whether each is in the starting XI for this match

    Returns a Series with two binary flags.
    """
    # weight per-90 by minutes to favour established producers over small samples
    g = group.assign(
        score_score=group["hist_goals_per_90"] * np.log1p(group["hist_minutes_total"]),
        assist_score=group["hist_assists_per_90"] * np.log1p(group["hist_minutes_total"]),
    )
    if g.empty or g["hist_minutes_total"].max() < 90:
        return pd.Series({"top_scorer_present": 0, "top_assister_present": 0})

    # both scores can be all-NaN early in the season (no player has minutes yet)
    if g["score_score"].notna().sum() == 0 or g["assist_score"].notna().sum() == 0:
        return pd.Series({"top_scorer_present": 0, "top_assister_present": 0})

    top_scorer = g.loc[g["score_score"].idxmax()]
    top_assister = g.loc[g["assist_score"].idxmax()]

    return pd.Series({
        "top_scorer_present": int(top_scorer["minutes_this_match"] >= starter_min),
        "top_assister_present": int(top_assister["minutes_this_match"] >= starter_min),
    })


# ─── opponent-aware features (Phase B) ─────────────────────────────────────────

def build_opponent_features(
        gold: pd.DataFrame,
        horizon: int = 5,
) -> pd.DataFrame:
    """
    For each (team, date), describe the strength of the NEXT `horizon` opponents
    using each opponent's pre-match form AS OF the prediction date.

    Why this is leakage-free:
      - The fixture list (which opponent at which date) is published before
        the season starts; using it does NOT use future information.
      - Each opponent's roll5_* stats are looked up via merge_asof with
        direction='backward' and allow_exact_matches=False: we take the
        opponent's most recent row STRICTLY BEFORE our prediction date.
      - No information from matches happening between now and target horizon
        leaks into the features.

    The hard part: pd.merge_asof. Read as "for each (opponent, prediction_date)
    in lookup, find the row in snapshot where opponent matches AND date is
    the latest one strictly before prediction_date".
    """
    cols_to_lookup = [
        "roll5_goals_against",
        "roll5_save_pct",
        "roll5_points",
        "roll5_goal_diff",
        "league_position",
        "cum_points",
    ]
    cols_present = [c for c in cols_to_lookup if c in gold.columns]
    if not cols_present:
        logger.warning("No opponent-lookup columns present in Gold; skipping opp features")
        return gold

    df = gold.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["season", "team", "date"]).reset_index(drop=True)

    # snapshot: each team's pre-match stats indexed by date — what we look up
    snapshot = (
        df[["team", "date"] + cols_present]
            .sort_values("date")
            .rename(columns={"team": "opponent"})
    )

    grp = df.groupby(["team", "season"], observed=True)
    per_step = []

    for k in range(1, horizon + 1):
        # who is the opponent k matches in the future, within this team-season?
        future_opp = grp["opponent"].shift(-k)

        lookup = pd.DataFrame({
            "opponent": future_opp.values,
            "date": df["date"].values,
            "_idx": df.index,
        }).dropna(subset=["opponent"]).sort_values("date")

        # asof merge: opponent's most recent stats STRICTLY BEFORE our date
        merged = pd.merge_asof(
            lookup,
            snapshot,
            on="date",
            by="opponent",
            direction="backward",
            allow_exact_matches=False,  # strictly < prediction date
        )

        # restore original row order
        per_step.append(
            merged.set_index("_idx")[cols_present].reindex(df.index)
        )

    # aggregate across the H opponents — mean is the natural choice
    for col in cols_present:
        stacked = pd.concat([s[col] for s in per_step], axis=1)
        df[f"opp{horizon}_avg_{col}"] = stacked.mean(axis=1)

    logger.info("Opponent features built: %d new columns for horizon=%d",
                len(cols_present), horizon)
    return df


# ─── orchestrator ──────────────────────────────────────────────────────────────

def build_gold(
        team_match: pd.DataFrame,
        player_agg: pd.DataFrame,
        player_history: pd.DataFrame | None = None,
        drop_incomplete_target: bool = True,
) -> pd.DataFrame:
    """
    Full Gold pipeline.

    Args:
        team_match:     Silver team_match
        player_agg:     Silver player_match_agg
        player_history: Silver player_history (optional, enables squad quality features)
        drop_incomplete_target: drop rows where forward target is NaN

    Returns:
        Model-ready DataFrame.
    """
    df = _merge_silver(team_match, player_agg)
    df = _build_context(df)
    df = _build_rolling(df)
    df = _build_target(df)

    # alternative target: momentum_change = next_5_points - (roll5_points * 5)
    # this removes the team-baseline component; a positive value means the
    # team OVER-performs its recent form, negative means form crash
    if TARGET_COL in df.columns and "roll5_points" in df.columns:
        df[MOMENTUM_TARGET_COL] = (
                df[TARGET_COL].astype(float) - (df["roll5_points"].astype(float) * 5)
        )

    # squad quality features from player history (Phase A)
    if player_history is not None and not player_history.empty:
        squad_q = build_squad_quality_features(player_history, team_match)
        df["date"] = pd.to_datetime(df["date"])
        squad_q["date"] = pd.to_datetime(squad_q["date"])
        df = df.merge(squad_q, on=["team", "date"], how="left")
        logger.info("Merged squad quality features: %d new columns",
                    len(squad_q.columns) - 2)
    else:
        logger.info("Skipping squad quality features (no player_history provided)")

    # opponent-aware features (Phase B) — must come AFTER rolling features
    # because it looks up other teams' roll5_* values
    df = build_opponent_features(df, horizon=TARGET_HORIZON)

    n_before = len(df)
    if drop_incomplete_target:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    logger.info("Target built: %d rows, %d dropped (incomplete forward horizon)",
                len(df), n_before - len(df))

    df[TARGET_COL] = df[TARGET_COL].astype("Int64")
    return df


def run_gold(
        silver_dir: str,
        seasons: list[str],
        gold_dir: str = "data/gold",
        save: bool = True,
) -> pd.DataFrame:
    """
    Load Silver for the given seasons, build Gold, optionally persist.
    """
    team_frames, player_frames, history_frames = [], [], []

    for season in seasons:
        sp = Path(silver_dir) / season
        tm = pd.read_parquet(sp / "team_match.parquet")
        pa = pd.read_parquet(sp / "player_match_agg.parquet")
        team_frames.append(tm)
        player_frames.append(pa)

        ph_path = sp / "player_history.parquet"
        if ph_path.exists():
            history_frames.append(pd.read_parquet(ph_path))
        else:
            logger.warning("player_history missing for %s — squad quality will be skipped", season)

    team_match = pd.concat(team_frames, ignore_index=True)
    player_agg = pd.concat(player_frames, ignore_index=True)
    player_history = pd.concat(history_frames, ignore_index=True) if history_frames else None

    gold = build_gold(team_match, player_agg, player_history)

    if save:
        out_dir = Path(gold_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "squad_momentum.parquet"
        gold.to_parquet(out_path, index=False)
        logger.info("Saved Gold -> %s", out_path)

    return gold
