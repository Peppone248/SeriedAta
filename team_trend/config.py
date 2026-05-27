"""
team_trend/config.py — configuration for the Squad Momentum Prediction project.

Target:
    Points accumulated over the next N matchweeks (regression).

Feature groups:
    ATTACKING   → rolling squad xG, shot quality, shot volume
    DEFENSIVE   → rolling PSxG conceded, goalkeeper form, defensive actions
    FORM        → rolling points, goal diff, weighted form
    TACTICAL    → progressive carries, pressing intensity
    CONTEXT     → league position, fixture difficulty, season progress
    AVAILABILITY→ squad rotation, accumulated minutes (future: injury data)

Data sources:
    Bronze / Silver: FBref match reports (scrapers/match_scraper.py)
    Gold:            etl/gold/squad_features.py
"""

from __future__ import annotations

# ─── TARGET ──────────────────────────────────────────────────────────────────
PREDICTION_HORIZON: int = 5    # predict points over next N matchweeks
TARGET_COL:         str = f"next_{PREDICTION_HORIZON}_matchweek_points"

# ─── CATEGORICAL FEATURES ────────────────────────────────────────────────────
CAT_FEATURES: list[str] = ["team"]

# ─── NUMERICAL FEATURES ──────────────────────────────────────────────────────
# All are pre-match rolling means (shift(1) applied in squad_features.py).

ATTACKING_FEATURES: list[str] = [
    "roll_squad_xg",              # rolling mean xG produced
    "roll_xg_per_shot",           # rolling shot quality
    "roll_shot_accuracy",         # rolling shots on target %
    "roll_squad_shots",           # rolling shot volume
    "roll_shots_from_pen_area",   # rolling shots from penalty area
    "roll_avg_shot_distance",     # rolling average shot distance
]

DEFENSIVE_FEATURES: list[str] = [
    "roll_psxg",                  # rolling xG faced by goalkeeper
    "gk_form",                    # rolling PSxG+/- (GK outperformance)
    "roll_goals_against",         # rolling goals conceded
    "roll_total_tackles",         # rolling defensive work rate
    "roll_interceptions",         # rolling interceptions
]

FORM_FEATURES: list[str] = [
    "roll_points",                # rolling points (last N)
    "roll_goal_diff",             # rolling goal differential
    "weighted_form",              # exponentially weighted form
    "form_consistency",           # std of points last 5 (stability)
    "points_trend",               # direction of rolling points
]

TACTICAL_FEATURES: list[str] = [
    "roll_progressive_carries",   # build-up quality
    "roll_dribbles_successful",   # individual quality in transition
    "squad_rotation",             # how much the coach rotates
    "formation_changed",          # tactical change from previous match
]

CONTEXT_FEATURES: list[str] = [
    "league_position",            # current standing
    "opp_league_position",        # next opponent strength
    "points_gap_top4",            # distance from Champions League
    "points_gap_relegation",      # safety margin
    "season_progress",            # matchweek / 38
    "is_home",                    # home / away
    "days_rest",                  # days since last match
]

# ─── COMBINED FEATURE SET ────────────────────────────────────────────────────
NUM_FEATURES: list[str] = (
    ATTACKING_FEATURES
    + DEFENSIVE_FEATURES
    + FORM_FEATURES
    + TACTICAL_FEATURES
    + CONTEXT_FEATURES
)

# ─── SCRAPING ────────────────────────────────────────────────────────────────
SERIE_A_SEASONS: list[str] = [
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]
SERIE_A_COMP_ID: str = "11"

# ─── DATA PATHS ──────────────────────────────────────────────────────────────
RAW_DIR:    str = "data/raw"
BRONZE_DIR: str = "data/bronze"
SILVER_DIR: str = "data/silver"
GOLD_DIR:   str = "data/gold"
