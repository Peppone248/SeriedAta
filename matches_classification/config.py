"""
config.py — single source of truth for feature lists and constants.

All pipeline modules import from here. No feature list is defined inline.
"""

# ─── SHARED CATEGORICAL FEATURES ─────────────────────────────────────────────
CAT_FEATURES: list[str] = ["team", "opponent", "venue"]

# ─── LOGISTIC REGRESSION ─────────────────────────────────────────────────────
# Used by the standalone Logistic pipeline and by model_comparison.py
# (same split for fair cross-model comparison).
LOGISTIC_NUM_FEATURES: list[str] = [
    "is_home",
    "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    "last_5_points", "last_5_goal_diff", "last_5_xg",
    "xg_trend", "points_trend", "days_rest",
    "cum_avg_points", "cum_avg_xg", "cum_avg_xga",
    "formation_changed", "weighted_form",
]

# ─── XGBOOST ─────────────────────────────────────────────────────────────────
# Extended feature set vs logistic:
#   dist             → average shot distance
#   h2h_win_rate     → historical head-to-head win rate
#   form_consistency → std of points over last 5 matches
#   parity features  → relative balance between the two squads
#   standings        → league position and contextual pressure
#   opp-adjusted     → form split by opponent quality tier
XGBOOST_NUM_FEATURES: list[str] = [
    "is_home",
    "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    "last_5_points", "last_5_goal_diff", "last_5_xg",
    "xg_trend", "points_trend", "days_rest",
    "dist", "formation_changed",
    "cum_avg_points", "cum_avg_xg", "cum_avg_xga",
    "h2h_win_rate", "weighted_form", "form_consistency",
    # parity features
    "strength_parity", "xg_parity", "form_parity",
    "h2h_draw_rate", "both_defensive",
    # standings + pressure
    "league_position", "opp_league_position",
    "points_gap_top4", "points_gap_relegation",
    "position_diff", "season_progress",
    "is_top_half", "is_relegation_zone",
    # opponent-adjusted form
    "form_vs_strong", "form_vs_weak",
    "xg_vs_strong",   "xg_vs_weak",
    "big_game_delta",
]

# ─── LIGHTGBM ─────────────────────────────────────────────────────────────────
# Same extended set as XGBoost for direct comparison between boosting models.
# Defined separately to allow independent future changes.
LGBM_NUM_FEATURES: list[str] = [
    "is_home",
    "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    "last_5_points", "last_5_goal_diff", "last_5_xg",
    "xg_trend", "points_trend", "days_rest",
    "dist", "formation_changed",
    "cum_avg_points", "cum_avg_xg", "cum_avg_xga",
    "h2h_win_rate", "weighted_form", "form_consistency",
    # parity features
    "strength_parity", "xg_parity", "form_parity",
    "h2h_draw_rate", "both_defensive",
    # standings + pressure
    "league_position", "opp_league_position",
    "points_gap_top4", "points_gap_relegation",
    "position_diff", "season_progress",
    "is_top_half", "is_relegation_zone",
    # opponent-adjusted form
    "form_vs_strong", "form_vs_weak",
    "xg_vs_strong",   "xg_vs_weak",
    "big_game_delta",
]

# ─── REGRESSION ──────────────────────────────────────────────────────────────
REGRESSION_FEATURES: list[str] = [
    "xg", "xga", "poss", "sot", "shot_accuracy", "is_home",
    "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    "finishing_efficiency", "defensive_efficiency",
    "last_5_points", "last_5_goal_diff", "last_5_xg",
    "xg_trend", "points_trend", "days_rest",
]
REGRESSION_TARGET: str = "goal_diff"

# ─── BACKWARDS-COMPATIBILITY ALIASES ─────────────────────────────────────────
# model_comparison.py imports NUM_FEATURES / CAT_FEATURES.
# Uses logistic features so all models train on identical features (fair split).
NUM_FEATURES = LOGISTIC_NUM_FEATURES