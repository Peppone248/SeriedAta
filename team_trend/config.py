"""
config.py — feature and target configuration.

Feature sets defined here based on empirical ablation results.
The ablation (models/ablation.py) tested each group's contribution by
removing it and measuring the MAE delta. Groups are classified as:

  USEFUL    (delta > +0.005)  → removing hurts the model → keep
  NEUTRAL   (|delta| <= 0.005) → no measurable effect → drop for parsimony
  HARMFUL   (delta < -0.005)  → removing HELPS the model → drop

Two feature sets are maintained:
  FEATURES_FULL   → all 41 features (for ablation reruns and debugging)
  FEATURES_CLEAN  → ablation-pruned set (for production and final results)

The target column is also configurable here.
"""

# ─── targets ───────────────────────────────────────────────────────────────────

TARGET_POINTS   = "next_5_matchweek_points"
TARGET_MOMENTUM = "momentum_change_5"

# default target
TARGET = TARGET_POINTS


# ─── feature groups with ablation classification ───────────────────────────────

# USEFUL: removing these groups increased test MAE
STANDINGS = [
    "cum_points", "cum_goal_diff", "league_position", "season_progress",
]
POSSESSION = ["roll5_possession"]
SHOOTING = [
    "roll5_shots", "roll5_shots_on_target",
    "roll5_shots_on_target_pct", "roll5_goals_per_shot",
]
KEEPER = ["roll5_save_pct", "roll5_saves"]
OPPONENT = [
    "opp5_avg_roll5_goals_against",
    "opp5_avg_roll5_save_pct",
    "opp5_avg_roll5_points",
    "opp5_avg_roll5_goal_diff",
    "opp5_avg_league_position",
    "opp5_avg_cum_points",
]
DEFENSE = ["roll5_sum_tackles_won", "roll5_sum_interceptions"]
DISCIPLINE = ["roll5_sum_yellow_cards", "roll5_sum_fouls"]
CONTEXT = ["is_home", "days_rest"]
CORE_FORM = [
    "roll5_points", "roll5_goal_diff",
    "roll5_goals_for", "roll5_goals_against",
]

# NEUTRAL: no measurable effect — dropped for parsimony
ROTATION = [
    "roll5_players_used", "roll5_starters_used",
    "roll5_minutes_std", "roll5_squad_age_mean",
]

# HARMFUL: removing these improved the model — dropped
VOLATILITY = ["roll5_points_std", "roll5_goal_diff_std"]
SQUAD_QUALITY = [
    "squad_quality_goals_per_90",
    "squad_quality_assists_per_90",
    "squad_quality_xg_proxy_per_90",
    "squad_quality_shots_per_90",
    "starter_avg_experience",
    "n_starters",
    "top_scorer_present",
    "top_assister_present",
]


# ─── assembled feature sets ────────────────────────────────────────────────────

FEATURES_FULL = (
    CORE_FORM + POSSESSION + SHOOTING + KEEPER + ROTATION
    + DEFENSE + DISCIPLINE + VOLATILITY + CONTEXT + STANDINGS
    + SQUAD_QUALITY + OPPONENT
)

# production feature set: only USEFUL groups retained
# drops: ROTATION (neutral), VOLATILITY (harmful), SQUAD_QUALITY (harmful)
FEATURES_CLEAN = (
    CORE_FORM + POSSESSION + SHOOTING + KEEPER
    + DEFENSE + DISCIPLINE + CONTEXT + STANDINGS + OPPONENT
)

# default
FEATURES = FEATURES_CLEAN