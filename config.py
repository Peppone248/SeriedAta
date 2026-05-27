"""Shared configuration for the SeriedAta project."""
from pathlib import Path

DROP_COLUMNS = ["Unnamed: 0", "notes", "match report"]

NUMERIC_COLUMNS = [
    "gf",
    "ga",
    "xg",
    "xga",
    "poss",
    "attendance",
    "sh",
    "sot",
    "dist",
    "fk",
    "pk",
    "pkatt",
    "season",
]

CATEGORICAL_COLUMNS = ["comp", "round", "day", "venue", "result", "team", "opponent"]

VALID_RESULTS = ["W", "D", "L"]
VALID_VENUES = ["Home", "Away"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RAW_FILE = RAW_DIR / "matches_seriea.csv"

# ─── FEATURE CATEGORIALI (condivise da tutti i modelli) ─────────────────────
CAT_FEATURES: list[str] = ["team", "opponent", "venue"]

# ─── LOGISTIC REGRESSION ────────────────────────────────────────────────────
# Usate da LogisticRegression standalone e da model_comparison.py (split equo).
LOGISTIC_NUM_FEATURES: list[str] = [
    "opp_league_position", "is_home",
    "xg_parity", "position_diff",
    "strength_xg_diff", "points_trend",
    "league_position", "formation_changed",
    "weighted_form",   # borderline ma con CI che tocca lo zero
]

# ─── XGBOOST (set esteso) ────────────────────────────────────────────────────
XGBOOST_NUM_FEATURES: list[str] = [
        "opp_league_position", "is_home",
        "xg_parity", "position_diff",
        "strength_xg_diff", "points_trend",
        "league_position", "formation_changed",
        "weighted_form",  # borderline ma con CI che tocca lo zero
    ]

# ─── LIGHTGBM ────────────────────────────────────────────────────────────────
# Stesso set esteso di XGBoost: confronto diretto tra i due boosting models.
# Definito separatamente per permettere variazioni future indipendenti.
LGBM_NUM_FEATURES: list[str] = [
        "opp_league_position", "is_home",
        "xg_parity", "position_diff",
        "strength_xg_diff", "points_trend",
        "league_position", "formation_changed",
        "weighted_form",  # borderline ma con CI che tocca lo zero
    ]

# ─── REGRESSIONE ────────────────────────────────────────────────────────────
REGRESSION_FEATURES: list[str] = [
    "xg", "xga", "poss", "sot", "shot_accuracy", "is_home",
    "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
    "finishing_efficiency", "defensive_efficiency",
    "last_5_points", "last_5_goal_diff", "last_5_xg",
    "xg_trend", "days_rest",
]
REGRESSION_TARGET: str = "goal_diff"

# ─── ALIAS — retro-compatibilità ─────────────────────────────────────────────
NUM_FEATURES = LOGISTIC_NUM_FEATURES