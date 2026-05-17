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

NUM_FEATURES = [
    # --- feature originali ---
    "xg",
    "xga",
    "poss",
    "sot",
    "shot_accuracy",
    "is_home",
    "strength_points_diff",
    "strength_xg_diff",
    "strength_xga_diff",
    #"finishing_efficiency",
    #"defensive_efficiency",
    "last_5_points",
    "last_5_goal_diff",
    "last_5_xg",
    "xg_trend",
    "points_trend",
    "days_rest",
    "dist",                # distanza media tiro
    "formation_changed",   # cambio modulo rispetto alla gara precedente
    "cum_avg_points",
    "cum_avg_xg",
    "cum_avg_xga",
    "h2h_win_rate",
    #"matches_last_14d",
    "weighted_form",
    "form_consistency"
]

CAT_FEATURES = [
    "team",
    "opponent",
    "venue",
]
