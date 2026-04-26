"""Shared configuration for the Serie A project."""

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
