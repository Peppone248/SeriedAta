"""Cleaning helpers."""

from __future__ import annotations

import pandas as pd

from config import CATEGORICAL_COLUMNS, DROP_COLUMNS, NUMERIC_COLUMNS


def load_matches(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    return pd.read_csv(csv_path)


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns and cast dtypes on the working DataFrame.

    This function mutates the given DataFrame and also returns it for chaining.
    """
    df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "attendance" in df.columns:
        df["attendance"] = df["attendance"].astype("Int64")

    if "season" in df.columns:
        df["season"] = df["season"].astype("Int64")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
