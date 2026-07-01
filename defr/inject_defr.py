"""Inject defr_proxy into the FBref Serie A dataset.

For each team-match row:
    1. Find the opponent's row in the same match (same date, swapped team)
    2. Extract the opponent's shot stats (opp_sh, opp_sot, opp_dist)
    3. Compute the 7 FBref-compatible features
    4. Standardize using the bridge's saved (μ, σ) parameters
    5. Apply the bridge coefficients to produce defr_proxy
    6. Add rolling 5-match version with .shift(1) for temporal hygiene
       (no future information leaks into a row's own features)

The rolling version mirrors the existing feature engineering style in
features.py (e.g., last_5_points, last_5_xg). It's the version most
likely to add predictive value: a team's recent defensive style is
more stable than any single match's noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEFR_DIR.parent
BRIDGE_PATH = DEFR_DIR / "output" / "injection" / "bridge_regression_fbref.json"
FBREF_CSV = REPO_ROOT / "matches_classification" / "data" / "raw" / "matches_seriea.csv"
OUT_PATH = DEFR_DIR / "output" / "injection" / "fbref_with_defr.parquet"


def load_bridge() -> dict:
    """Load the FBref-compatible bridge regression coefficients."""
    with open(BRIDGE_PATH) as f:
        return json.load(f)


def pair_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, attach the opponent's match stats from the mirror row.

    Each Serie A match appears twice in the data — once per team. We
    pair them on (date, team, opponent) such that for row (date=D,
    team=A, opp=B), we attach the columns from the row (date=D, team=B,
    opp=A). The opponent's `sh`, `sot`, `dist` become this row's
    `opp_sh`, `opp_sot`, `opp_dist`.
    """
    # Build a lookup on (date, team) → row to fetch opponent stats
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cols_to_pair = ["sh", "sot", "dist", "xg", "poss"]
    opp_lookup = df.set_index(["date", "team"])[cols_to_pair]

    # For each row, look up the opponent's stats by (date, opponent)
    idx = pd.MultiIndex.from_arrays(
        [df["date"], df["opponent"]], names=["date", "team"]
    )
    opp_data = opp_lookup.reindex(idx).reset_index(drop=True)
    opp_data.columns = [f"opp_{c}" for c in cols_to_pair]
    return pd.concat([df.reset_index(drop=True), opp_data], axis=1)


def compute_fbref_bridge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 features used by the FBref bridge.

    Column-to-feature mapping:
        shots_against       ←  opp_sh
        sot_against         ←  opp_sot
        poss_pct            ←  poss
        goals_against       ←  ga
        shots_for           ←  sh
        sot_ratio_against   ←  opp_sot / opp_sh (with 0 guard)
        shot_distance_proxy ←  opp_dist  (FBref: yards from goal; high = far)

    Note on shot_distance_proxy:
        In the Wyscout bridge, this feature was computed as
        100 − avg_opp_shot_x, where avg_opp_shot_x ∈ [0, 100] with 100
        being the attacking goal. That formula yields HIGH values for
        shots taken FAR from goal — same direction as FBref's `dist`.
        So we can plug FBref's dist directly (it's roughly in yards;
        the bridge's standardisation absorbs scale differences).
    """
    df = df.copy()
    df["shots_against"] = df["opp_sh"]
    df["sot_against"] = df["opp_sot"]
    df["poss_pct"] = df["poss"]
    df["goals_against"] = df["ga"]
    df["shots_for"] = df["sh"]
    df["sot_ratio_against"] = np.where(
        df["opp_sh"] > 0, df["opp_sot"] / df["opp_sh"], 0.0
    )
    df["shot_distance_proxy"] = df["opp_dist"]
    return df


def apply_bridge(df: pd.DataFrame, bridge: dict) -> pd.Series:
    """Apply the bridge regression formula: defr_proxy = β·x_std + α."""
    features = bridge["feature_names"]
    means = np.array([bridge["scaler_means"][f] for f in features])
    stds = np.array([bridge["scaler_stds"][f] for f in features])
    coefs = np.array([bridge["coefficients"][f] for f in features])
    intercept = bridge["intercept"]

    X = df[features].values.astype(float)
    X_std = (X - means) / stds
    defr_proxy = X_std @ coefs + intercept
    return pd.Series(defr_proxy, index=df.index, name="defr_proxy")


def add_rolling_defr(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5-match rolling DefR proxy with .shift(1) to avoid leakage.

    This mirrors the style in features.py:
        last_5_points = groupby(team)[points].shift(1).rolling(5).mean()
    The shift ensures a row's own match doesn't contribute to its
    feature value — no future information leakage.
    """
    df = df.sort_values(["team", "date"]).copy()

    df["last_5_defr_proxy"] = (
        df.groupby("team")["defr_proxy"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    )

    # Z-score within the same window (standardized momentum)
    df["last_5_defr_proxy_z"] = (
        df.groupby("team")["defr_proxy"]
        .transform(
            lambda x: (
                (x.shift(1).rolling(5, min_periods=2).mean()
                 - x.shift(1).expanding(min_periods=5).mean())
                / x.shift(1).expanding(min_periods=5).std().replace(0, np.nan)
            )
        )
    )
    return df


def main():
    print("=" * 64)
    print("DefR PROXY INJECTION — FBref Serie A (2020–2025)")
    print("=" * 64)

    print(f"\n[1] Loading bridge from {BRIDGE_PATH.name}")
    bridge = load_bridge()
    print(f"    R² (CV) = {bridge['cv_r2_mean']:.3f}  "
          f"(reduced from 0.589 by dropping pass-count features)")
    print(f"    Features: {len(bridge['feature_names'])}")

    print(f"\n[2] Loading FBref CSV from {FBREF_CSV.name}")
    raw = pd.read_csv(FBREF_CSV)
    print(f"    {len(raw):,} team-match rows, "
          f"{raw['season'].nunique()} seasons, "
          f"{raw['team'].nunique()} teams")

    print(f"\n[3] Pairing opponent stats from mirror rows")
    df = pair_opponent_stats(raw)
    opp_missing = df[["opp_sh", "opp_sot", "opp_dist"]].isna().any(axis=1).sum()
    print(f"    Rows with missing opponent stats: {opp_missing} "
          f"({100 * opp_missing / len(df):.1f}%)")

    print(f"\n[4] Computing the 7 bridge features in FBref units")
    df = compute_fbref_bridge_features(df)

    print(f"\n[5] Applying bridge → defr_proxy")
    df["defr_proxy"] = apply_bridge(df, bridge)

    valid = df["defr_proxy"].dropna()
    print(f"    defr_proxy valid rows: {len(valid):,} / {len(df):,}")
    print(f"    Range: [{valid.min():+.2f}, {valid.max():+.2f}]")
    print(f"    Mean:  {valid.mean():+.2f}")
    print(f"    Std:   {valid.std():.2f}")

    print(f"\n[6] Computing 5-match rolling DefR (with .shift(1))")
    df = add_rolling_defr(df)
    rolling_valid = df["last_5_defr_proxy"].dropna()
    print(f"    last_5_defr_proxy valid rows: {len(rolling_valid):,}")

    # ── Sanity check: do top/bottom 2024/25 teams match expectations? ────
    print(f"\n[7] Sanity check — season DefR rankings (2024 season)")
    s24 = (
        df[df["season"] == 2024]
        .groupby("team")["defr_proxy"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(f"  Top 5:")
    for team, row in s24.head(5).iterrows():
        print(f"    {team:25s}  mean={row['mean']:+6.2f}  std={row['std']:5.2f}  n={int(row['count'])}")
    print(f"  Bottom 5:")
    for team, row in s24.tail(5).iterrows():
        print(f"    {team:25s}  mean={row['mean']:+6.2f}  std={row['std']:5.2f}  n={int(row['count'])}")

    # Save
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n[8] Saved to {OUT_PATH}")
    print(f"    Columns added: defr_proxy, last_5_defr_proxy, last_5_defr_proxy_z")
    print(f"    + 7 bridge input features and opp_* mirror columns")

    return df


if __name__ == "__main__":
    main()
