"""Refit the bridge regression using ONLY features available from FBref.

FBref via soccerdata exposes per-match: gf, ga, xg, xga, poss, sh, sot,
dist, fk, pk, pkatt — but NOT pass counts. The original bridge used
n_opp_passes (largest coef = −50) and def_pressure_ratio (= shots/passes).
Both must be dropped for a faithful transfer.

This script:
  1. Loads the Wyscout-derived bridge dataset
  2. Refits Ridge with only FBref-compatible features
  3. Reports the R² delta vs the original bridge
  4. Saves bridge_regression_fbref.json with the new coefficients
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

DEFR_DIR = Path(__file__).resolve().parent
OUT_DIR = DEFR_DIR / "output" / "injection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the merged bridge dataset (DefR + Wyscout aggregates + derived features)
merged = pd.read_parquet(DEFR_DIR / "output" / "data" / "bridge_dataset.parquet")

# Convert Wyscout avg_opp_shot_x → FBref-style shot distance proxy
# (in Wyscout, x ∈ [0, 100] with attacking goal at 100; FBref dist is in
# yards from goal. A monotone proxy is dist_proxy = 100 − avg_opp_shot_x.)
# Note: we already have shot_distance_proxy from the original bridge,
# computed the same way, so we just reuse it.

# FBref-compatible features (must be derivable from FBref columns)
FBREF_FEATURES = [
    "shots_against",       # ←→ opponent sh
    "sot_against",         # ←→ opponent sot
    "poss_pct",            # ←→ poss
    "goals_against",       # ←→ ga
    "shots_for",           # ←→ sh
    "sot_ratio_against",   # derived: opp_sot / opp_sh
    "shot_distance_proxy", # ←→ derived from dist
]

X = merged[FBREF_FEATURES].values
y = merged["defr_score"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0, random_state=42)
cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
cv_mae = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# Compare against the original (10-feature) bridge from the previous step
with open(DEFR_DIR / "output" / "data" / "bridge_regression.json") as f:
    original = json.load(f)

print("=" * 64)
print("BRIDGE REFIT — FBref-only features")
print("=" * 64)
print(f"\nFeatures dropped (no FBref equivalent):")
print(f"  - n_opp_passes        (original coef: {original['coefficients']['n_opp_passes']:+.3f})")
print(f"  - def_pressure_ratio  (original coef: {original['coefficients']['def_pressure_ratio']:+.3f})")
print(f"  - avg_opp_shot_x      (redundant with shot_distance_proxy)")
print()
print(f"{'metric':<20} {'original (10f)':>16} {'refit (7f)':>16} {'delta':>10}")
print("-" * 64)
orig_r2, orig_mae = original["cv_r2_mean"], original["cv_mae_mean"]
new_r2, new_mae = float(cv_r2.mean()), float(-cv_mae.mean())
print(f"{'CV R²':<20} {orig_r2:>16.4f} {new_r2:>16.4f} {new_r2 - orig_r2:>+10.4f}")
print(f"{'CV MAE':<20} {orig_mae:>16.4f} {new_mae:>16.4f} {new_mae - orig_mae:>+10.4f}")
print(f"{'CV R² std':<20} {original['cv_r2_std']:>16.4f} {float(cv_r2.std()):>16.4f}")

print(f"\nFull-data R²:   {r2_score(y, y_pred):.4f}")
print(f"Full-data MAE:  {mean_absolute_error(y, y_pred):.4f}")
print(f"N samples:      {len(y)}")

print(f"\nNew coefficient ranking (standardized):")
for name, coef in sorted(
    zip(FBREF_FEATURES, model.coef_),
    key=lambda x: abs(x[1]), reverse=True
):
    sign = "+" if coef > 0 else "−"
    print(f"  {sign} {name:<24}  coef = {coef:+.4f}")

# Save the new bridge
result = {
    "feature_names": FBREF_FEATURES,
    "coefficients": {n: float(c) for n, c in zip(FBREF_FEATURES, model.coef_)},
    "intercept": float(model.intercept_),
    "scaler_means": {n: float(m) for n, m in zip(FBREF_FEATURES, scaler.mean_)},
    "scaler_stds": {n: float(s) for n, s in zip(FBREF_FEATURES, scaler.scale_)},
    "cv_r2_mean": float(cv_r2.mean()),
    "cv_r2_std": float(cv_r2.std()),
    "cv_r2_folds": cv_r2.tolist(),
    "cv_mae_mean": float(-cv_mae.mean()),
    "cv_mae_std": float(cv_mae.std()),
    "full_r2": float(r2_score(y, y_pred)),
    "full_mae": float(mean_absolute_error(y, y_pred)),
    "n_samples": len(y),
    "vs_original_r2_delta": float(cv_r2.mean()) - orig_r2,
    "notes": "FBref-compatible refit. Drops n_opp_passes and def_pressure_ratio "
             "(not exposed by FBref). Designed for direct application to "
             "soccerdata-fetched FBref data via the apply_bridge_to_fbref formula.",
}
with open(OUT_DIR / "bridge_regression_fbref.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT_DIR}/bridge_regression_fbref.json")
