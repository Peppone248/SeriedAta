"""Sanity-check the rescue hypothesis on data we ALREADY have.

This script doesn't need FBref scraping — it operates on the Wyscout
data from Phase 1 that already contains all 10 bridge features. The
purpose is to verify that:

    1. The full bridge produces meaningfully different DefR scores than
       the reduced bridge would on the same data.
    2. The full bridge's predictions correlate LESS with simple
       team-quality indicators (like goals_against alone) than the
       reduced bridge.

If both hold, we have reason to believe the rescue path will work once
n_opp_passes is fetched from FBref. If not, the rescue hypothesis is
probably wrong and the pass-volume scrape would be wasted effort.

Run this BEFORE running fetch_passing_data.py to confirm the rescue
is worth pursuing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

DEFR_DIR = Path(__file__).resolve().parent
BRIDGE_PATH = DEFR_DIR / "output" / "data" / "bridge_regression.json"
DATASET_PATH = DEFR_DIR / "output" / "data" / "bridge_dataset.parquet"

print("=" * 70)
print("RESCUE HYPOTHESIS SANITY CHECK")
print("=" * 70)
print()
print("Question: Does the FULL bridge (with n_opp_passes) produce a different")
print("DefR signal than the REDUCED bridge (without n_opp_passes) would,")
print("on the same Wyscout data?")
print()

if not BRIDGE_PATH.exists() or not DATASET_PATH.exists():
    print(f"ERROR: missing required artifacts. Run run_defr_analysis.py first.")
    exit(1)

# Load
with open(BRIDGE_PATH) as f:
    bridge = json.load(f)
df = pd.read_parquet(DATASET_PATH)

# 1) Apply the FULL bridge to its training data
features_full = bridge["feature_names"]
means_f = np.array([bridge["scaler_means"][f] for f in features_full])
stds_f = np.array([bridge["scaler_stds"][f] for f in features_full])
coefs_f = np.array([bridge["coefficients"][f] for f in features_full])
X_full = df[features_full].values
X_full_std = (X_full - means_f) / stds_f
defr_full = X_full_std @ coefs_f + bridge["intercept"]

# 2) Refit a reduced bridge IN-SAMPLE for comparison
features_red = [f for f in features_full if f not in ("n_opp_passes", "def_pressure_ratio")]
X_red = df[features_red].values
sc_red = StandardScaler()
X_red_std = sc_red.fit_transform(X_red)
y = df["defr_score"].values
m_red = Ridge(alpha=1.0, random_state=42).fit(X_red_std, y)
defr_red = m_red.predict(X_red_std)

# 3) Correlation of each proxy with team-quality indicators in the
#    Wyscout-aggregate data
print("[1] Proxy correlations with team-quality indicators")
print("-" * 70)
quality_indicators = ["goals_against", "shots_against", "poss_pct", "shots_for"]
print(f"{'feature':<25}{'full bridge':>15}{'reduced':>15}{'delta':>10}")
print("-" * 70)
for q in quality_indicators:
    rf = np.corrcoef(defr_full, df[q].values)[0, 1]
    rr = np.corrcoef(defr_red, df[q].values)[0, 1]
    print(f"  {q:<23}{rf:>+15.3f}{rr:>+15.3f}{abs(rf) - abs(rr):>+10.3f}")
print()

# 4) Agreement between the two proxies
agree_corr = np.corrcoef(defr_full, defr_red)[0, 1]
print(f"[2] Correlation between full-bridge and reduced-bridge proxy: {agree_corr:+.3f}")
print()

# 5) Ranking divergence at the season level
df_with = df.copy()
df_with["defr_full_proxy"] = defr_full
df_with["defr_red_proxy"] = defr_red

if "team_name" in df_with.columns:
    season_full = df_with.groupby("team_name")["defr_full_proxy"].mean().sort_values(ascending=False)
    season_red = df_with.groupby("team_name")["defr_red_proxy"].mean().sort_values(ascending=False)
    full_rank = {t: i + 1 for i, t in enumerate(season_full.index)}
    red_rank = {t: i + 1 for i, t in enumerate(season_red.index)}
    rank_div = pd.DataFrame({
        "team": list(full_rank.keys()),
        "full_rank": [full_rank[t] for t in full_rank],
        "reduced_rank": [red_rank.get(t, np.nan) for t in full_rank],
    })
    rank_div["delta"] = rank_div["full_rank"] - rank_div["reduced_rank"]

    print(f"[3] Season ranking divergence — top 5 teams that move most")
    print("-" * 70)
    print(rank_div.reindex(rank_div["delta"].abs().sort_values(ascending=False).index)
          .head(5).to_string(index=False))
    print()

# 6) Verdict
print("[4] Verdict")
print("-" * 70)
max_q_corr_full = max(abs(np.corrcoef(defr_full, df[q].values)[0, 1]) for q in quality_indicators)
max_q_corr_red  = max(abs(np.corrcoef(defr_red,  df[q].values)[0, 1]) for q in quality_indicators)

if max_q_corr_full < max_q_corr_red - 0.05:
    print(f"  ✓ Full bridge decouples from team quality:")
    print(f"    max |corr| with quality indicators is {max_q_corr_full:.2f} (full)")
    print(f"    vs {max_q_corr_red:.2f} (reduced). Difference: {max_q_corr_red - max_q_corr_full:+.2f}")
    print(f"  ✓ Rescue path is worth pursuing.")
elif max_q_corr_full < max_q_corr_red:
    print(f"  ~ Modest improvement: max |corr| with quality is {max_q_corr_full:.2f} (full)")
    print(f"    vs {max_q_corr_red:.2f} (reduced). The rescue MAY help.")
else:
    print(f"  ✗ Full bridge does NOT decouple from team quality:")
    print(f"    max |corr| {max_q_corr_full:.2f} (full) vs {max_q_corr_red:.2f} (reduced)")
    print(f"  ✗ The rescue path is unlikely to add value.")

print(f"\n  Proxy agreement: {agree_corr:+.3f}")
print(f"  ({'high — proxies measure similar things' if agree_corr > 0.7 else 'moderate — proxies diverge meaningfully' if agree_corr > 0.4 else 'low — proxies measure different things'})")
print("=" * 70)
