"""Refit the bridge using defensive action features from Wyscout events.

This is the step between Phase 1 (run_defr_analysis.py) and the
injection (inject_defr_full.py). It:

    1. Reads the 380 Wyscout match JSONs (from Phase 1 download)
    2. Counts tackles, clearances, fouls, crosses, aerials per team per match
    3. Merges these into the bridge dataset
    4. Fits Ridge regression: defr_score ~ 7 standard + 10 defensive features
    5. Saves bridge_regression_rescue.json

Prerequisites:
    - Phase 1 must have completed (data/wyscout_matches/ has 380 JSONs,
      output/data/bridge_dataset.parquet exists)

Run order:
    1. python run_defr_analysis.py         # Phase 1
    2. python refit_bridge_rescue.py       # THIS SCRIPT
    3. python fetch_defensive_data.py      # local: fetch FBref misc stats
    4. python inject_defr_full.py          # apply rescue bridge
    5. python walkforward_validate_full.py # validate
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
MATCH_DIR = DEFR_DIR / "data" / "wyscout_matches"
BRIDGE_DS_PATH = DEFR_DIR / "output" / "data" / "bridge_dataset.parquet"
ORIG_BRIDGE_PATH = DEFR_DIR / "output" / "data" / "bridge_regression.json"
OUT_PATH = DEFR_DIR / "output" / "injection" / "bridge_regression_rescue.json"

TAG_ACCURATE = 1801


def compute_defensive_stats_from_wyscout() -> pd.DataFrame:
    match_files = sorted(MATCH_DIR.glob("*.json"))
    if not match_files:
        raise FileNotFoundError(
            f"No match JSONs found in {MATCH_DIR}. "
            "Run run_defr_analysis.py (Phase 1) first."
        )
    print(f"  Reading {len(match_files)} match files...")

    rows = []
    for fp in match_files:
        with open(fp) as f:
            data = json.load(f)
        mid = int(fp.stem)
        teams = {int(tid): info["name"] for tid, info in data.get("teams", {}).items()}

        for tid in teams:
            evs = [e for e in data["events"] if e["teamId"] == tid]

            def has_tag(ev, tag_id):
                return tag_id in {t["id"] for t in ev.get("tags", [])}

            tackles_won = sum(
                1 for e in evs
                if e["eventName"] == "Duel"
                and e.get("subEventName") == "Ground defending duel"
                and has_tag(e, TAG_ACCURATE)
            )
            clearances = sum(
                1 for e in evs
                if e["eventName"] == "Others on the ball"
                and e.get("subEventName") == "Clearance"
            )
            fouls = sum(
                1 for e in evs
                if e["eventName"] == "Foul"
                and e.get("subEventName") == "Foul"
            )
            crosses = sum(
                1 for e in evs
                if e["eventName"] == "Pass"
                and e.get("subEventName") == "Cross"
            )
            aerials = sum(
                1 for e in evs
                if e["eventName"] == "Duel"
                and e.get("subEventName") == "Air duel"
            )

            rows.append({
                "match_id": mid, "team_id": tid,
                "tklw": tackles_won, "intrcpt": clearances,
                "fls": fouls, "crs": crosses, "aerial": aerials,
            })

    return pd.DataFrame(rows)


def pair_opponent_stats(def_stats: pd.DataFrame) -> pd.DataFrame:
    team_pairs = {}
    for fp in sorted(MATCH_DIR.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        tids = [int(t) for t in d.get("teams", {})]
        if len(tids) == 2:
            team_pairs[int(fp.stem)] = tids

    def get_opp(row):
        p = team_pairs.get(row["match_id"], [])
        if len(p) != 2:
            return None
        return p[1] if row["team_id"] == p[0] else p[0]

    def_stats = def_stats.copy()
    def_stats["opp_team_id"] = def_stats.apply(get_opp, axis=1)

    opp = def_stats[["match_id", "team_id", "tklw", "intrcpt", "fls", "crs", "aerial"]].copy()
    opp.columns = ["match_id", "opp_team_id"] + [f"opp_{c}" for c in ["tklw", "intrcpt", "fls", "crs", "aerial"]]

    return def_stats.merge(opp, on=["match_id", "opp_team_id"], how="left")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("BRIDGE REFIT — rescue version with defensive action features")
    print("=" * 64)

    if not BRIDGE_DS_PATH.exists():
        raise FileNotFoundError(
            f"{BRIDGE_DS_PATH} not found. Run run_defr_analysis.py first."
        )
    bridge_ds = pd.read_parquet(BRIDGE_DS_PATH)

    with open(ORIG_BRIDGE_PATH) as f:
        orig_bridge = json.load(f)

    print(f"\n[1] Computing defensive stats from Wyscout events...")
    def_stats = compute_defensive_stats_from_wyscout()
    print(f"    {len(def_stats)} team-match rows")

    print(f"[2] Pairing opponent stats...")
    def_merged = pair_opponent_stats(def_stats)

    print(f"[3] Enriching bridge dataset...")
    def_cols = ["match_id", "team_id", "tklw", "intrcpt", "fls", "crs", "aerial",
                "opp_tklw", "opp_intrcpt", "opp_fls", "opp_crs", "opp_aerial"]
    bridge_enriched = bridge_ds.merge(def_merged[def_cols], on=["match_id", "team_id"], how="left")

    y = bridge_enriched["defr_score"].values

    feats_orig = orig_bridge["feature_names"]
    feats_reduced = [f for f in feats_orig
                     if f not in ("n_opp_passes", "def_pressure_ratio", "avg_opp_shot_x")]
    defensive_cols = ["tklw", "intrcpt", "fls", "crs", "aerial",
                      "opp_tklw", "opp_intrcpt", "opp_fls", "opp_crs", "opp_aerial"]
    feats_rescue = feats_reduced + defensive_cols

    print(f"\n[4] Comparing bridge variants")
    print(f"{'variant':<35} {'feats':>6} {'CV R2':>8} {'Full R2':>8}")
    print("-" * 62)

    for name, feats in [
        ("A) Original (pass-based, 10 feat)", feats_orig),
        ("B) FBref-only (7 feat)", feats_reduced),
        ("C) Rescue (FBref + defensive, 17)", feats_rescue),
    ]:
        X = bridge_enriched[feats].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = Ridge(alpha=1.0, random_state=42)
        cv = cross_val_score(m, Xs, y, cv=5, scoring="r2")
        m.fit(Xs, y)
        r2 = r2_score(y, m.predict(Xs))
        print(f"  {name:<33} {len(feats):>6} {cv.mean():>8.4f} {r2:>8.4f}")

    print(f"\n[5] Fitting and saving rescue bridge...")
    X_rescue = bridge_enriched[feats_rescue].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_rescue)
    model = Ridge(alpha=1.0, random_state=42)
    cv_r2 = cross_val_score(model, X_std, y, cv=5, scoring="r2")
    model.fit(X_std, y)

    pred = model.predict(X_std)
    print(f"\n    Quality correlations:")
    for q in ["goals_against", "poss_pct", "shots_against"]:
        r = np.corrcoef(pred, bridge_enriched[q].values)[0, 1]
        print(f"      corr(proxy, {q:18s}) = {r:+.3f}")

    result = {
        "feature_names": feats_rescue,
        "coefficients": {n: float(c) for n, c in zip(feats_rescue, model.coef_)},
        "intercept": float(model.intercept_),
        "scaler_means": {n: float(m) for n, m in zip(feats_rescue, scaler.mean_)},
        "scaler_stds": {n: float(s) for n, s in zip(feats_rescue, scaler.scale_)},
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_r2_folds": cv_r2.tolist(),
        "n_samples": int(len(y)),
        "feature_ranking": sorted(
            [{"name": n, "coef": float(c), "abs_coef": float(abs(c))}
             for n, c in zip(feats_rescue, model.coef_)],
            key=lambda x: x["abs_coef"], reverse=True,
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n    Saved: {OUT_PATH}")
    print(f"    CV R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"\n    Top features:")
    for feat in result["feature_ranking"][:7]:
        sign = "+" if feat["coef"] > 0 else "-"
        print(f"      {sign} {feat['name']:<25} |coef|={feat['abs_coef']:.3f}")

    print(f"\n{'='*64}")
    print(f"Done. Next: python fetch_defensive_data.py (locally)")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
