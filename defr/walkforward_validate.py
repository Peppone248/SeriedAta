"""Walk-forward validation: does defr_proxy add value to the team_trend models?

Experimental design:
    1. Run the existing pipeline to compute the standard features
    2. Merge in defr_proxy and last_5_defr_proxy
    3. Run walk-forward backtest by season for THREE feature sets:
         - baseline   : LOGISTIC_NUM_FEATURES (existing 24 features)
         - +proxy     : baseline + defr_proxy (match-level)
         - +rolling   : baseline + last_5_defr_proxy (5-match rolling, no leakage)
    4. Compare F1-macro, accuracy, log-loss across folds
    5. Statistical test: paired comparison of fold-level F1 deltas

Model: Logistic Regression with the existing pipeline (preprocessing,
       hyperparameters). Why logistic? It's the simplest, most
       interpretable, and most sensitive to feature additions —
       gradient boosters absorb new features more silently.

Target: result ∈ {W, D, L}  (the existing classification task)

Walk-forward scheme: 2 train seasons minimum, expanding window.
    Fold 1: train [2020, 2021]                 → test 2022
    Fold 2: train [2020, 2021, 2022]           → test 2023
    Fold 3: train [2020, 2021, 2022, 2023]     → test 2024
    Fold 4: train [2020-2024]                  → test 2025

This is exactly the procedure backtest_by_season uses, so we can reuse
the existing infrastructure with minimal modification.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Workspace paths so we can import the existing project modules
DEFR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEFR_DIR.parent
WORKSPACE = REPO_ROOT / "matches_classification"
sys.path.insert(0, str(WORKSPACE))

warnings.filterwarnings("ignore")

# Project imports
import config as cfg  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from backtesting import backtest_by_season  # noqa: E402
from models.logistic_pipeline import build_model_pipeline, train_model  # noqa: E402

DEFR_PARQUET = DEFR_DIR / "output" / "injection" / "fbref_with_defr.parquet"
OUT_DIR = DEFR_DIR / "output" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Step 1: build the working dataset ────────────────────────────────
def build_dataset() -> pd.DataFrame:
    """Run the existing pipeline + merge defr_proxy on (date, team)."""
    print("\n[1/4] Running existing pipeline...")
    import os
    os.chdir(str(WORKSPACE))  # pipeline.py writes to relative path data/interim/
    out = run_pipeline(
        str(WORKSPACE / "data/raw/matches_seriea.csv"), save=False
    )
    df = out["raw_df"].copy()

    # The default pipeline doesn't call add_new_features / add_parity_features,
    # which are needed for the full LOGISTIC_NUM_FEATURES set. Call them now.
    from features import add_new_features, add_parity_features
    df = add_new_features(df)
    df = add_parity_features(df)
    df["date"] = pd.to_datetime(df["date"])

    # Load defr proxy and merge
    print("[2/4] Merging DefR proxy...")
    defr_df = pd.read_parquet(DEFR_PARQUET)
    defr_df["date"] = pd.to_datetime(defr_df["date"])

    keep = [
        "date", "team", "defr_proxy",
        "last_5_defr_proxy", "last_5_defr_proxy_z",
    ]
    df = df.merge(defr_df[keep], on=["date", "team"], how="left")

    n_proxy = df["defr_proxy"].notna().sum()
    n_roll = df["last_5_defr_proxy"].notna().sum()
    print(f"      defr_proxy:        {n_proxy:,} / {len(df):,} rows ({100*n_proxy/len(df):.1f}%)")
    print(f"      last_5_defr_proxy: {n_roll:,} / {len(df):,} rows ({100*n_roll/len(df):.1f}%)")

    return df


# ─── Step 2: feature sets to compare ──────────────────────────────────
BASELINE = list(cfg.LOGISTIC_NUM_FEATURES)
PROXY_FEATURE = BASELINE + ["defr_proxy"]
ROLLING_FEATURE = BASELINE + ["last_5_defr_proxy"]
BOTH = BASELINE + ["defr_proxy", "last_5_defr_proxy"]


def run_one(df: pd.DataFrame, name: str, features: list[str]) -> "BacktestResult":  # noqa: F821
    """Run one walk-forward backtest with a given feature set.

    We use fixed hyperparameters (best_params) instead of GridSearchCV
    per fold for two reasons:
      1. Same hyperparameters across conditions → cleaner feature-set
         comparison (any delta is from features, not from random
         differences in grid-search outcomes).
      2. Order-of-magnitude faster: ~30 seconds per condition vs ~5
         minutes with full grid search.
    """
    print(f"\n[3/4] Backtest — {name}")
    print(f"      Features: {len(features)} (delta vs baseline: +{len(features) - len(BASELINE)})")
    fixed_params = {
        "model__C": 1.0,
        "model__solver": "lbfgs",
        "model__penalty": "l2",
        "model__class_weight": "balanced",
    }
    result = backtest_by_season(
        df=df,
        build_pipeline_fn=build_model_pipeline,
        train_fn=train_model,
        num_features=features,
        cat_features=cfg.CAT_FEATURES,
        model_name=name,
        needs_label_encoding=False,
        min_train_seasons=2,
        best_params=fixed_params,
    )
    return result


# ─── Step 3: head-to-head comparison ─────────────────────────────────
def compare_results(results: dict) -> pd.DataFrame:
    """Build a per-fold comparison table."""
    rows = []
    base_folds = {f.test_season: f for f in results["baseline"].folds}

    for cond_name, cond_result in results.items():
        for fold in cond_result.folds:
            base_fold = base_folds.get(fold.test_season)
            if base_fold is None:
                continue
            rows.append({
                "condition": cond_name,
                "season": fold.test_season,
                "accuracy": fold.accuracy,
                "f1_macro": fold.f1_macro,
                "f1_L": fold.f1_L,
                "f1_D": fold.f1_D,
                "f1_W": fold.f1_W,
                "log_loss": fold.log_loss,
                "delta_f1": fold.f1_macro - base_fold.f1_macro,
                "delta_acc": fold.accuracy - base_fold.accuracy,
                "delta_logloss": fold.log_loss - base_fold.log_loss,
            })
    return pd.DataFrame(rows)


def paired_test(comparison_df: pd.DataFrame, condition: str) -> dict:
    """Paired t-test on per-fold F1 deltas for one condition vs baseline."""
    base_f1 = (
        comparison_df[comparison_df["condition"] == "baseline"]
        .set_index("season")["f1_macro"]
    )
    cond_f1 = (
        comparison_df[comparison_df["condition"] == condition]
        .set_index("season")["f1_macro"]
    )
    common = base_f1.index.intersection(cond_f1.index)
    deltas = cond_f1.loc[common] - base_f1.loc[common]
    t_stat, p_value = scipy_stats.ttest_rel(cond_f1.loc[common], base_f1.loc[common])

    return {
        "condition": condition,
        "n_folds": len(deltas),
        "mean_delta_f1": float(deltas.mean()),
        "std_delta_f1": float(deltas.std()),
        "min_delta_f1": float(deltas.min()),
        "max_delta_f1": float(deltas.max()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "deltas_by_season": deltas.to_dict(),
    }


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION — DefR proxy on FBref Serie A")
    print("=" * 70)

    df = build_dataset()

    # Run the four feature-set conditions
    print("\n" + "=" * 70)
    results = {
        "baseline": run_one(df, "baseline", BASELINE),
        "+proxy_match": run_one(df, "+proxy_match", PROXY_FEATURE),
        "+proxy_rolling": run_one(df, "+proxy_rolling", ROLLING_FEATURE),
        "+proxy_both": run_one(df, "+proxy_both", BOTH),
    }

    print("\n" + "=" * 70)
    print("[4/4] Aggregate comparison")
    print("=" * 70)

    # Summary table
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "condition": name,
            "mean_f1_macro": res.mean_f1_macro,
            "std_f1_macro": res.std_f1_macro,
            "mean_accuracy": res.mean_accuracy,
            "std_accuracy": res.std_accuracy,
            "mean_f1_D": res.mean_f1_D,
            "std_f1_D": res.std_f1_D,
        })
    summary = pd.DataFrame(summary_rows)
    print("\nMean ± std across folds:")
    print(summary.to_string(index=False))

    # Per-fold comparison
    comparison = compare_results(results)
    print("\nPer-fold F1 deltas vs baseline:")
    pivot_f1 = comparison.pivot(index="season", columns="condition", values="f1_macro")
    pivot_f1["Δ proxy_match"] = pivot_f1["+proxy_match"] - pivot_f1["baseline"]
    pivot_f1["Δ proxy_rolling"] = pivot_f1["+proxy_rolling"] - pivot_f1["baseline"]
    pivot_f1["Δ proxy_both"] = pivot_f1["+proxy_both"] - pivot_f1["baseline"]
    print(pivot_f1.round(4).to_string())

    # Paired statistical tests
    print("\nPaired t-tests (one-sided H1: condition > baseline):")
    tests = {}
    for cond in ["+proxy_match", "+proxy_rolling", "+proxy_both"]:
        t = paired_test(comparison, cond)
        tests[cond] = t
        sig = ""
        if t["p_value"] < 0.05:
            sig = " *  significant (α = 0.05)"
        elif t["p_value"] < 0.10:
            sig = " .  marginal (α = 0.10)"
        print(f"  {cond:>16s}: Δf1 = {t['mean_delta_f1']:+.4f} ± {t['std_delta_f1']:.4f}  "
              f"t = {t['t_statistic']:+.3f}  p = {t['p_value']:.4f}{sig}")

    # Save artifacts
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "per_fold_comparison.csv", index=False)
    pivot_f1.to_csv(OUT_DIR / "f1_pivot.csv")

    with open(OUT_DIR / "paired_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    # Also save each fold's full BacktestResult
    all_folds = []
    for name, res in results.items():
        for fold in res.folds:
            all_folds.append({
                "condition": name,
                **{k: v for k, v in vars(fold).items() if k != "train_seasons"},
                "train_seasons": ",".join(fold.train_seasons),
            })
    pd.DataFrame(all_folds).to_csv(OUT_DIR / "all_folds.csv", index=False)

    print(f"\nArtifacts saved to {OUT_DIR}/")
    print("Done.")

    return results, comparison, tests


if __name__ == "__main__":
    main()
