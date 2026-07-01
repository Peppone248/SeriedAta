"""Walk-forward validation: full-bridge DefR vs reduced + baseline.

Tests three feature sets head-to-head:
    1. baseline             — existing 23 LOGISTIC_NUM_FEATURES
    2. +reduced_rolling     — baseline + last_5_defr_proxy
                              (FBref-only bridge, R² = 0.14)
    3. +full_rolling        — baseline + last_5_defr_proxy_full
                              (full 10-feature bridge, R² = 0.59)

Hypothesis: the full bridge should produce a proxy with LESS correlation
to existing strength features (because it captures defensive style, not
team quality), and therefore some marginal F1 lift on the walk-forward
test. If the full-bridge proxy is ALSO redundant, that's a more decisive
negative finding — it would mean the spatial defensive signal genuinely
doesn't add to the W/D/L prediction task at the team level.

Same methodology as walkforward_validate.py:
    - 4 seasonal folds (test on 2022 → 2023 → 2024 → 2025)
    - Logistic Regression with fixed hyperparameters
    - Paired t-tests on per-fold F1 deltas
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEFR_DIR.parent
WORKSPACE = REPO_ROOT / "matches_classification"
sys.path.insert(0, str(WORKSPACE))

warnings.filterwarnings("ignore")

import config as cfg  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from backtesting import backtest_by_season  # noqa: E402
from models.logistic_pipeline import build_model_pipeline, train_model  # noqa: E402

REDUCED_PARQUET = DEFR_DIR / "output" / "injection" / "fbref_with_defr.parquet"
FULL_PARQUET = DEFR_DIR / "output" / "injection" / "fbref_with_defr_full.parquet"
OUT_DIR = DEFR_DIR / "output" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> pd.DataFrame:
    """Run the pipeline + merge BOTH proxies (reduced and full) onto rows."""
    print("\n[1/4] Running the matches_classification pipeline...")
    import os
    os.chdir(str(WORKSPACE))
    out = run_pipeline(str(WORKSPACE / "data/raw/matches_seriea.csv"), save=False)
    df = out["raw_df"].copy()

    from features import add_new_features, add_parity_features
    df = add_new_features(df)
    df = add_parity_features(df)
    df["date"] = pd.to_datetime(df["date"])

    print("[2/4] Merging reduced-bridge proxy...")
    reduced = pd.read_parquet(REDUCED_PARQUET)
    reduced["date"] = pd.to_datetime(reduced["date"])
    df = df.merge(
        reduced[["date", "team", "last_5_defr_proxy"]],
        on=["date", "team"], how="left",
    )
    n_reduced = df["last_5_defr_proxy"].notna().sum()
    print(f"      last_5_defr_proxy: {n_reduced:,} / {len(df):,} ({100*n_reduced/len(df):.1f}%)")

    print("[3/4] Merging full-bridge proxy...")
    if not FULL_PARQUET.exists():
        raise FileNotFoundError(
            f"{FULL_PARQUET} not found. Run inject_defr_full.py first."
        )
    full = pd.read_parquet(FULL_PARQUET)
    full["date"] = pd.to_datetime(full["date"])
    df = df.merge(
        full[["date", "team", "defr_proxy_full", "last_5_defr_proxy_full"]],
        on=["date", "team"], how="left",
    )
    n_full = df["last_5_defr_proxy_full"].notna().sum()
    print(f"      last_5_defr_proxy_full: {n_full:,} / {len(df):,} ({100*n_full/len(df):.1f}%)")

    return df


# ─── feature sets ─────────────────────────────────────────────────────
BASELINE = list(cfg.LOGISTIC_NUM_FEATURES)
PLUS_REDUCED = BASELINE + ["last_5_defr_proxy"]
PLUS_FULL = BASELINE + ["last_5_defr_proxy_full"]
PLUS_BOTH = BASELINE + ["last_5_defr_proxy", "last_5_defr_proxy_full"]


def run_one(df: pd.DataFrame, name: str, features: list[str]):
    """Run one walk-forward backtest with fixed hyperparameters."""
    print(f"  ── {name} ({len(features)} features)")
    fixed_params = {
        "model__C": 1.0,
        "model__solver": "lbfgs",
        "model__penalty": "l2",
        "model__class_weight": "balanced",
    }
    return backtest_by_season(
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


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of both proxies with existing strength/form features."""
    key_feats = [
        "cum_avg_points", "cum_avg_xg", "cum_avg_xga",
        "strength_points_diff", "strength_xga_diff",
        "weighted_form", "last_5_points", "last_5_xg",
        "last_5_goal_diff",
    ]
    sub = df.dropna(subset=key_feats + ["last_5_defr_proxy", "last_5_defr_proxy_full"])
    out = pd.DataFrame({
        "feature": key_feats,
        "corr_reduced": [sub[f].corr(sub["last_5_defr_proxy"]) for f in key_feats],
        "corr_full": [sub[f].corr(sub["last_5_defr_proxy_full"]) for f in key_feats],
    })
    out["abs_delta"] = (out["corr_full"].abs() - out["corr_reduced"].abs()).round(3)
    return out


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION — Full vs Reduced bridge DefR proxy")
    print("=" * 70)

    df = build_dataset()

    # Correlation analysis first — does the full proxy actually decouple
    # from the existing features?
    print("\n[Correlation with existing features]")
    print("-" * 70)
    corr = correlation_analysis(df)
    print(corr.round(3).to_string(index=False))
    print()
    if corr["corr_full"].abs().max() < corr["corr_reduced"].abs().max():
        print(">> Full-bridge proxy is LESS correlated with existing features.")
        print(">> This is the hoped-for sign that it captures different signal.")
    else:
        print(">> Full-bridge proxy has SIMILAR correlation to the reduced one.")
        print(">> Both may be redundant with existing strength features.")
    corr.to_csv(OUT_DIR / "full_bridge_correlations.csv", index=False)

    # Walk-forward conditions
    print("\n[4/4] Running 4 walk-forward conditions")
    print("-" * 70)
    results = {
        "baseline":        run_one(df, "baseline", BASELINE),
        "+reduced_rolling": run_one(df, "+reduced_rolling", PLUS_REDUCED),
        "+full_rolling":    run_one(df, "+full_rolling", PLUS_FULL),
        "+both":            run_one(df, "+both", PLUS_BOTH),
    }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "condition": name,
            "mean_f1_macro": res.mean_f1_macro,
            "std_f1_macro": res.std_f1_macro,
            "mean_accuracy": res.mean_accuracy,
            "mean_f1_D": res.mean_f1_D,
        })
    summary = pd.DataFrame(summary_rows)
    print("\nMean across folds:")
    print(summary.round(4).to_string(index=False))

    # Per-fold
    rows = []
    base_folds = {f.test_season: f for f in results["baseline"].folds}
    for cond_name, cond_result in results.items():
        for fold in cond_result.folds:
            base = base_folds.get(fold.test_season)
            if base is None:
                continue
            rows.append({
                "condition": cond_name,
                "season": fold.test_season,
                "f1_macro": fold.f1_macro,
                "accuracy": fold.accuracy,
                "log_loss": fold.log_loss,
                "delta_f1": fold.f1_macro - base.f1_macro,
                "delta_acc": fold.accuracy - base.accuracy,
            })
    comparison = pd.DataFrame(rows)
    pivot = comparison.pivot(index="season", columns="condition", values="f1_macro").round(4)
    pivot["Δ reduced"] = (pivot["+reduced_rolling"] - pivot["baseline"]).round(4)
    pivot["Δ full"]    = (pivot["+full_rolling"]    - pivot["baseline"]).round(4)
    pivot["Δ both"]    = (pivot["+both"]            - pivot["baseline"]).round(4)
    print("\nPer-fold F1 deltas vs baseline:")
    print(pivot.to_string())

    # Paired t-tests
    print("\nPaired t-tests vs baseline (one-sided H1: condition > baseline):")
    tests = {}
    base_f1 = comparison[comparison["condition"] == "baseline"].set_index("season")["f1_macro"]
    for cond in ["+reduced_rolling", "+full_rolling", "+both"]:
        cond_f1 = comparison[comparison["condition"] == cond].set_index("season")["f1_macro"]
        common = base_f1.index.intersection(cond_f1.index)
        deltas = cond_f1.loc[common] - base_f1.loc[common]
        try:
            t_stat, p_value = scipy_stats.ttest_rel(cond_f1.loc[common], base_f1.loc[common])
        except Exception:
            t_stat, p_value = float("nan"), float("nan")
        tests[cond] = {
            "mean_delta_f1": float(deltas.mean()),
            "std_delta_f1": float(deltas.std()),
            "t_statistic": float(t_stat) if t_stat == t_stat else None,
            "p_value": float(p_value) if p_value == p_value else None,
            "deltas_by_season": deltas.to_dict(),
        }
        sig = ""
        if p_value == p_value and p_value < 0.05:
            sig = "  * significant (α=0.05)"
        elif p_value == p_value and p_value < 0.10:
            sig = "  . marginal (α=0.10)"
        print(f"  {cond:>18s}: Δf1 = {deltas.mean():+.4f} ± {deltas.std():.4f}  "
              f"t={t_stat:+.3f}  p={p_value:.4f}{sig}")

    # Save everything
    summary.to_csv(OUT_DIR / "full_summary.csv", index=False)
    pivot.to_csv(OUT_DIR / "full_f1_pivot.csv")
    comparison.to_csv(OUT_DIR / "full_per_fold.csv", index=False)
    with open(OUT_DIR / "full_paired_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    all_folds = []
    for name, res in results.items():
        for fold in res.folds:
            all_folds.append({
                "condition": name,
                **{k: v for k, v in vars(fold).items() if k != "train_seasons"},
                "train_seasons": ",".join(fold.train_seasons),
            })
    pd.DataFrame(all_folds).to_csv(OUT_DIR / "full_all_folds.csv", index=False)

    print(f"\nArtifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
