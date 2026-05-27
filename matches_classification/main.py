"""
main.py — single entry point for the Serie A Match Intelligence pipeline.

Full workflow:
  1.  Data pipeline + feature engineering
  2.  Feature analysis (Pearson correlation + Mutual Information)
  3.  Classification — Logistic Regression
  4.  Classification — XGBoost  (with draw threshold tuning)
  5.  Classification — LightGBM
  6.  Classification — Cascaded (W|DL → D|L)
  7.  Regression — goal difference (Linear Regression baseline)
  8.  Model comparison (leaderboard + plots)
  9.  Permutation importance audit (post-training)
  10. Walk-forward backtesting (season by season)
  11. Summary report (markdown)
  12. Interpretability — Logistic (per-class coefficients)
  13. Interpretability — XGBoost (SHAP global + per-class + single match)

Each section is a standalone function — comment one out in main()
without touching the rest.
"""

import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Parameters:.*scale_pos_weight.*are not used",
    category=UserWarning,
)

from pipeline import run_pipeline
from features import add_match_features, add_new_features, add_rolling_team_form
from standings_features import (
    add_standings_features,
    add_opponent_adjusted_features,
    print_standings_sample,
)
from config import (
    REGRESSION_FEATURES, REGRESSION_TARGET,
    LOGISTIC_NUM_FEATURES, XGBOOST_NUM_FEATURES, LGBM_NUM_FEATURES,
    CAT_FEATURES,
)

from models.base import RegressionResult
from models.logistic_pipeline import (
    run_classification_pipeline as run_logistic,
    build_model_pipeline        as build_logistic,
    train_model                 as train_logistic,
)
from models.xgboost_pipeline import (
    run_classification_pipeline as run_xgboost,
    build_model_pipeline        as build_xgboost,
    train_model                 as train_xgboost,
)
from models.lgbm_pipeline import (
    run_classification_pipeline as run_lgbm,
    build_model_pipeline        as build_lgbm,
    train_model                 as train_lgbm,
)
from models.cascaded_pipeline import (
    run_classification_pipeline as run_cascaded,
    plot_cascade_probabilities,
)

from linear_regression_model import run_regression_pipeline
from feature_selection import run_feature_analysis, audit_leakage, run_permutation_audit

from evaluation import (
    print_classification_leaderboard,
    plot_classification_comparison,
    plot_confusion_matrices,
    print_regression_leaderboard,
)

from backtesting import (
    compare_models_backtest,
    plot_backtest_results,
    print_backtest_summary,
)

from reporting import generate_summary_md

# ── logistic interpretability ─────────────────────────────────────────────────
from classification_model_interpretation import (
    plot_feature_importance_per_class,
    plot_class_distribution,
    plot_probability_distribution,
)

# ── XGBoost interpretability — SHAP (file unchanged) ─────────────────────────
from classification_model_interpretation_xgboost import (
    plot_shap_global_importance,
    plot_shap_per_class,
    plot_shap_beeswarm,
    explain_single_match,
    print_match_explanation,
    plot_match_shap_report,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)


# ─── HELPER ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# ─── SECTION 1 — DATA + FEATURES ─────────────────────────────────────────────

def build_dataset(csv_path: str = "data/raw/matches_seriea.csv"):
    """
    Full data pipeline + feature engineering.

    Returns (df, pipeline_outputs):
        df               → enriched DataFrame, used by all models
        pipeline_outputs → dict with team_stats, season_champions, etc.
                           used by generate_report()
    """
    _section("1 — DATA PIPELINE & FEATURE ENGINEERING")

    pipeline_outputs = run_pipeline(csv_path, save=True, output_dir="data/processed")
    df               = pipeline_outputs["raw_df"]

    df = add_match_features(df)
    df = add_new_features(df)
    df = add_rolling_team_form(df, window=5)
    df = add_standings_features(df)
    df = add_opponent_adjusted_features(df)

    pipeline_outputs["raw_df"] = df

    print_standings_sample(df, matchweek=10)
    print(f"  Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, pipeline_outputs


# ─── SECTION 2 — FEATURE ANALYSIS ────────────────────────────────────────────

def run_feature_selection(
    df:             pd.DataFrame,
    corr_threshold: float = 0.90,
    mi_threshold:   float = 0.01,
) -> None:
    """
    Leakage audit + redundancy analysis on all numerical feature sets.
    Does not modify config.py — prints suggestions for manual review.
    XGBoost and LightGBM share the same set → one analysis covers both.
    """
    _section("2 — FEATURE ANALYSIS: LEAKAGE AUDIT + REDUNDANCY")

    print("\n  === LEAKAGE AUDIT — XGBoost / LightGBM features ===")
    audit_leakage(df, XGBOOST_NUM_FEATURES, warn_threshold=0.25, danger_threshold=0.45)

    print("\n  === LEAKAGE AUDIT — Logistic features ===")
    audit_leakage(df, LOGISTIC_NUM_FEATURES, warn_threshold=0.25, danger_threshold=0.45)

    run_feature_analysis(
        df, LOGISTIC_NUM_FEATURES,
        label          = "Logistic Regression",
        corr_threshold = corr_threshold,
        mi_threshold   = mi_threshold,
    )
    run_feature_analysis(
        df, XGBOOST_NUM_FEATURES,
        label          = "XGBoost / LightGBM",
        corr_threshold = corr_threshold,
        mi_threshold   = mi_threshold,
    )


# ─── SECTIONS 3-6 — CLASSIFICATION ───────────────────────────────────────────

def run_classifiers(df: pd.DataFrame):
    """
    Train all four classifiers and print an immediate summary.
    All return ClassificationResult → directly comparable.

    XGBoost is run twice (without and with draw threshold tuning)
    to explicitly show the gain on F1-D.
    """
    _section("3 — CLASSIFICATION: LOGISTIC REGRESSION")
    logistic = run_logistic(df)
    _print_clf_summary(logistic)

    _section("4 — CLASSIFICATION: XGBOOST")
    xgboost_base  = run_xgboost(df, tune_draw_threshold=False)
    xgboost_tuned = run_xgboost(df, tune_draw_threshold=True)
    print("\n  Draw threshold tuning impact:")
    print_classification_leaderboard([xgboost_base, xgboost_tuned])

    _section("5 — CLASSIFICATION: LIGHTGBM")
    lgbm = run_lgbm(df)
    _print_clf_summary(lgbm)

    _section("6 — CLASSIFICATION: CASCADED (W|DL → D|L)")
    cascaded = run_cascaded(df)
    _print_clf_summary(cascaded)
    plot_cascade_probabilities(cascaded.probabilities, cascaded.y_test)

    # xgboost_tuned is the canonical version for subsequent steps
    return logistic, xgboost_tuned, lgbm, cascaded


def _print_clf_summary(r) -> None:
    print(f"  Accuracy : {r.accuracy:.4f}")
    print(f"  F1 macro : {r.f1_macro:.4f}")
    print(f"  F1 [L={r.f1_per_class['L']:.3f}"
          f"  D={r.f1_per_class['D']:.3f}"
          f"  W={r.f1_per_class['W']:.3f}]")
    if r.log_loss is not None:
        print(f"  Log loss : {r.log_loss:.4f}")


# ─── SECTION 7 — REGRESSION ──────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> RegressionResult:
    """
    Linear baseline for goal difference prediction.
    Wraps run_regression_pipeline dict into RegressionResult
    for the unified leaderboard.
    """
    _section("7 — REGRESSION: GOAL DIFFERENCE")

    raw = run_regression_pipeline(df, REGRESSION_FEATURES, REGRESSION_TARGET)
    tm  = raw["test_metrics"]
    cv  = raw["cv_metrics"]

    print(f"  Test → MAE={tm['mae']:.4f}  RMSE={tm['rmse']:.4f}  R²={tm['r2']:.4f}")
    print(f"  CV   → MAE={cv['cv_mae_mean']:.4f}  "
          f"RMSE={cv['cv_rmse_mean']:.4f}  R²={cv['cv_r2_mean']:.4f}")

    return RegressionResult(
        model_name  = "Linear Regression (goal diff)",
        mae         = tm["mae"],
        rmse        = tm["rmse"],
        r2          = tm["r2"],
        cv_mae      = cv["cv_mae_mean"],
        cv_rmse     = cv["cv_rmse_mean"],
        cv_r2       = cv["cv_r2_mean"],
        predictions = raw["predictions"],
        model       = raw["model"],
    )


# ─── SECTION 8 — MODEL COMPARISON ────────────────────────────────────────────

def compare_models(logistic, xgboost, lgbm, cascaded, regression) -> None:
    """
    Unified leaderboard + comparison plots.
    Adding a model = appending it to the list, nothing else.
    """
    _section("8 — MODEL COMPARISON")

    print_classification_leaderboard([logistic, xgboost, lgbm, cascaded])
    plot_classification_comparison([logistic, xgboost, lgbm, cascaded])
    plot_confusion_matrices([logistic, xgboost, lgbm, cascaded])
    print_regression_leaderboard([regression])


# ─── SECTION 9 — PERMUTATION IMPORTANCE ──────────────────────────────────────

def run_importance_audit(xgboost, lgbm) -> None:
    """
    Post-training permutation importance for XGBoost and LightGBM.
    Identifies features that do not impact predictions and can be removed.
    """
    _section("9 — PERMUTATION IMPORTANCE AUDIT")

    for result, name in [(xgboost, "XGBoost"), (lgbm, "LightGBM")]:
        run_permutation_audit(
            grid          = result.model,
            X_test        = result.X_test,
            y_test        = result.y_test,
            threshold     = 0.001,
            n_repeats     = 15,
            model_name    = name,
            label_encoder = result.label_encoder,
        )


# ─── SECTION 10 — BACKTESTING ─────────────────────────────────────────────────

def run_backtesting_analysis(df: pd.DataFrame, logistic, xgboost, lgbm) -> None:
    """
    Walk-forward backtest across all three base classifiers.

    Uses best_params_ from the main training run to avoid re-running
    GridSearch per fold — the backtest measures temporal stability, not tuning.

    Key metric: std_f1_macro — if > 0.10, there is inter-season drift.
    """
    _section("10 — WALK-FORWARD BACKTESTING")

    model_configs = [
        {
            "model_name":           "Logistic Regression",
            "build_pipeline_fn":    build_logistic,
            "train_fn":             train_logistic,
            "num_features":         LOGISTIC_NUM_FEATURES,
            "cat_features":         CAT_FEATURES,
            "needs_label_encoding": False,
            "best_params":          logistic.model.best_params_,
        },
        {
            "model_name":           "XGBoost",
            "build_pipeline_fn":    build_xgboost,
            "train_fn":             train_xgboost,
            "num_features":         XGBOOST_NUM_FEATURES,
            "cat_features":         CAT_FEATURES,
            "needs_label_encoding": True,
            "best_params":          xgboost.model.best_params_,
        },
        {
            "model_name":           "LightGBM",
            "build_pipeline_fn":    build_lgbm,
            "train_fn":             train_lgbm,
            "num_features":         LGBM_NUM_FEATURES,
            "cat_features":         CAT_FEATURES,
            "needs_label_encoding": False,
            "best_params":          lgbm.model.best_params_,
        },
    ]

    backtest_results = compare_models_backtest(df, model_configs)
    print_backtest_summary(backtest_results)
    plot_backtest_results(backtest_results)


# ─── SECTION 11 — REPORT ─────────────────────────────────────────────────────

def generate_report(
    pipeline_outputs: dict,
    best_clf,
    regression:  RegressionResult,
    output_path: str = "reports/summary.md",
) -> None:
    """
    Assembles the dict expected by generate_summary_md and writes the report.
    Passes the model with the highest f1_macro — selected automatically in main().
    """
    _section("11 — REPORT")

    pipeline_outputs["ml_outputs"] = {
        "classification": best_clf,
        "regression": {
            "test_metrics": {
                "mae":  regression.mae,
                "rmse": regression.rmse,
                "r2":   regression.r2,
            },
            "cv_metrics": {
                "cv_mae_mean":  regression.cv_mae,
                "cv_rmse_mean": regression.cv_rmse,
                "cv_r2_mean":   regression.cv_r2,
            },
            "predictions": regression.predictions,
        },
    }

    report_path = generate_summary_md(pipeline_outputs, output_path=output_path)
    print(f"  Report saved to: {report_path}")


# ─── SECTION 12 — LOGISTIC INTERPRETABILITY ───────────────────────────────────

def interpret_logistic(logistic) -> None:
    _section("12 — INTERPRETABILITY: LOGISTIC REGRESSION")

    plot_feature_importance_per_class(logistic.model)
    plot_class_distribution(logistic.y_test, logistic.predictions)
    plot_probability_distribution(logistic.probabilities)


# ─── SECTION 13 — XGBOOST INTERPRETABILITY (SHAP) ────────────────────────────

def interpret_xgboost(df: pd.DataFrame, xgboost, sample_idx: int = 0) -> None:
    """
    SHAP global, per-class, beeswarm + single-match explanation.
    All SHAP logic stays in classification_model_interpretation_xgboost.py.

    Args:
        sample_idx: index in X_test of the match to explain.
                    Change this value to analyse a different match.
    """
    _section("13 — INTERPRETABILITY: XGBOOST (SHAP)")

    grid   = xgboost.model
    X_test = xgboost.X_test

    importance_df = plot_shap_global_importance(grid, X_test, top_k=20)
    print("\nTop 10 features (global SHAP):")
    print(importance_df.head(10).to_string(index=False))

    plot_shap_per_class(grid, X_test)
    plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W")

    _section("13b — SINGLE MATCH EXPLANATION (SHAP)")

    y_test       = xgboost.y_test
    sample_match = X_test.iloc[[sample_idx]]
    true_label   = y_test.iloc[sample_idx]

    team     = sample_match["team"].values[0]
    opponent = sample_match["opponent"].values[0]
    venue    = sample_match["venue"].values[0]
    location = "at home" if venue == "Home" else "away"

    print(f"\n  Match analysed (index {sample_idx}):")
    print(f"  {team}  vs  {opponent}  —  {team} plays {location}")
    print(f"  Actual result: {true_label}")
    print()

    X_background = df[XGBOOST_NUM_FEATURES + CAT_FEATURES].sample(
        n=300, random_state=42
    )
    explanation = explain_single_match(
        grid, sample_match, X_background, true_label, top_k=8
    )
    plot_match_shap_report(explanation, top_k=10)
    print_match_explanation(explanation)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. data ───────────────────────────────────────────────────────────
    df, pipeline_outputs = build_dataset()

    # ── 2. feature analysis (comment out to skip) ─────────────────────────
    run_feature_selection(df)

    # ── 3-6. classification ───────────────────────────────────────────────
    logistic, xgboost, lgbm, cascaded = run_classifiers(df)

    # ── 7. regression ─────────────────────────────────────────────────────
    regression = run_regression(df)

    # ── 8. comparison ─────────────────────────────────────────────────────
    compare_models(logistic, xgboost, lgbm, cascaded, regression)

    # ── 9. permutation importance ─────────────────────────────────────────
    run_importance_audit(xgboost, lgbm)

    # ── 10. backtesting ───────────────────────────────────────────────────
    run_backtesting_analysis(df, logistic, xgboost, lgbm)

    # ── 11. report (best model by f1_macro) ───────────────────────────────
    best_clf = max([logistic, xgboost, lgbm, cascaded], key=lambda r: r.f1_macro)
    generate_report(pipeline_outputs, best_clf, regression)

    # ── 12-13. interpretability ───────────────────────────────────────────
    interpret_logistic(logistic)
    interpret_xgboost(df, xgboost, sample_idx=48)

    _section("DONE")
    print("  Pipeline completed.")


if __name__ == "__main__":
    main()