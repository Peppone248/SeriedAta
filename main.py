"""
main.py — entry point unico del progetto Serie A Match Intelligence.

Sostituisce main.py, main_xgboost.py e main_goal_diff.py.
Ogni sezione è una funzione autonoma richiamabile indipendentemente.

Flusso:
  1. Data pipeline + feature engineering
  2. Classificazione — Logistic Regression
  3. Classificazione — XGBoost
  4. Regressione — goal diff
  5. Confronto modelli (leaderboard + grafici)
  6. Interpretabilità — Logistic (coefficienti per classe)
  7. Interpretabilità — XGBoost (SHAP globale + per classe + singola partita)
"""

import pandas as pd
import numpy as np

from pipeline import run_pipeline
from features import add_match_features, add_new_features, add_rolling_team_form
from config import (
    REGRESSION_FEATURES, REGRESSION_TARGET,
    XGBOOST_NUM_FEATURES, CAT_FEATURES,
)

from models.base import RegressionResult
from models.logistic_pipeline import run_classification_pipeline as run_logistic
from models.xgboost_pipeline import run_classification_pipeline as run_xgboost
from models.lgbm_pipeline import run_classification_pipeline as run_lgbm

from linear_regression_model import run_regression_pipeline

from evaluation import (
    print_classification_leaderboard,
    plot_classification_comparison,
    plot_confusion_matrices,
    print_regression_leaderboard,
)

# ── interpretabilità logistic ────────────────────────────────────────────────
from classification_model_interpretation import (
    plot_feature_importance_per_class,
    plot_class_distribution,
    plot_probability_distribution,
)

# ── interpretabilità XGBoost — SHAP (file invariato) ────────────────────────
from classification_model_interpretation_xgboost import (
    plot_shap_global_importance,
    plot_shap_per_class,
    plot_shap_beeswarm,
    explain_single_match,
    print_match_explanation,
    plot_match_shap_report,
)

from reporting import generate_summary_md

pd.set_option("display.width", 1200)


# ─── HELPER ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# ─── SEZIONE 1 — DATA + FEATURES ─────────────────────────────────────────────

def build_dataset(csv_path: str = "data/raw/matches_seriea.csv"):
    """
    Pipeline dati + feature engineering completo.
    Restituisce (df, pipeline_outputs) — il secondo serve a generate_report().
    """
    _section("DATA PIPELINE & FEATURE ENGINEERING")

    pipeline_outputs = run_pipeline(csv_path, save=True, output_dir="data/processed")
    df = pipeline_outputs["raw_df"]

    df = add_match_features(df)
    df = add_new_features(df)
    df = add_rolling_team_form(df, window=5)

    pipeline_outputs["raw_df"] = df  # aggiorna con le feature aggiunte

    print(f"  Dataset pronto: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df, pipeline_outputs


# ─── SEZIONE 2 & 3 — CLASSIFICAZIONE ─────────────────────────────────────────

def run_classifiers(df: pd.DataFrame):
    """
    Allena tutti i classificatori e stampa un riepilogo immediato.
    Tutti restituiscono ClassificationResult → confrontabili direttamente.
    """
    _section("CLASSIFICAZIONE — LOGISTIC REGRESSION")
    logistic = run_logistic(df)
    _print_clf_summary(logistic)

    _section("CLASSIFICAZIONE — XGBOOST")
    xgboost = run_xgboost(df)
    _print_clf_summary(xgboost)

    _section("CLASSIFICAZIONE — LIGHTGBM")
    lgbm = run_lgbm(df)
    _print_clf_summary(lgbm)

    return logistic, xgboost, lgbm


def _print_clf_summary(r) -> None:
    print(f"  Accuracy : {r.accuracy:.4f}")
    print(f"  F1 macro : {r.f1_macro:.4f}")
    print(f"  F1 [L={r.f1_per_class['L']:.3f}  D={r.f1_per_class['D']:.3f}  W={r.f1_per_class['W']:.3f}]")
    if r.log_loss is not None:
        print(f"  Log loss : {r.log_loss:.4f}")


# ─── SEZIONE 4 — REGRESSIONE ─────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> RegressionResult:
    """
    Wrapper che converte l'output di run_regression_pipeline
    (dict) in RegressionResult per la leaderboard unificata.
    """
    _section("REGRESSIONE — GOAL DIFF")

    raw = run_regression_pipeline(df, REGRESSION_FEATURES, REGRESSION_TARGET)

    tm = raw["test_metrics"]
    cv = raw["cv_metrics"]
    print(f"  Test → MAE={tm['mae']:.4f}  RMSE={tm['rmse']:.4f}  R²={tm['r2']:.4f}")
    print(f"  CV   → MAE={cv['cv_mae_mean']:.4f}  RMSE={cv['cv_rmse_mean']:.4f}  R²={cv['cv_r2_mean']:.4f}")

    return RegressionResult(
        model_name="Linear Regression (goal diff)",
        mae=tm["mae"],
        rmse=tm["rmse"],
        r2=tm["r2"],
        cv_mae=cv["cv_mae_mean"],
        cv_rmse=cv["cv_rmse_mean"],
        cv_r2=cv["cv_r2_mean"],
        predictions=raw["predictions"],
        model=raw["model"],
    )


# ─── SEZIONE 5 — CONFRONTO MODELLI ───────────────────────────────────────────

def compare_models(logistic, xgboost, lgbm, regression) -> None:
    """
    Confronto unificato: leaderboard testuale + grafici comparativi.
    Funziona su qualsiasi lista di ClassificationResult / RegressionResult.
    """
    _section("CONFRONTO MODELLI")

    print_classification_leaderboard([logistic, xgboost, lgbm])
    plot_classification_comparison([logistic, xgboost, lgbm])
    plot_confusion_matrices([logistic, xgboost, lgbm])

    print_regression_leaderboard([regression])


# ─── SEZIONE 8 — REPORT ──────────────────────────────────────────────────────

def generate_report(
        pipeline_outputs: dict,
        best_clf,
        regression: RegressionResult,
        output_path: str = "reports/summary.md",
) -> None:
    """
    Assembla il dict atteso da generate_summary_md e scrive il report.

    Args:
        pipeline_outputs: dict restituito da run_pipeline (contiene team_stats,
                          season_champions, venue_merged, ecc.)
        best_clf:         ClassificationResult del modello da includere nel report.
                          Tipicamente il modello con f1_macro più alto.
        regression:       RegressionResult del modello di regressione.
    """
    _section("REPORT")

    pipeline_outputs["ml_outputs"] = {
        "classification": best_clf,
        "regression": {
            "test_metrics": {
                "mae": regression.mae,
                "rmse": regression.rmse,
                "r2": regression.r2,
            },
            "cv_metrics": {
                "cv_mae_mean": regression.cv_mae,
                "cv_rmse_mean": regression.cv_rmse,
                "cv_r2_mean": regression.cv_r2,
            },
            "predictions": regression.predictions,
        },
    }

    report_path = generate_summary_md(pipeline_outputs, output_path=output_path)
    print(f"  Report salvato in: {report_path}")


# ─── SEZIONE 6 — INTERPRETABILITÀ LOGISTIC ───────────────────────────────────

def interpret_logistic(logistic) -> None:
    _section("INTERPRETABILITÀ — LOGISTIC REGRESSION")

    plot_feature_importance_per_class(logistic.model)
    plot_class_distribution(logistic.y_test, logistic.predictions)
    plot_probability_distribution(logistic.probabilities)


# ─── SEZIONE 7 — INTERPRETABILITÀ XGBOOST (SHAP) ─────────────────────────────

def interpret_xgboost(df: pd.DataFrame, xgboost) -> None:
    """
    SHAP: importanza globale, per classe, beeswarm e spiegazione singola partita.
    Tutto il codice SHAP rimane in classification_model_interpretation_xgboost.py.
    """
    _section("INTERPRETABILITÀ — XGBOOST (SHAP)")

    grid = xgboost.model
    X_test = xgboost.X_test

    importance_df = plot_shap_global_importance(grid, X_test, top_k=20)
    print("\nTop 10 feature (SHAP globale):")
    print(importance_df.head(10).to_string(index=False))

    plot_shap_per_class(grid, X_test)
    plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W")

    _section("SPIEGAZIONE SINGOLA PARTITA (XGBOOST — SHAP)")

    y_test = xgboost.y_test
    sample_idx = 48  # change here to select another match

    sample_match = X_test.iloc[[sample_idx]]
    true_label = y_test.iloc[sample_idx]

    # ── identità della partita ────────────────────────────────────────────
    team = sample_match["team"].values[0]
    opponent = sample_match["opponent"].values[0]
    venue = sample_match["venue"].values[0]
    location = "in casa" if venue == "Home" else "in trasferta"

    print(f"\n  Partita analizzata (indice {sample_idx}):")
    print(f"  {team}  vs  {opponent}  —  {team} gioca {location}")
    print(f"  Esito reale: {true_label}")
    print()

    X_background = df[XGBOOST_NUM_FEATURES + CAT_FEATURES].sample(n=300, random_state=42)
    explanation = explain_single_match(grid, sample_match, X_background, true_label, top_k=8)
    plot_match_shap_report(explanation, top_k=10)
    print_match_explanation(explanation)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    df, pipeline_outputs = build_dataset()

    logistic, xgboost, lgbm = run_classifiers(df)
    regression = run_regression(df)

    compare_models(logistic, xgboost, lgbm, regression)

    # passa il modello con f1_macro più alto al report
    best_clf = max([logistic, xgboost, lgbm], key=lambda r: r.f1_macro)
    generate_report(pipeline_outputs, best_clf, regression)

    interpret_logistic(logistic)
    interpret_xgboost(df, xgboost)

    _section("DONE")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()
