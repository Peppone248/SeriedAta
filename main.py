"""
main.py — entry point unico del progetto Serie A Match Intelligence.

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
from models.xgboost_pipeline  import run_classification_pipeline as run_xgboost

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

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)


# ─── HELPER ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# ─── SEZIONE 1 — DATA + FEATURES ─────────────────────────────────────────────

def build_dataset(csv_path: str = "data/raw/matches_seriea.csv") -> pd.DataFrame:
    """
    Pipeline dati + feature engineering completo.
    Unico punto in cui vengono chiamate le funzioni di features.py.
    """
    _section("DATA PIPELINE & FEATURE ENGINEERING")

    outputs = run_pipeline(csv_path, save=True, output_dir="data/processed")
    df = outputs["raw_df"]

    df = add_match_features(df)
    df = add_new_features(df)
    df = add_rolling_team_form(df, window=5)

    print(f"  Dataset pronto: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df


# ─── SEZIONE 2 & 3 — CLASSIFICAZIONE ─────────────────────────────────────────

def run_classifiers(df: pd.DataFrame):
    """
    Allena entrambi i classificatori e stampa un riepilogo immediato.
    Entrambi restituiscono ClassificationResult → confrontabili direttamente.
    """
    _section("CLASSIFICAZIONE — LOGISTIC REGRESSION")
    logistic = run_logistic(df)
    _print_clf_summary(logistic)

    _section("CLASSIFICAZIONE — XGBOOST")
    xgboost = run_xgboost(df)
    _print_clf_summary(xgboost)

    return logistic, xgboost


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


# ─── SEZIONE 5 — CONFRONTO MODELLI ───────────────────────────────────────────

def compare_models(logistic, xgboost, regression) -> None:
    """
    Confronto unificato: leaderboard testuale + grafici comparativi.
    Funziona su qualsiasi lista di ClassificationResult / RegressionResult.
    """
    _section("CONFRONTO MODELLI")

    print_classification_leaderboard([logistic, xgboost])
    plot_classification_comparison([logistic, xgboost])
    plot_confusion_matrices([logistic, xgboost])

    print_regression_leaderboard([regression])


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

    grid   = xgboost.model
    X_test = xgboost.X_test

    importance_df = plot_shap_global_importance(grid, X_test, top_k=20)
    print("\nTop 10 feature (SHAP globale):")
    print(importance_df.head(10).to_string(index=False))

    plot_shap_per_class(grid, X_test)
    plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W")

    _section("SPIEGAZIONE SINGOLA PARTITA (XGBOOST — SHAP)")

    y_test       = xgboost.y_test
    sample_match = X_test.iloc[[0]]
    true_label   = y_test.iloc[0]

    X_background = df[XGBOOST_NUM_FEATURES + CAT_FEATURES].sample(n=300, random_state=42)
    explanation  = explain_single_match(grid, sample_match, X_background, true_label, top_k=8)
    plot_match_shap_report(explanation, top_k=10)
    print_match_explanation(explanation)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    df = build_dataset()

    logistic, xgboost = run_classifiers(df)
    regression        = run_regression(df)

    compare_models(logistic, xgboost, regression)

    interpret_logistic(logistic)
    interpret_xgboost(df, xgboost)

    _section("DONE")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()
