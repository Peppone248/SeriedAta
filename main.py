"""
main.py — entry point unico del progetto Serie A Match Intelligence.

Flusso completo:
  1.  Data pipeline + feature engineering
  2.  Feature analysis (correlazione Pearson + Mutual Information)
  3.  Classificazione — Logistic Regression
  4.  Classificazione — XGBoost  (con threshold tuning per i draw)
  5.  Classificazione — LightGBM
  6.  Regressione — goal diff (Linear Regression baseline)
  7.  Confronto modelli (leaderboard + grafici)
  8.  Backtesting walk-forward stagione per stagione
  9.  Report markdown
  10. Interpretabilità — Logistic (coefficienti per classe)
  11. Interpretabilità — XGBoost (SHAP globale + per classe + singola partita)

Ogni sezione è una funzione autonoma: puoi commentarne una in main()
senza toccare il resto.
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
from config import (
    REGRESSION_FEATURES, REGRESSION_TARGET,
    LOGISTIC_NUM_FEATURES, XGBOOST_NUM_FEATURES, LGBM_NUM_FEATURES,
    CAT_FEATURES,
)

from models.base import RegressionResult
from models.logistic_pipeline import (
    run_classification_pipeline as run_logistic,
    build_model_pipeline as build_logistic,
    train_model as train_logistic,
)
from models.xgboost_pipeline import (
    run_classification_pipeline as run_xgboost,
    build_model_pipeline as build_xgboost,
    train_model as train_xgboost,
)
from models.lgbm_pipeline import (
    run_classification_pipeline as run_lgbm,
    build_model_pipeline as build_lgbm,
    train_model as train_lgbm,
)

from linear_regression_model import run_regression_pipeline
from feature_selection import run_feature_analysis

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

def build_dataset(csv_path: str = "data/raw/matches_seriea.csv"):
    """
    Pipeline dati + feature engineering completo.
    Restituisce (df, pipeline_outputs):
      - df               → DataFrame arricchito, usato da tutti i modelli
      - pipeline_outputs → dict con team_stats, season_champions, ecc.
                           usato da generate_report()
    """
    _section("1 — DATA PIPELINE & FEATURE ENGINEERING")

    pipeline_outputs = run_pipeline(csv_path, save=True, output_dir="data/processed")
    df = pipeline_outputs["raw_df"]

    df = add_match_features(df)
    df = add_new_features(df)
    df = add_rolling_team_form(df, window=5)

    pipeline_outputs["raw_df"] = df  # aggiorna con le feature aggiunte

    print(f"  Dataset pronto: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df, pipeline_outputs


# ─── SEZIONE 2 — FEATURE ANALYSIS ────────────────────────────────────────────

def run_feature_selection(
        df: pd.DataFrame,
        corr_threshold: float = 0.90,
        mi_threshold: float = 0.01,
) -> None:
    """
    Analisi di ridondanza (Pearson + Mutual Information) sui set numerici.
    Non modifica config.py — stampa suggerimenti da valutare manualmente.
    XGBoost e LightGBM condividono lo stesso set → un solo plot per entrambi.
    """
    _section("2 — FEATURE ANALYSIS: CORRELATION & REDUNDANCY")

    run_feature_analysis(
        df, LOGISTIC_NUM_FEATURES,
        label="Logistic Regression",
        corr_threshold=corr_threshold,
        mi_threshold=mi_threshold,
    )
    run_feature_analysis(
        df, XGBOOST_NUM_FEATURES,
        label="XGBoost / LightGBM",
        corr_threshold=corr_threshold,
        mi_threshold=mi_threshold,
    )


# ─── SEZIONI 3-5 — CLASSIFICAZIONE ───────────────────────────────────────────

def run_classifiers(df: pd.DataFrame):
    """
    Allena i tre classificatori e stampa un riepilogo immediato.
    Tutti restituiscono ClassificationResult → confrontabili direttamente.

    XGBoost viene eseguito due volte (senza e con threshold tuning) per
    mostrare esplicitamente il guadagno su F1-D.
    """
    _section("3 — CLASSIFICAZIONE: LOGISTIC REGRESSION")
    logistic = run_logistic(df)
    _print_clf_summary(logistic)

    _section("4 — CLASSIFICAZIONE: XGBOOST")
    xgboost_base = run_xgboost(df, tune_draw_threshold=False)
    xgboost_tuned = run_xgboost(df, tune_draw_threshold=True)
    print("\n  Impatto del draw threshold tuning:")
    print_classification_leaderboard([xgboost_base, xgboost_tuned])

    _section("5 — CLASSIFICAZIONE: LIGHTGBM")
    lgbm = run_lgbm(df)
    _print_clf_summary(lgbm)

    # restituisce xgboost_tuned come versione canonico per i passi successivi
    return logistic, xgboost_tuned, lgbm


def _print_clf_summary(r) -> None:
    print(f"  Accuracy : {r.accuracy:.4f}")
    print(f"  F1 macro : {r.f1_macro:.4f}")
    print(f"  F1 [L={r.f1_per_class['L']:.3f}"
          f"  D={r.f1_per_class['D']:.3f}"
          f"  W={r.f1_per_class['W']:.3f}]")
    if r.log_loss is not None:
        print(f"  Log loss : {r.log_loss:.4f}")


# ─── SEZIONE 6 — REGRESSIONE ─────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> RegressionResult:
    """
    Baseline lineare per la predizione del goal difference.
    Wrappa il dict di run_regression_pipeline in RegressionResult
    per la leaderboard unificata.
    """
    _section("6 — REGRESSIONE: GOAL DIFF")

    raw = run_regression_pipeline(df, REGRESSION_FEATURES, REGRESSION_TARGET)
    tm = raw["test_metrics"]
    cv = raw["cv_metrics"]

    print(f"  Test → MAE={tm['mae']:.4f}  RMSE={tm['rmse']:.4f}  R²={tm['r2']:.4f}")
    print(f"  CV   → MAE={cv['cv_mae_mean']:.4f}  "
          f"RMSE={cv['cv_rmse_mean']:.4f}  R²={cv['cv_r2_mean']:.4f}")

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


# ─── SEZIONE 7 — CONFRONTO MODELLI ───────────────────────────────────────────

def compare_models(logistic, xgboost, lgbm, regression) -> None:
    """
    Leaderboard testuale + grafici comparativi su ClassificationResult list.
    Aggiungere un modello = aggiungerlo alla lista, nient'altro.
    """
    _section("7 — CONFRONTO MODELLI")

    print_classification_leaderboard([logistic, xgboost, lgbm])
    plot_classification_comparison([logistic, xgboost, lgbm])
    plot_confusion_matrices([logistic, xgboost, lgbm])
    print_regression_leaderboard([regression])


# ─── SEZIONE 8 — BACKTESTING ─────────────────────────────────────────────────

def run_backtesting(df: pd.DataFrame, logistic, xgboost, lgbm) -> None:
    """
    Walk-forward backtest stagione per stagione su tutti e tre i modelli.

    Usa best_params_ dal training principale per evitare GridSearch
    su ogni fold: il backtest misura la stabilità nel tempo, non il tuning.

    Metrica chiave: std_f1_macro — se > 0.10 c'è drift tra stagioni.
    """
    _section("8 — BACKTESTING: WALK-FORWARD PER STAGIONE")

    model_configs = [
        {
            "model_name": "Logistic Regression",
            "build_pipeline_fn": build_logistic,
            "train_fn": train_logistic,
            "num_features": LOGISTIC_NUM_FEATURES,
            "cat_features": CAT_FEATURES,
            "needs_label_encoding": False,
            "best_params": logistic.model.best_params_,
        },
        {
            "model_name": "XGBoost",
            "build_pipeline_fn": build_xgboost,
            "train_fn": train_xgboost,
            "num_features": XGBOOST_NUM_FEATURES,
            "cat_features": CAT_FEATURES,
            "needs_label_encoding": True,
            "best_params": xgboost.model.best_params_,
        },
        {
            "model_name": "LightGBM",
            "build_pipeline_fn": build_lgbm,
            "train_fn": train_lgbm,
            "num_features": LGBM_NUM_FEATURES,
            "cat_features": CAT_FEATURES,
            "needs_label_encoding": False,
            "best_params": lgbm.model.best_params_,
        },
    ]

    backtest_results = compare_models_backtest(df, model_configs)
    print_backtest_summary(backtest_results)
    plot_backtest_results(backtest_results)


# ─── SEZIONE 9 — REPORT ──────────────────────────────────────────────────────

def generate_report(
        pipeline_outputs: dict,
        best_clf,
        regression: RegressionResult,
        output_path: str = "reports/summary.md",
) -> None:
    """
    Assembla il dict atteso da generate_summary_md e scrive il report.
    Passa il modello con f1_macro più alto — scelto automaticamente in main().
    """
    _section("9 — REPORT")

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


# ─── SEZIONE 10 — INTERPRETABILITÀ LOGISTIC ──────────────────────────────────

def interpret_logistic(logistic) -> None:
    _section("10 — INTERPRETABILITÀ: LOGISTIC REGRESSION")

    plot_feature_importance_per_class(logistic.model)
    plot_class_distribution(logistic.y_test, logistic.predictions)
    plot_probability_distribution(logistic.probabilities)


# ─── SEZIONE 11 — INTERPRETABILITÀ XGBOOST (SHAP) ────────────────────────────

def interpret_xgboost(df: pd.DataFrame, xgboost, sample_idx: int = 0) -> None:
    """
    SHAP globale, per classe, beeswarm + spiegazione singola partita.
    Tutto il codice SHAP rimane in classification_model_interpretation_xgboost.py.

    Args:
        sample_idx: indice in X_test della partita da spiegare.
                    Cambia questo valore per analizzare un match diverso.
    """
    _section("11 — INTERPRETABILITÀ: XGBOOST (SHAP)")

    grid = xgboost.model
    X_test = xgboost.X_test

    importance_df = plot_shap_global_importance(grid, X_test, top_k=20)
    print("\nTop 10 feature (SHAP globale):")
    print(importance_df.head(10).to_string(index=False))

    plot_shap_per_class(grid, X_test)
    plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W")

    _section("11b — SPIEGAZIONE SINGOLA PARTITA (SHAP)")

    y_test = xgboost.y_test
    sample_match = X_test.iloc[[sample_idx]]
    true_label = y_test.iloc[sample_idx]

    team = sample_match["team"].values[0]
    opponent = sample_match["opponent"].values[0]
    venue = sample_match["venue"].values[0]
    location = "in casa" if venue == "Home" else "in trasferta"

    print(f"\n  Partita analizzata (indice {sample_idx}):")
    print(f"  {team}  vs  {opponent}  —  {team} gioca {location}")
    print(f"  Esito reale: {true_label}")
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
    # ── 1. dati ───────────────────────────────────────────────────────────
    df, pipeline_outputs = build_dataset()

    # ── 2. analisi feature (commenta per saltare) ─────────────────────────
    run_feature_selection(df)

    # ── 3-5. classificazione ──────────────────────────────────────────────
    logistic, xgboost, lgbm = run_classifiers(df)

    # ── 6. regressione ────────────────────────────────────────────────────
    regression = run_regression(df)

    # ── 7. confronto ──────────────────────────────────────────────────────
    compare_models(logistic, xgboost, lgbm, regression)

    # ── 8. backtesting ────────────────────────────────────────────────────
    run_backtesting(df, logistic, xgboost, lgbm)

    # ── 9. report (modello migliore per f1_macro) ─────────────────────────
    best_clf = max([logistic, xgboost, lgbm], key=lambda r: r.f1_macro)
    generate_report(pipeline_outputs, best_clf, regression)

    # ── 10-11. interpretabilità ───────────────────────────────────────────
    interpret_logistic(logistic)
    interpret_xgboost(df, xgboost, sample_idx=48)

    _section("DONE")
    print("  Pipeline completata.")


if __name__ == "__main__":
    main()
