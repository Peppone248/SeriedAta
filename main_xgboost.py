import pandas as pd
import numpy as np

from pipeline import run_pipeline
from features import add_rolling_team_form, add_match_features, add_new_features
from config import NUM_FEATURES, CAT_FEATURES

from models.classification_pipeline_xgboost import (
    run_classification_pipeline,
    plot_confusion_matrix,
)

from classification_model_interpretation_xgboost import (
    plot_shap_global_importance,
    plot_shap_per_class,
    plot_shap_beeswarm,
    explain_single_match,
    print_match_explanation,
    plot_class_distribution,
    plot_match_shap_report
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # ─────────────────────────────────────────
    # 1. DATA PIPELINE
    # ─────────────────────────────────────────
    outputs = run_pipeline(
        "data/raw/matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    raw_df = outputs["raw_df"]

    # ─────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ─────────────────────────────────────────
    raw_df = add_match_features(raw_df)
    raw_df = add_new_features(raw_df)
    raw_df = add_rolling_team_form(raw_df, window=5)

    # ─────────────────────────────────────────
    # 3. CLASSIFICATION (XGBoost + split temporale)
    # ─────────────────────────────────────────
    print_section("CLASSIFICATION PIPELINE")

    # feature sospette da testare
    leaky_features = ["finishing_efficiency", "defensive_efficiency"]

    print("Correlazione con result_num:")
    raw_df["result_num"] = raw_df["result"].map({"W": 1, "D": 0, "L": -1})
    print(raw_df[leaky_features].corrwith(raw_df["result_num"]).sort_values(ascending=False))

    classification_outputs = run_classification_pipeline(raw_df)

    # ─────────────────────────────────────────
    # 4. METRICHE
    # ─────────────────────────────────────────
    print_section("MODEL METRICS")
    metrics = classification_outputs["metrics"]
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1 macro : {metrics['f1_macro']:.4f}")
    print(f"F1 [L,D,W]: {metrics['f1_per_class']}")
    print("\nClassification report:")
    print(pd.DataFrame(metrics["report"]).T.round(3))

    # ─────────────────────────────────────────
    # 5. CONFUSION MATRIX
    # ─────────────────────────────────────────
    print_section("CONFUSION MATRIX")
    plot_confusion_matrix(
        classification_outputs["y_test"],
        classification_outputs["predictions"],
    )

    plot_class_distribution(
        classification_outputs["y_test"],
        classification_outputs["predictions"],
    )

    # ─────────────────────────────────────────
    # 6. FEATURE IMPORTANCE (SHAP)
    # ─────────────────────────────────────────

    print_section("FEATURE IMPORTANCE — SHAP")

    grid = classification_outputs["model"]
    X_test = classification_outputs["X_test"]

    # importanza globale (risponde a: "quali feature contano di più?")
    importance_df = plot_shap_global_importance(grid, X_test, top_k=20)
    print("\nTop 10 feature per importanza SHAP globale:")
    print(importance_df.head(10).to_string(index=False))

    # importanza per classe (risponde a: "cosa spinge verso W / D / L?")
    plot_shap_per_class(grid, X_test)

    # beeswarm per la classe Win (direzione + distribuzione)
    plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W")

    # ─────────────────────────────────────────
    # 7. SPIEGAZIONE SINGOLA PARTITA
    # ─────────────────────────────────────────
    print_section("SINGLE MATCH EXPLANATION")

    y_test = classification_outputs["y_test"]
    sample_idx = 0  # modifica per scegliere un'altra partita

    sample_match = X_test.iloc[[sample_idx]]
    true_label = y_test.iloc[sample_idx]

    X_background = raw_df[NUM_FEATURES + CAT_FEATURES].sample(
        n=300,
        random_state=42
    )
    explanation = explain_single_match(grid, sample_match, X_background, true_label, top_k=8)
    plot_match_shap_report(explanation, top_k=10)
    print_match_explanation(explanation)

    # ─────────────────────────────────────────
    # 8. SAMPLE TABLES
    # ─────────────────────────────────────────
    print_section("SAMPLE PREDICTIONS")
    print(classification_outputs["prediction_table"].head(10).to_string())

    print_section("DONE")


if __name__ == "__main__":
    main()
