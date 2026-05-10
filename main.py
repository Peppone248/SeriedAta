import pandas as pd

from pipeline import run_pipeline
from reporting import generate_summary_md
from features import add_rolling_team_form, add_match_features

from classification_model_interpretation import (
    plot_feature_importance_per_class,
    plot_class_distribution,
    plot_probability_distribution,
    plot_match_recap
)

from visualizations import (
    plot_residuals,
    plot_predicted_vs_actual,
    plot_residual_vs_predicted,
    plot_correlation_matrix,
    plot_residual_distribution,
    print_target_correlation,
    show_target_distribution
)

from modeling import run_regression_pipeline, split_errors
# from models.classification_model import (
#    build_classification_dataset,
#    split_classification_data,
#    train_logistic_regression,
#    evaluate_classification
#)

from models.classification_pipeline import run_classification_pipeline, plot_confusion_matrix

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)
pd.set_option("display.max_colwidth", None)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    # =========================
    # 1. DATA PIPELINE
    # =========================
    outputs = run_pipeline(
        "data/raw/matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    raw_df = outputs["raw_df"]

    # =========================
    # 2. FEATURE ENGINEERING
    # =========================
    raw_df = add_match_features(raw_df)
    raw_df = add_rolling_team_form(raw_df, window=5)

    # =========================
    # 3. CLASSIFICATION MODEL
    # =========================
    classification_features = [
        "xg",
        "xga",
        "poss",
        "sot",
        "shot_accuracy",
        "is_home",
        "strength_points_diff",
        "strength_xg_diff",
        "strength_xga_diff",
        "finishing_efficiency",
        "defensive_efficiency"
    ]
    print_section("CLASSIFICATION PIPELINE")

    classification_outputs = run_classification_pipeline(
        raw_df,
        run_eda_flag=False
    )

    # =========================
    # 4. METRICS
    # =========================
    print_section("MODEL RESULTS")

    print("\n=== METRICS ===")
    print(classification_outputs["metrics"])

    # =========================
    # 5. VISUAL CHECKS (OUTSIDE PIPELINE)
    # =========================
    print_section("VISUALIZATION")

    plot_confusion_matrix(
        classification_outputs["y_test"],
        classification_outputs["predictions"]
    )

    print("\n=== SAMPLE PREDICTIONS ===")
    print(classification_outputs["prediction_table"].head())

    print("\n=== PROBABILITY SAMPLE ===")
    print(classification_outputs["probability_table"].head())

    # =========================
    # 6. REPORT GENERATION
    # =========================
    outputs["ml_outputs"] = {
        "classification": classification_outputs
    }

    model = classification_outputs["model"].best_estimator_
    pred_table = classification_outputs["prediction_table"]
    prob_table = classification_outputs["probability_table"]
    X_test = classification_outputs["X_test"]
    y_test = classification_outputs["y_test"]

    y_pred = classification_outputs["predictions"]
    proba = classification_outputs["probabilities"]
    # Feature importance (FIXED)
    plot_feature_importance_per_class(
        classification_outputs["model"],  # pipeline, not estimator
        )
    plot_class_distribution(y_test, y_pred)

    # ✔ Probability distribution
    plot_probability_distribution(proba)

    # ✔ Match recap (sample interpretation)
    plot_match_recap(prob_table, n_samples=5)

    report_path = generate_summary_md(outputs, output_path="reports/summary.md")

    print_section("DONE")

    print(f"Report saved to: {report_path}")
    print("Pipeline completed successfully.")


"""def main():

    # =========================
    # PIPELINE
    # =========================
    outputs = run_pipeline(
        "data/raw/matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    raw_df = outputs["raw_df"]

    raw_df = add_match_features(raw_df)
    raw_df = add_rolling_team_form(raw_df, window=5)

    classification_features = [
        "xg",
        "xga",
        "poss",
        "sot",
        "shot_accuracy",
        "is_home",
        "strength_points_diff",
        "strength_xg_diff",
        "strength_xga_diff",
        "finishing_efficiency",
        "defensive_efficiency"
    ]

    classification_outputs = run_classification_pipeline(
        df=raw_df,
        feature_cols=classification_features,
        target_col="result",
        enable_plots=True
    )

    # =========================
    # FEATURE ENGINEERING
    # =========================
    # print_section("ROLLING FEATURES")
    # raw_df = add_rolling_team_form(raw_df, window=5)
    #
    # # =========================
    # # EXPLORATION
    # # =========================
    # print_section("DATA EXPLORATION")
    #
    # show_target_distribution(raw_df)
    # print_target_correlation(raw_df)
    # plot_correlation_matrix(raw_df)
    #
    # # =========================
    # # FEATURES
    # # =========================
    # feature_cols = [
    #     "xg",
    #     "xga",
    #     "poss",
    #     "sot",
    #     "shot_accuracy",
    #     "is_home",
    #     "strength_points_diff",
    #     "strength_xg_diff",
    #     "strength_xga_diff",
    # ]
    #
    # # =========================
    # # REGRESSION
    # # =========================
    # print_section("REGRESSION MODEL")
    #
    # regression_outputs = run_regression_pipeline(
    #     df=raw_df,
    #     feature_cols=feature_cols,
    #     target_col="goal_diff",
    # )
    #
    # print("\n=== TEST METRICS ===")
    # print(regression_outputs["test_metrics"])
    #
    # print("\n=== CROSS VALIDATION ===")
    # print(regression_outputs["cv_metrics"])
    #
    # preds = regression_outputs["predictions"]
    #
    # # =========================
    # # RESIDUAL ANALYSIS
    # # =========================
    # print_section("RESIDUAL ANALYSIS")
    #
    # print(preds["residual"].describe())
    #
    # easy, medium, hard = split_errors(preds)
    #
    # print("\nHardest predictions:")
    # print(hard.head())
    #
    # # =========================
    # # CLASSIFICATION
    # # =========================
    # print_section("CLASSIFICATION MODEL")
    #
    # X, y = build_classification_dataset(raw_df, feature_cols, target="result")
    #
    # X_train, X_test, y_train, y_test = split_classification_data(X, y)
    #
    # clf_model = train_logistic_regression(X_train, y_train)
    #
    # classification_outputs = evaluate_classification(
    #     clf_model, X_test, y_test
    # )
    #
    # print("\n=== CLASSIFICATION METRICS ===")
    # print(classification_outputs["metrics"])
    #
    # # =========================
    # # ATTACH ML OUTPUTS
    # # =========================
    # outputs["ml_outputs"] = {
    #     "regression": regression_outputs,
    #     "classification": classification_outputs
    # }
    #
    # # =========================
    # # VISUALIZATION
    # # =========================
    # print_section("VISUALIZATIONS")
    #
    # plot_predicted_vs_actual(preds)
    # plot_residuals(preds)
    # plot_residual_vs_predicted(preds)
    # plot_residual_distribution(preds)
    #
    # # =========================
    # # REPORT
    # # =========================
    # print_section("GENERATING REPORT")
    #
    # report_path = generate_summary_md(outputs, output_path="reports/summary.md")
    #
    # print("\nPipeline completed successfully.")
    # print(f"Summary report saved to: {report_path}")"""


if __name__ == "__main__":
    main()