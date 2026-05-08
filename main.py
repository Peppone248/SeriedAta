import pandas as pd
import numpy as np
from pipeline import run_pipeline
from reporting import generate_summary_md
from visualizations import (
    build_all_figures,
    plot_residuals,
    plot_predicted_vs_actual,
    plot_residual_vs_predicted,
    plot_correlation_matrix,
    plot_residual_distribution,
    print_target_correlation,
    show_target_distribution
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)
pd.set_option("display.max_colwidth", None)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    outputs = run_pipeline(
        "data/raw/matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    raw_df = outputs["raw_df"]

    build_all_figures(outputs, output_dir="reports/figures")
    report_path = generate_summary_md(outputs, output_path="reports/summary.md")

    print_section("VALIDATION SUMMARY")
    print(outputs["validation_summary"])

    print_section("AGGREGATION CHECKS")
    print(outputs["aggregation_checks"])

    print_section("SEASON CHAMPIONS")
    print(outputs["season_champions"])

    print_section("DONE")
    print("Processed tables saved to: data/processed")
    print("Figures saved to: reports/figures")
    print(f"Summary report saved to: {report_path}")

    from modeling import split_errors, run_regression_pipeline

    feature_cols = ["xg",
                    "xga",
                    "poss",
                    "sh",
                    "sot",
                    "shot_accuracy",
                    "is_home",
                    "strength_points_diff",
                    "strength_xg_diff",
                    "strength_xga_diff",
                    "roll_xg",
                    "roll_xga",
                    "roll_points",
                    ]
    target_col = "goal_diff"

    # -------------------------
    # FEATURE ENGINEERING (ROLLING)
    # -------------------------
    print("\n========== ROLLING FEATURES ==========")

    raw_df = add_rolling_team_form(raw_df, window=5)

    # -------------------------
    # DATA EXPLORATION
    # -------------------------
    print("\n========== TARGET ANALYSIS ==========")

    show_target_distribution(raw_df)
    print_target_correlation(raw_df)

    plot_correlation_matrix(raw_df)

    # 2. Run ML pipeline
    ml_outputs = run_regression_pipeline(
        df=raw_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    print("\n=== TEST METRICS ===")
    print(ml_outputs["test_metrics"])

    print("\n=== CROSS VALIDATION ===")
    print(ml_outputs["cv_metrics"])

    print("\n=== SAMPLE PREDICTIONS ===")
    print(ml_outputs["predictions"].head())

    preds = ml_outputs["predictions"]

    # -------------------------
    # RESIDUAL ANALYSIS
    # -------------------------
    print("\n========== RESIDUAL ANALYSIS ==========")

    print(preds["residual"].describe())

    easy, medium, hard = split_errors(preds)

    print("\nHardest predictions sample:")
    print(hard.head())

    # -------------------------
    # VISUALIZATIONS
    # -------------------------
    print("\n========== GENERATING PLOTS ==========")

    plot_predicted_vs_actual(preds)
    plot_residuals(preds)
    plot_residual_vs_predicted(preds)
    plot_residual_distribution(preds)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
