import pandas as pd

from pipeline import run_pipeline
from features import add_match_features, add_rolling_team_form

from linear_regression_model import run_regression_pipeline, split_errors

from visualizations import (
    plot_predicted_vs_actual,
    plot_residuals,
    plot_residual_vs_predicted,
    plot_residual_distribution
)

from reporting import generate_summary_md


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    # =========================
    # PIPELINE
    # =========================
    outputs = run_pipeline(
        "data/raw/matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    raw_df = outputs["raw_df"]

    # =========================
    # FEATURE ENGINEERING
    # =========================
    raw_df = add_match_features(raw_df)
    raw_df = add_rolling_team_form(raw_df, window=5)

    # =========================
    # REGRESSION FEATURES
    # =========================
    feature_cols = [
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
        "defensive_efficiency",
        "last_5_points",
        "last_5_goal_diff",
        "last_5_xg",
        "xg_trend",
        "points_trend",
        "days_rest"
    ]

    target_col = "goal_diff"

    # =========================
    # REGRESSION MODEL
    # =========================
    print_section("GOAL DIFF REGRESSION MODEL")

    regression_outputs = run_regression_pipeline(
        df=raw_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    print("\n=== TEST METRICS ===")
    print(regression_outputs["test_metrics"])

    print("\n=== CROSS VALIDATION ===")
    print(regression_outputs["cv_metrics"])

    preds = regression_outputs["predictions"]

    # =========================
    # RESIDUAL ANALYSIS
    # =========================
    print_section("RESIDUAL ANALYSIS")

    print(preds["residual"].describe())

    easy, medium, hard = split_errors(preds)

    print("\nHardest predictions:")
    print(hard.head())

    # =========================
    # VISUALIZATION
    # =========================
    print_section("VISUALIZATIONS")

    plot_predicted_vs_actual(preds)
    plot_residuals(preds)
    plot_residual_vs_predicted(preds)
    plot_residual_distribution(preds)

    # =========================
    # REPORT
    # =========================
    print_section("GENERATING REPORT")

    outputs["ml_outputs"] = {
        "regression": regression_outputs
    }

    report_path = generate_summary_md(outputs, output_path="reports/summary.md")

    print("\nPipeline completed successfully.")
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    main()