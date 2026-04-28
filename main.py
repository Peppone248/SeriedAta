import pandas as pd

from pipeline import run_pipeline
from reporting import generate_summary_md
from visualizations import build_all_figures

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

    from modeling import run_linear_regression_baseline

    feature_cols = ["xg", "xga", "poss", "sh", "sot", "is_home", "matchweek"]
    target_col = "goal_diff"

    model_outputs = run_linear_regression_baseline(
        df=outputs["raw_df"],
        feature_cols=feature_cols,
        target_col=target_col,
        save=True,
        output_dir="reports/metrics",
        prefix="goal_diff_baseline",
    )

    print(model_outputs["metrics"])
    print(model_outputs["coefficients"])
    print(model_outputs["predictions"].head())


if __name__ == "__main__":
    main()