import pandas as pd

from pipeline import run_pipeline
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
        "matches_seriea.csv",
        save=True,
        output_dir="data/processed",
    )

    build_all_figures(outputs, output_dir="reports/figures")

    print_section("VALIDATION SUMMARY")
    print(outputs["validation_summary"])

    print_section("AGGREGATION CHECKS")
    print(outputs["aggregation_checks"])

    print_section("SEASON CHAMPIONS")
    print(outputs["season_champions"])

    print_section("TITLE RACE")
    print(outputs["title_race"])

    print_section("TOP 5 HOME ADVANTAGE TEAMS")
    print(
        outputs["venue_merged"]
        .sort_values("avg_points_diff", ascending=False)[
            ["team", "home_avg_points", "away_avg_points", "avg_points_diff"]
        ]
        .head(5)
    )

    print_section("MATCH-LEVEL DAY STATS")
    print(outputs["day_stats_matches"])

    print_section("DONE")
    print("Processed tables saved to: data/processed")
    print("Figures saved to: reports/figures")


if __name__ == "__main__":
    main()