import pandas as pd
from pathlib import Path
from pipeline import run_pipeline

pd.set_option("display.max_columns", None)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    outputs = run_pipeline("matches_seriea.csv")

    print_section("VALIDATION SUMMARY")
    print(outputs["validation_summary"])

    print_section("AGGREGATION CHECKS")
    print(outputs["aggregation_checks"])

    print_section("BASIC STATISTICS")
    for key, value in outputs["stats_summary"].items():
        print(f"{key}:")
        print(value)
        print()

    print_section("TEAM STATS - TOP 10")
    print(outputs["team_stats"].head(10))

    print_section("TEAM SEASON STATS - TOP 20")
    print(outputs["team_season_stats"].head(20))

    print_section("SEASON CHAMPIONS")
    print(outputs["season_champions"])

    print_section("SECOND PLACES")
    print(outputs["second_places"])

    print_section("TITLE RACE")
    print(outputs["title_race"])

    print_section("TEAM BY VENUE")
    print(outputs["team_by_venue"].head(20))

    print_section("TEAM COUNTS")
    print(outputs["team_counts"])

    print_section("HOME / AWAY COMPARISON")
    print(outputs["venue_merged"].sort_values("avg_points_diff", ascending=False).head(10))

    print_section("DAILY STATS - TEAM ROW LEVEL")
    print(outputs["daily_stats"])

    print_section("MATCH-LEVEL DAY STATS")
    print(outputs["day_stats_matches"])

    # Optional: save processed outputs
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs["team_stats"].to_csv(output_dir / "team_stats.csv", index=False)
    outputs["team_season_stats"].to_csv(output_dir / "team_season_stats.csv", index=False)
    outputs["season_champions"].to_csv(output_dir / "season_champions.csv", index=False)
    outputs["second_places"].to_csv(output_dir / "second_places.csv", index=False)
    outputs["title_race"].to_csv(output_dir / "title_race.csv", index=False)
    outputs["team_by_venue"].to_csv(output_dir / "team_by_venue.csv", index=False)
    outputs["venue_merged"].to_csv(output_dir / "venue_merged.csv", index=False)
    outputs["daily_stats"].to_csv(output_dir / "daily_stats.csv", index=False)
    outputs["day_stats_matches"].to_csv(output_dir / "day_stats_matches.csv", index=False)

    print_section("DONE")
    print(f"Saved processed tables to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()