"""End-to-end workflow."""

from __future__ import annotations

import pandas as pd

from aggregations import (
    build_daily_team_row_stats,
    build_home_away_comparison,
    build_match_level_day_stats,
    build_team_by_venue,
    build_team_season_stats,
    build_team_stats,
    build_title_race_table,
)
from cleaning import clean_matches, load_matches
from features import add_match_features, add_match_identifiers
from stats import basic_statistics
from validation import validate_raw_values, validate_team_season_stats
from output_utils import save_outputs
import logging
logger = logging.getLogger(__name__)


def run_pipeline(
    csv_path: str,
    save: bool = False,
    output_dir: str = "data/processed",
) -> dict[str, pd.DataFrame | dict[str, object] | object]:
    """Run the full Serie A prep and analysis workflow."""
    raw_df = load_matches(csv_path)
    logger.info("Loading raw data from %s", csv_path)
    clean_matches(raw_df)
    logger.info("Cleaning data")

    validation_summary = validate_raw_values(raw_df)

    add_match_features(raw_df)
    add_match_identifiers(raw_df)

    raw_df.to_csv("data/interim/matches_featured.csv", index=False)

    logger.info("Adding match features")
    stats_summary = basic_statistics(raw_df)

    team_stats = build_team_stats(raw_df)
    team_season_stats = build_team_season_stats(raw_df)
    season_champions, second_places, title_race = build_title_race_table(team_season_stats)
    team_by_venue = build_team_by_venue(raw_df)
    team_counts, venue_merged = build_home_away_comparison(team_by_venue)
    daily_stats = build_daily_team_row_stats(raw_df)
    match_df, day_stats_matches = build_match_level_day_stats(raw_df)

    aggregation_checks = validate_team_season_stats(team_season_stats)

    logger.info("Building aggregate tables")

    outputs = {
        "raw_df": raw_df,
        "validation_summary": validation_summary,
        "stats_summary": stats_summary,
        "team_stats": team_stats,
        "team_season_stats": team_season_stats,
        "season_champions": season_champions,
        "second_places": second_places,
        "title_race": title_race,
        "team_by_venue": team_by_venue,
        "team_counts": team_counts,
        "venue_merged": venue_merged,
        "daily_stats": daily_stats,
        "match_df": match_df,
        "day_stats_matches": day_stats_matches,
        "aggregation_checks": aggregation_checks,
    }

    if save:
        save_outputs(outputs, output_dir=output_dir, save_raw=False)

    logger.info("Saving processed outputs to %s", output_dir)

    return outputs
