"""
team_trend/pipeline.py — ETL orchestrator.

Runs the full Bronze → Silver → Gold pipeline for all configured seasons.
Each layer is idempotent: re-running saves to the same files without
corrupting previous results (unless force=True).

Typical usage:
    python pipeline.py                   # full run, all seasons
    python pipeline.py --season 2024-2025  # single season
    python pipeline.py --layer gold      # gold layer only (assumes silver exists)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import SERIE_A_SEASONS, BRONZE_DIR, SILVER_DIR, GOLD_DIR
from team_trend.scrapers.schedule_scraper import ScheduleScraper
from team_trend.scrapers.match_scraper import MatchScraper
from parse_match import (
    parse_summary, parse_keeper, parse_shots,
    parse_passing, parse_defense, parse_possession, parse_misc,
)
from clean_players import (
    clean_summary, clean_keeper, clean_shots,
    clean_passing, clean_defense, clean_possession, clean_misc,
)
from squad_features import aggregate_squad_per_match, build_squad_features

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, path: str, label: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s: %d rows → %s", label, len(df), path)


def _load(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ─── BRONZE LAYER ────────────────────────────────────────────────────────────

def run_bronze(seasons: list[str], force: bool = False) -> None:
    """
    Scrape FBref and persist raw structured tables to Bronze layer.

    For each season:
      1. Scrape all match URLs from the schedule page
      2. Scrape each match report
      3. Parse each table type (summary, keeper, shots, ...)
      4. Save to data/bronze/{season}/{table_type}.parquet
    """
    scraper = MatchScraper(use_cache=True)
    sched   = ScheduleScraper(use_cache=True)

    for season in seasons:
        bronze_season_dir = Path(BRONZE_DIR) / season
        bronze_season_dir.mkdir(parents=True, exist_ok=True)

        # skip if already done and not forced
        done_flag = bronze_season_dir / ".done"
        if done_flag.exists() and not force:
            logger.info("Bronze season %s already done — skipping", season)
            continue

        logger.info("=== BRONZE: season %s ===", season)

        match_urls = sched.scrape_match_urls(season)
        logger.info("Found %d match URLs for season %s", len(match_urls), season)

        tables: dict[str, list[pd.DataFrame]] = {
            "summary": [], "keeper": [], "shots": [],
            "passing": [], "defense": [], "possession": [], "misc": [],
        }

        for url in match_urls:
            report = scraper.scrape_match(url)
            date   = report.date

            for team_id, is_home in [
                (report.home_team, 1), (report.away_team, 0)
            ]:
                if "summary"    in report.tables:
                    raw = report.tables["summary"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["summary"].append(
                        parse_summary(raw_team, url, date, team_id, is_home)
                    )
                if "keeper_stats" in report.tables:
                    raw = report.tables["keeper_stats"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["keeper"].append(
                        parse_keeper(raw_team, url, date, team_id, is_home)
                    )
                if "passing" in report.tables:
                    raw = report.tables["passing"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["passing"].append(
                        parse_passing(raw_team, url, date, team_id, is_home)
                    )
                if "defense" in report.tables:
                    raw = report.tables["defense"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["defense"].append(
                        parse_defense(raw_team, url, date, team_id, is_home)
                    )
                if "possession" in report.tables:
                    raw = report.tables["possession"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["possession"].append(
                        parse_possession(raw_team, url, date, team_id, is_home)
                    )
                if "misc" in report.tables:
                    raw = report.tables["misc"]
                    raw_team = raw[raw["team"] == team_id] if "team" in raw.columns else raw
                    tables["misc"].append(
                        parse_misc(raw_team, url, date, team_id, is_home)
                    )

            if "shots" in report.tables:
                tables["shots"].append(
                    parse_shots(report.tables["shots"], url, date)
                )

        # save each table type
        for name, frames in tables.items():
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                _save(combined, str(bronze_season_dir / f"{name}.parquet"), name)

        done_flag.touch()
        logger.info("Bronze season %s complete", season)


# ─── SILVER LAYER ────────────────────────────────────────────────────────────

def run_silver(seasons: list[str], force: bool = False) -> None:
    """
    Clean Bronze tables and save to Silver layer.
    """
    clean_fns = {
        "summary":    clean_summary,
        "keeper":     clean_keeper,
        "shots":      clean_shots,
        "passing":    clean_passing,
        "defense":    clean_defense,
        "possession": clean_possession,
        "misc":       clean_misc,
    }

    for season in seasons:
        bronze_dir = Path(BRONZE_DIR) / season
        silver_dir = Path(SILVER_DIR) / season
        silver_dir.mkdir(parents=True, exist_ok=True)

        done_flag = silver_dir / ".done"
        if done_flag.exists() and not force:
            logger.info("Silver season %s already done — skipping", season)
            continue

        logger.info("=== SILVER: season %s ===", season)

        for table_name, clean_fn in clean_fns.items():
            bronze_path = bronze_dir / f"{table_name}.parquet"
            if not bronze_path.exists():
                logger.debug("Bronze %s not found for season %s", table_name, season)
                continue

            raw = pd.read_parquet(bronze_path)
            cleaned = clean_fn(raw)
            _save(cleaned, str(silver_dir / f"{table_name}.parquet"), table_name)

        done_flag.touch()
        logger.info("Silver season %s complete", season)


# ─── GOLD LAYER ──────────────────────────────────────────────────────────────

def run_gold(seasons: list[str], match_results: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Aggregate Silver player stats to squad-level Gold features.

    Args:
        seasons:        List of season strings.
        match_results:  DataFrame with (team, date, points, goal_diff)
                        — from the existing matches_classification dataset.
        force:          Recompute even if Gold file exists.

    Returns:
        Concatenated Gold DataFrame across all seasons.
    """
    all_frames = []

    for season in seasons:
        silver_dir = Path(SILVER_DIR) / season
        gold_dir   = Path(GOLD_DIR)
        gold_dir.mkdir(parents=True, exist_ok=True)

        gold_path = gold_dir / f"squad_features_{season}.parquet"
        if gold_path.exists() and not force:
            logger.info("Gold season %s already done — loading", season)
            all_frames.append(pd.read_parquet(gold_path))
            continue

        logger.info("=== GOLD: season %s ===", season)

        def load(name: str) -> pd.DataFrame:
            p = silver_dir / f"{name}.parquet"
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()

        summary_df    = load("summary")
        keeper_df     = load("keeper")
        shots_df      = load("shots")
        defense_df    = load("defense")
        possession_df = load("possession")

        squad_match = aggregate_squad_per_match(
            summary_df, keeper_df, shots_df, defense_df, possession_df
        )

        if squad_match.empty:
            logger.warning("No data aggregated for season %s", season)
            continue

        season_results = match_results[
            match_results["date"].dt.year.astype(str).str.startswith(season[:4])
        ] if not match_results.empty else pd.DataFrame()

        gold_df = build_squad_features(squad_match, season_results)
        _save(gold_df, str(gold_path), f"squad_features_{season}")
        all_frames.append(gold_df)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

def run_pipeline(
    seasons:       list[str]     = SERIE_A_SEASONS,
    match_results: pd.DataFrame  = pd.DataFrame(),
    force:         bool          = False,
    start_from:    str           = "bronze",   # "bronze" | "silver" | "gold"
) -> pd.DataFrame:
    """
    Run the full Bronze → Silver → Gold ETL pipeline.

    Args:
        seasons:       Seasons to process.
        match_results: Existing match results for form features in Gold.
        force:         Re-run all layers even if output files exist.
        start_from:    Skip earlier layers if Bronze/Silver already exist.

    Returns:
        Gold DataFrame ready for ML models.
    """
    logger.info("Starting ETL pipeline for seasons: %s", seasons)

    if start_from == "bronze":
        run_bronze(seasons, force=force)
        run_silver(seasons, force=force)
    elif start_from == "silver":
        run_silver(seasons, force=force)

    gold_df = run_gold(seasons, match_results, force=force)

    logger.info("Pipeline complete. Gold shape: %s", gold_df.shape)
    return gold_df


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="team_trend ETL pipeline")
    parser.add_argument("--seasons", nargs="+", default=SERIE_A_SEASONS)
    parser.add_argument("--layer",   choices=["bronze", "silver", "gold"], default="bronze")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    run_pipeline(
        seasons    = args.seasons,
        force      = args.force,
        start_from = args.layer,
    )
