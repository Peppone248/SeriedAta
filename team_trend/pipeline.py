"""
pipeline.py — ETL orchestrator: Bronze -> Silver -> Gold.

Design choices (explained in detail in the accompanying message):

  1. Layer functions are imported, not reimplemented: the pipeline only
     sequences them. All transformation logic lives in etl/.
  2. Idempotency via sentinel files (.done) per (layer, season): re-running
     skips completed work unless force=True.
  3. Layer selection via start_from: iterate on Silver/Gold without
     touching the network (Bronze is the only layer that talks to soccerdata).
  4. Fail-fast per season, continue across seasons: one broken season
     doesn't kill the whole run; it's logged and skipped.
  5. The pipeline returns the final Gold DataFrame so callers (main.py,
     notebooks) can chain straight into modelling.

CLI usage:
    python pipeline.py                               # all seasons, all layers
    python pipeline.py --seasons 2024-2025           # one season
    python pipeline.py --start-from silver           # skip Bronze
    python pipeline.py --force                       # rebuild everything
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from team_trend.etl.bronze.extract import run_bronze
from team_trend.etl.silver.clean_team import clean_team_match, save_silver_team_match
from team_trend.etl.silver.clean_players import (
    clean_player_match,
    save_silver_player_match,
    build_player_history,
    save_silver_player_history,
)
from team_trend.etl.gold.build_features import run_gold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ─── configuration ─────────────────────────────────────────────────────────────

SEASONS = [
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

BRONZE_DIR = "data/bronze"
SILVER_DIR = "data/silver"
GOLD_DIR = "data/gold"

LAYERS = ["bronze", "silver", "gold"]


# ─── silver orchestration ──────────────────────────────────────────────────────

def run_silver(season: str, force: bool = False) -> None:
    """
    Run both Silver cleaners for one season and persist.

    Idempotent: skipped if data/silver/{season}/.done exists and not force.
    The .done flag is written only after BOTH tables saved successfully, so
    a crash between the two leaves the season marked incomplete and it will
    be fully re-run next time (Silver is cheap to recompute from Bronze).
    """
    out_dir = Path(SILVER_DIR) / season
    done_flag = out_dir / ".done"

    if done_flag.exists() and not force:
        logger.info("Silver %s already done - skipping", season)
        return

    logger.info("=== SILVER: %s ===", season)

    team_match = clean_team_match(BRONZE_DIR, season)
    save_silver_team_match(team_match, SILVER_DIR, season)

    player_agg = clean_player_match(BRONZE_DIR, season)
    save_silver_player_match(player_agg, SILVER_DIR, season)

    # per-player history (no aggregation): used by Gold to build squad-quality features
    player_history = build_player_history(BRONZE_DIR, season)
    save_silver_player_history(player_history, SILVER_DIR, season)

    done_flag.touch()
    logger.info("Silver %s complete", season)


# ─── full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
        seasons: list[str] = SEASONS,
        start_from: str = "bronze",
        force: bool = False,
        save_gold: bool = True,
) -> pd.DataFrame:
    """
    Execute the ETL pipeline.

    Args:
        seasons:    seasons to process.
        start_from: 'bronze' | 'silver' | 'gold' — entry layer. Layers before
                    it are assumed to exist on disk (validated, not rebuilt).
        force:      rebuild layers even if their .done flags exist.
        save_gold:  persist the Gold parquet.

    Returns:
        The Gold DataFrame spanning all successfully processed seasons.
    """
    if start_from not in LAYERS:
        raise ValueError(f"start_from must be one of {LAYERS}")

    start_idx = LAYERS.index(start_from)
    processed: list[str] = []

    for season in seasons:
        try:
            if start_idx <= 0:
                run_bronze(season, bronze_dir=BRONZE_DIR, force=force)

            if start_idx <= 1:
                run_silver(season, force=force)
            else:
                _assert_silver_exists(season)

            processed.append(season)

        except Exception:
            logger.exception("Season %s failed — skipping", season)

    if not processed:
        raise RuntimeError("No season processed successfully.")

    logger.info("=== GOLD: %s ===", processed)
    gold = run_gold(SILVER_DIR, processed, gold_dir=GOLD_DIR, save=save_gold)

    logger.info("Pipeline complete. Gold: %s rows x %s cols from %d seasons",
                gold.shape[0], gold.shape[1], len(processed))
    return gold


def _assert_silver_exists(season: str) -> None:
    """Loud precondition check when starting from gold."""
    sp = Path(SILVER_DIR) / season
    missing = [f for f in ("team_match.parquet", "player_match_agg.parquet")
               if not (sp / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"start_from='gold' but Silver files missing for {season}: {missing}. "
            f"Run with --start-from silver first."
        )


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="team_trend ETL pipeline")
    parser.add_argument("--seasons", nargs="+", default=SEASONS,
                        help="Seasons to process, e.g. 2024-2025")
    parser.add_argument("--start-from", choices=LAYERS, default="bronze",
                        dest="start_from",
                        help="Entry layer (earlier layers must exist on disk)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if .done flags exist")
    args = parser.parse_args()

    run_pipeline(
        seasons=args.seasons,
        start_from=args.start_from,
        force=args.force,
    )
