"""
test_bronze.py — run the Bronze layer for one season and inspect output.

Run from team_trend/:
    python test_bronze.py

First run is slow (player-match pulls all 380 matches via soccerdata).
Re-runs are instant (soccerdata cache + Bronze .done flag).

Delete this file once Bronze is validated.
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

from team_trend.scrapers.fbref_source import FBrefSource
from team_trend.etl.bronze.extract import _prepare

source = FBrefSource(seasons="2024-2025")

for st in ["schedule", "shooting", "keeper"]:
    print("=" * 60)
    print(f"stat_type = {st}")
    try:
        raw = source.read_team_match_stats(stat_type=st)
        print("  raw shape:", raw.shape)
        print("  raw index names:", raw.index.names)
        df = _prepare(raw)
        print("  prepared shape:", df.shape)
        print("  prepared columns:", df.columns.tolist()[:8])
        df.to_parquet(f"data/bronze/2024-2025/team_{st}.parquet", index=False)
        print("  SAVED OK")
    except Exception as exc:
        import traceback

        print("  FAILED:", exc)
        traceback.print_exc()
