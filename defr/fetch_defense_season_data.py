"""Fetch season-level defense stats from FBref via soccerdata.

This is OPTIONAL enrichment for the team profiling report. If this
fetch fails or the user skips it, build_profiles_and_cluster.py falls
back to the misc match-level data only.

The season-level defense table contains:
    - Tackles by pitch third (Def 3rd, Mid 3rd, Att 3rd) — directly
      distinguishes high press from low block
    - Press attempts and pressure success rate
    - Blocks (shots blocked, passes blocked)
    - Errors leading to opp shot

These are aggregated by FBref across the full season, so the output
is one row per team per season — exactly the granularity we need
for clustering.

Run locally:
    cd SerieAwithPandas/defr
    python fetch_defense_season_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFR_DIR = Path(__file__).resolve().parent
OUT_PATH = DEFR_DIR / "output" / "profiles" / "fbref_defense_season.parquet"

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_INT = {"2020-21": 2020, "2021-22": 2021, "2022-23": 2022,
              "2023-24": 2023, "2024-25": 2024}


def main():
    import soccerdata as sd
    import pandas as pd

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("FBref Serie A defense season stats — soccerdata fetch")
    print("=" * 64)

    all_rows = []
    for season in SEASONS:
        print(f"\n  Fetching {season} (stat_type='defense')...")
        try:
            fbref = sd.FBref(leagues="ITA-Serie A", seasons=season)
            df = fbref.read_team_season_stats(stat_type="defense")
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            print(f"    Skipping this season — fall back will use misc data only.")
            continue

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_")
                          for col in df.columns]

        print(f"    Shape: {df.shape}")
        print(f"    Sample columns: {list(df.columns)[:12]}")

        df["season"] = SEASON_INT[season]
        all_rows.append(df)

    if not all_rows:
        print("\nNo seasons fetched. Defense season data unavailable.")
        sys.exit(1)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print(f"  Shape: {result.shape}, {result['season'].nunique()} seasons")


if __name__ == "__main__":
    main()
