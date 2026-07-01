"""Fetch team-match defensive stats from FBref via soccerdata.

Fetches the "misc" match log (the only stat_type that contains
per-match defensive action counts) for Serie A 2020-2025.

Key columns extracted:
    Int    — interceptions per match (a direct defensive action)
    TklW   — tackles won per match (a direct defensive action)
    Fls    — fouls committed per match (a defensive engagement)
    Fld    — fouls drawn per match
    Crs    — crosses per match
    CrdY   — yellow cards per match

These + their opponent mirrors are the enriched features for the
bridge refit. Unlike n_opp_passes (which soccerdata doesn't expose
for match logs), these are actual defensive action counts — closer
to what DefR measures than pass volume was.

Run locally:
    cd SerieAwithPandas/defr
    python fetch_defensive_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFR_DIR = Path(__file__).resolve().parent
OUT_PATH = DEFR_DIR / "output" / "injection" / "fbref_defensive_data.parquet"

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_INT = {"2020-21": 2020, "2021-22": 2021, "2022-23": 2022,
              "2023-24": 2023, "2024-25": 2024}

# Columns to extract from the misc stat_type
MISC_COLS = ["Int", "TklW", "Fls", "Fld", "Crs", "CrdY"]


def main():
    import soccerdata as sd

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("FBref Serie A defensive match logs — soccerdata fetch")
    print("=" * 64)

    all_rows = []
    for season in SEASONS:
        print(f"\n  Fetching {season} (stat_type='misc')...")
        try:
            fbref = sd.FBref(leagues="ITA-Serie A", seasons=season)
            df = fbref.read_team_match_stats(stat_type="misc")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            sys.exit(1)

        # Flatten multi-index columns
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c) for c in col).strip("_")
                for col in df.columns
            ]

        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)[:15]}...")

        # Find the actual column names after flattening
        # Multi-index becomes things like "Performance_Int", "Performance_TklW"
        col_map = {}
        for target in MISC_COLS:
            matches = [c for c in df.columns if c.endswith(target)]
            if matches:
                col_map[target] = matches[0]
            else:
                print(f"    WARNING: column {target} not found in flattened columns")

        # Also find date, team, opponent
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        team_col = next((c for c in df.columns if c.lower() == "team"), None)
        opp_col = next((c for c in df.columns if "opponent" in c.lower()), None)

        if not date_col or not team_col:
            print(f"    ERROR: cannot find date/team columns. Available: {list(df.columns)}")
            continue

        # Build output subset
        sub = pd.DataFrame()
        sub["date"] = pd.to_datetime(df[date_col])
        sub["team"] = df[team_col]
        if opp_col:
            sub["opponent"] = df[opp_col]
        sub["season"] = SEASON_INT[season]

        for target, src_col in col_map.items():
            sub[target.lower()] = pd.to_numeric(df[src_col], errors="coerce")

        all_rows.append(sub)
        print(f"    Extracted {len(sub)} rows with {len(col_map)} defensive columns")

    if not all_rows:
        print("\nFAILED: no data fetched.")
        sys.exit(1)

    result = pd.concat(all_rows, ignore_index=True)

    # Pair opponent stats
    print(f"\n  Pairing opponent defensive stats...")
    for col in [c for c in result.columns if c in [t.lower() for t in MISC_COLS]]:
        opp_col_name = f"opp_{col}"
        lookup = result.set_index(["date", "team"])[col]
        idx = pd.MultiIndex.from_arrays(
            [result["date"], result["opponent"]], names=["date", "team"]
        )
        result[opp_col_name] = lookup.reindex(idx).values

    n_total = len(result)
    n_missing = result[[f"opp_{c.lower()}" for c in MISC_COLS
                         if c.lower() in result.columns]].isna().any(axis=1).sum()
    print(f"\n  Total rows: {n_total:,}")
    print(f"  Rows with missing opponent data: {n_missing} ({100*n_missing/n_total:.1f}%)")

    # Summary
    print(f"\n  Per-match stat averages:")
    for col in [c.lower() for c in MISC_COLS if c.lower() in result.columns]:
        print(f"    {col:8s}: {result[col].mean():.1f}")

    result.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")
    print(f"  Use this with inject_defr_full.py (after updating it to use")
    print(f"  the defensive features instead of n_opp_passes).")


if __name__ == "__main__":
    main()
