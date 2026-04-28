"""Validation checks."""

from __future__ import annotations

import pandas as pd

from config import VALID_RESULTS, VALID_VENUES


def validate_raw_values(df):
    return {
        "valid_result_values": df["result"].isin(["W", "D", "L"]).all(),
        "valid_venue_values": df["venue"].isin(["Home", "Away"]).all(),
        "duplicate_rows": int(df.duplicated().sum()),
        "possible_key_duplicates": int(
            df.duplicated(subset=["date", "team", "opponent", "season"]).sum()
        ),
    }


def validate_team_season_stats(team_season_stats):
    checks = team_season_stats.assign(
        check_matches=lambda d: d["wins"] + d["draws"] + d["losses"]
    )

    return {
        "matches_balance_ok": bool((checks["matches"] == checks["check_matches"]).all()),
        "no_negative_points": bool((team_season_stats["points"] >= 0).all()),
        "rank_starts_at_1": bool((team_season_stats.groupby("season")["rank"].min() == 1).all())
        if "rank" in team_season_stats.columns else None,
    }
