"""Validation checks."""

from __future__ import annotations

import pandas as pd

from config import VALID_RESULTS, VALID_VENUES


def validate_raw_values(df: pd.DataFrame) -> dict[str, object]:
    return {
        "valid_result_values_only": df["result"].isin(VALID_RESULTS).all(),
        "valid_venue_values_only": df["venue"].isin(VALID_VENUES).all(),
        "duplicate_rows": int(df.duplicated().sum()),
        "possible_key_duplicates": int(
            df.duplicated(subset=["date", "team", "opponent", "season"]).sum()
        ),
    }



def validate_team_season_stats(team_season_stats: pd.DataFrame) -> dict[str, object]:
    check = team_season_stats.assign(
        check_matches=lambda d: d["wins"] + d["draws"] + d["losses"]
    )
    return {
        "matches_check_passed": bool((check["matches"] == check["check_matches"]).all())
    }
