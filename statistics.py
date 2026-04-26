"""Descriptive statistics helpers."""

from __future__ import annotations

import pandas as pd


def basic_statistics(df: pd.DataFrame) -> dict[str, object]:
    """Return descriptive statistics used in the notebook workflow."""
    attendance_mean = df["attendance"].mean()
    attendance_median = df["attendance"].median()

    iqr_poss = df["poss"].quantile(0.75) - df["poss"].quantile(0.25)
    iqr_xg = df["xg"].quantile(0.75) - df["xg"].quantile(0.25)

    top_goal_diff = df.nlargest(5, "goal_diff")[
        ["date", "season", "team", "opponent", "gf", "ga", "goal_diff"]
    ]

    top_matches_by_xg_diff = df.nlargest(5, "xg_diff")[
        ["date", "season", "team", "opponent", "venue", "gf", "ga", "goal_diff", "result"]
    ]

    shot_accuracy_by_team = (
        df.groupby("team", observed=True)
        .agg(avg_shot_accuracy=("shot_accuracy", "mean"))
        .sort_values("avg_shot_accuracy", ascending=True)
    )

    return {
        "attendance_mean": attendance_mean,
        "attendance_median": attendance_median,
        "iqr_poss": iqr_poss,
        "iqr_xg": iqr_xg,
        "top_goal_diff": top_goal_diff,
        "bottom_teams_shot_accuracy": shot_accuracy_by_team.head(5),
        "top_teams_shot_accuracy": shot_accuracy_by_team.tail(5).sort_values(
            "avg_shot_accuracy", ascending=False
        ),
        "top_matches_by_xg_diff": top_matches_by_xg_diff,
    }
