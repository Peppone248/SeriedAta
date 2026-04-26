"""Grouped analysis helpers."""

from __future__ import annotations

import pandas as pd


def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("team", observed=True)
        .agg(
            matches=("team", "size"),
            avg_goals_for=("gf", "mean"),
            avg_goals_against=("ga", "mean"),
            avg_xg=("xg", "mean"),
            avg_xga=("xga", "mean"),
            avg_points=("points", "mean"),
            win_rate=("win_flag", "mean"),
            clean_sheet_rate=("clean_sheet", "mean"),
            avg_shot_accuracy=("shot_accuracy", "mean"),
        )
        .sort_values("avg_points", ascending=False)
        .reset_index()
    )



def build_team_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    team_season_stats = (
        df.groupby(["season", "team"], observed=True)
        .agg(
            matches=("team", "size"),
            wins=("win_flag", "sum"),
            draws=("draw_flag", "sum"),
            losses=("loss_flag", "sum"),
            goals_for=("gf", "sum"),
            goals_against=("ga", "sum"),
            points=("points", "sum"),
            avg_xg=("xg", "mean"),
            avg_poss=("poss", "mean"),
            points_per_match=("points", "mean"),
            clean_sheet_rate=("clean_sheet", "mean"),
            avg_shot_accuracy=("shot_accuracy", "mean"),
            home_rate=("is_home", "mean"),
        )
        .assign(goal_diff=lambda d: d["goals_for"] - d["goals_against"])
        .sort_values(["season", "points", "goal_diff"], ascending=[True, False, False])
        .reset_index()
    )

    team_season_stats["rank"] = team_season_stats.groupby("season").cumcount() + 1
    return team_season_stats



def build_title_race_table(team_season_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    season_champions = (
        team_season_stats.loc[
            team_season_stats["rank"] == 1,
            ["season", "team", "points", "goal_diff", "rank"],
        ]
        .rename(
            columns={
                "team": "champion_team",
                "points": "champion_points",
                "goal_diff": "champion_goal_diff",
            }
        )
        .reset_index(drop=True)
    )

    second_places = (
        team_season_stats.loc[
            team_season_stats["rank"] == 2,
            ["season", "team", "points", "goal_diff", "rank"],
        ]
        .rename(
            columns={
                "team": "second_place_team",
                "points": "second_place_points",
                "goal_diff": "second_place_goal_diff",
            }
        )
        .reset_index(drop=True)
    )

    title_race = season_champions.merge(second_places, on="season", validate="one_to_one")
    title_race["title_margin"] = (
        title_race["champion_points"] - title_race["second_place_points"]
    )
    return season_champions, second_places, title_race



def build_team_by_venue(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["team", "venue"], observed=True)
        .agg(
            matches=("team", "size"),
            win_rate=("win_flag", "mean"),
            loss_rate=("loss_flag", "mean"),
            points=("points", "sum"),
            avg_points=("points", "mean"),
            avg_goals_for=("gf", "mean"),
            avg_goals_against=("ga", "mean"),
        )
        .reset_index()
    )



def build_home_away_comparison(team_by_venue: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    team_counts = team_by_venue.groupby("team", observed=True).size()

    home_teams = (
        team_by_venue.loc[
            team_by_venue["venue"] == "Home",
            [
                "team",
                "matches",
                "points",
                "avg_points",
                "win_rate",
                "avg_goals_for",
                "avg_goals_against",
            ],
        ]
        .rename(
            columns={
                "matches": "home_matches",
                "points": "home_points",
                "avg_points": "home_avg_points",
                "win_rate": "home_win_rate",
                "avg_goals_for": "home_avg_goals_for",
                "avg_goals_against": "home_avg_goals_against",
            }
        )
        .reset_index(drop=True)
    )

    away_teams = (
        team_by_venue.loc[
            team_by_venue["venue"] == "Away",
            [
                "team",
                "matches",
                "points",
                "avg_points",
                "win_rate",
                "avg_goals_for",
                "avg_goals_against",
            ],
        ]
        .rename(
            columns={
                "matches": "away_matches",
                "points": "away_points",
                "avg_points": "away_avg_points",
                "win_rate": "away_win_rate",
                "avg_goals_for": "away_avg_goals_for",
                "avg_goals_against": "away_avg_goals_against",
            }
        )
        .reset_index(drop=True)
    )

    venue_merged = home_teams.merge(away_teams, on="team", validate="one_to_one")
    venue_merged["points_diff"] = venue_merged["home_points"] - venue_merged["away_points"]
    venue_merged["avg_points_diff"] = (
        venue_merged["home_avg_points"] - venue_merged["away_avg_points"]
    )
    venue_merged["win_rate_diff"] = (
        venue_merged["home_win_rate"] - venue_merged["away_win_rate"]
    )
    venue_merged["goals_for_diff"] = (
        venue_merged["home_avg_goals_for"] - venue_merged["away_avg_goals_for"]
    )

    return team_counts, venue_merged



def build_daily_team_row_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("day", observed=True)
        .agg(
            avg_attendance=("attendance", "mean"),
            avg_clean_sheets=("clean_sheet", "mean"),
            avg_goals_for=("gf", "mean"),
            avg_goals_against=("ga", "mean"),
            avg_points=("points", "mean"),
            team_rows=("day", "size"),
        )
        .sort_values("team_rows", ascending=False)
        .reset_index()
    )



def build_match_level_day_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_df = (
        df.sort_values(["season", "date", "home_team", "away_team"])
        .drop_duplicates(subset=["season", "date", "home_team", "away_team"])
        .copy()
    )

    day_stats_matches = (
        match_df.groupby("day", observed=True)
        .agg(
            matches=("day", "size"),
            avg_total_goals=("total_goals", "mean"),
            avg_attendance=("attendance", "mean"),
        )
        .sort_values("matches", ascending=False)
        .reset_index()
    )

    return match_df, day_stats_matches
