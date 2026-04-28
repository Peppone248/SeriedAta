from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _df_code_block(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Render a DataFrame as a markdown code block."""
    if df.empty:
        return "```text\n<empty>\n```"
    preview = df.head(max_rows)
    return f"```text\n{preview.to_string(index=False)}\n```"


def generate_summary_md(
    outputs: dict,
    output_path: str = "reports/summary.md",
    project_title: str = "Serie A Data Engineering / Data Science Project",
) -> Path:
    """
    Generate a markdown summary report from pipeline outputs.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = outputs["raw_df"]
    validation_summary = outputs["validation_summary"]
    aggregation_checks = outputs["aggregation_checks"]
    team_stats = outputs["team_stats"]
    season_champions = outputs["season_champions"]
    title_race = outputs["title_race"]
    venue_merged = outputs["venue_merged"]
    day_stats_matches = outputs["day_stats_matches"]

    n_rows, n_cols = raw_df.shape
    n_seasons = int(raw_df["season"].nunique()) if "season" in raw_df.columns else None
    n_teams = int(raw_df["team"].nunique()) if "team" in raw_df.columns else None
    date_min = raw_df["date"].min() if "date" in raw_df.columns else None
    date_max = raw_df["date"].max() if "date" in raw_df.columns else None

    biggest_margin_row = title_race.loc[title_race["title_margin"].idxmax()] if not title_race.empty else None
    closest_race_row = title_race.loc[title_race["title_margin"].idxmin()] if not title_race.empty else None
    biggest_home_adv_row = venue_merged.loc[venue_merged["avg_points_diff"].idxmax()] if not venue_merged.empty else None
    smallest_home_gap_row = venue_merged.loc[venue_merged["avg_points_diff"].idxmin()] if not venue_merged.empty else None
    top_home_win_row = venue_merged.loc[venue_merged["home_win_rate"].idxmax()] if not venue_merged.empty else None
    busiest_day_row = day_stats_matches.loc[day_stats_matches["matches"].idxmax()] if not day_stats_matches.empty else None
    highest_scoring_day_row = (
        day_stats_matches.loc[day_stats_matches["avg_total_goals"].idxmax()]
        if not day_stats_matches.empty else None
    )

    lines: list[str] = []

    lines.append(f"# {project_title}")
    lines.append("")
    lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    lines.append("## Project objective")
    lines.append(
        "Build a reproducible pipeline that transforms raw Serie A match data into "
        "validated analytical tables, visual reports, and machine-learning-ready features."
    )
    lines.append("")

    lines.append("## Dataset overview")
    lines.append(f"- Row count: **{n_rows}**")
    lines.append(f"- Column count: **{n_cols}**")
    lines.append(f"- Seasons: **{n_seasons}**")
    lines.append(f"- Teams: **{n_teams}**")
    if date_min is not None and pd.notna(date_min):
        lines.append(f"- Date range: **{date_min.date()}** to **{date_max.date()}**")
    lines.append("")

    lines.append("## Pipeline stages")
    lines.append("1. Ingestion from raw CSV")
    lines.append("2. Cleaning and schema standardization")
    lines.append("3. Validation and quality checks")
    lines.append("4. Feature engineering")
    lines.append("5. Aggregations and analytical marts")
    lines.append("6. Visualization and reporting")
    lines.append("")

    lines.append("## Validation summary")
    for key, value in validation_summary.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")

    lines.append("## Aggregation checks")
    for key, value in aggregation_checks.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")

    if biggest_margin_row is not None:
        lines.append("## Key findings")
        lines.append(
            f"- Biggest title margin: **{biggest_margin_row['champion_team']}** in "
            f"**{biggest_margin_row['season']}** by **{biggest_margin_row['title_margin']}** points."
        )
        lines.append(
            f"- Closest title race: **{closest_race_row['season']}**, margin of "
            f"**{closest_race_row['title_margin']}** points."
        )
        lines.append(
            f"- Biggest home advantage: **{biggest_home_adv_row['team']}**, with an average points gap of "
            f"**{biggest_home_adv_row['avg_points_diff']:.3f}**."
        )
        lines.append(
            f"- Smallest home-away gap: **{smallest_home_gap_row['team']}**, with an average points gap of "
            f"**{smallest_home_gap_row['avg_points_diff']:.3f}**."
        )
        lines.append(
            f"- Highest home win rate: **{top_home_win_row['team']}** at "
            f"**{top_home_win_row['home_win_rate']:.3f}**."
        )
        lines.append(
            f"- Busiest match day: **{busiest_day_row['day']}** with **{int(busiest_day_row['matches'])}** matches."
        )
        lines.append(
            f"- Highest scoring day: **{highest_scoring_day_row['day']}** with an average of "
            f"**{highest_scoring_day_row['avg_total_goals']:.2f}** total goals."
        )
        lines.append("")

    lines.append("## Season champions")
    lines.append(
        _df_code_block(
            season_champions[["season", "team", "points", "goal_diff", "rank"]]
            if {"season", "team", "points", "goal_diff", "rank"}.issubset(season_champions.columns)
            else season_champions
        )
    )
    lines.append("")

    lines.append("## Title race table")
    title_cols = [
        "season",
        "champion_team",
        "champion_points",
        "second_place_team",
        "second_place_points",
        "title_margin",
    ]
    available_title_cols = [c for c in title_cols if c in title_race.columns]
    lines.append(_df_code_block(title_race[available_title_cols] if available_title_cols else title_race))
    lines.append("")

    lines.append("## Top 10 teams by average points")
    team_cols = ["team", "matches", "avg_points", "win_rate", "avg_xg", "avg_xga"]
    available_team_cols = [c for c in team_cols if c in team_stats.columns]
    lines.append(_df_code_block(team_stats[available_team_cols], max_rows=10))
    lines.append("")

    lines.append("## Top 10 home-advantage teams")
    venue_cols = [
        "team",
        "home_avg_points",
        "away_avg_points",
        "avg_points_diff",
        "home_win_rate",
        "away_win_rate",
        "win_rate_diff",
    ]
    available_venue_cols = [c for c in venue_cols if c in venue_merged.columns]
    venue_top = venue_merged.sort_values("avg_points_diff", ascending=False).head(10)
    lines.append(_df_code_block(venue_top[available_venue_cols], max_rows=10))
    lines.append("")

    lines.append("## Match-level day stats")
    day_cols = ["day", "matches", "avg_total_goals", "avg_attendance"]
    available_day_cols = [c for c in day_cols if c in day_stats_matches.columns]
    lines.append(_df_code_block(day_stats_matches[available_day_cols], max_rows=10))
    lines.append("")

    lines.append("## Saved artifacts")
    lines.append("- Processed tables: `data/processed/`")
    lines.append("- Figures: `reports/figures/`")
    lines.append("- This report: `reports/summary.md`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path