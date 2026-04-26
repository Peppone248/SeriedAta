from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_figure_dir(output_dir: str = "reports/figures") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_champion_points(season_champions: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = season_champions.sort_values("season").copy()
    data["season"] = data["season"].astype(str)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="season", y="champion_points")
    ax.set_title("Champion Points by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    plt.tight_layout()
    plt.savefig(out_path / "champion_points.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_title_margin(title_race: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = title_race.sort_values("season").copy()
    data["season"] = data["season"].astype(str)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=data, x="season", y="title_margin")
    ax.set_title("Title Margin by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points Margin")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    plt.tight_layout()
    plt.savefig(out_path / "title_margin.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_home_away_diff(venue_merged: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = venue_merged.sort_values("avg_points_diff", ascending=False).copy()

    plt.figure(figsize=(12, 9))
    ax = sns.barplot(data=data, y="team", x="avg_points_diff")
    ax.set_title("Home vs Away Average Points Difference")
    ax.set_xlabel("Home Avg Points - Away Avg Points")
    ax.set_ylabel("Team")

    plt.tight_layout()
    plt.savefig(out_path / "home_away_avg_points_diff.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_team_avg_points(team_stats: pd.DataFrame, output_dir: str = "reports/figures", top_n: int = 10) -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = team_stats.sort_values("avg_points", ascending=False).head(top_n).copy()

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=data, x="avg_points", y="team")
    ax.set_title(f"Top {top_n} Teams by Average Points")
    ax.set_xlabel("Average Points per Match")
    ax.set_ylabel("Team")

    plt.tight_layout()
    plt.savefig(out_path / "team_avg_points_top10.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_xg_vs_points(team_stats: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = team_stats.copy()

    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(data=data, x="avg_xg", y="avg_points", s=100)

    for _, row in data.iterrows():
        ax.text(row["avg_xg"] + 0.01, row["avg_points"] + 0.005, row["team"], fontsize=9)

    ax.set_title("Average xG vs Average Points")
    ax.set_xlabel("Average xG")
    ax.set_ylabel("Average Points")

    plt.tight_layout()
    plt.savefig(out_path / "xg_vs_points.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_day_stats(day_stats_matches: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    out_path = ensure_figure_dir(output_dir)
    set_plot_style()

    data = day_stats_matches.copy()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="day", y="avg_total_goals")
    ax.set_title("Average Total Goals by Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Total Goals")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.savefig(out_path / "avg_total_goals_by_day.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_all_figures(outputs: dict, output_dir: str = "reports/figures") -> None:
    plot_champion_points(outputs["season_champions"], output_dir)
    plot_title_margin(outputs["title_race"], output_dir)
    plot_home_away_diff(outputs["venue_merged"], output_dir)
    plot_team_avg_points(outputs["team_stats"], output_dir)
    plot_xg_vs_points(outputs["team_stats"], output_dir)
    plot_day_stats(outputs["day_stats_matches"], output_dir)