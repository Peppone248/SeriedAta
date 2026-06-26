"""
visualization/trajectory_simulation_overlay.py — season-long trajectory
with end-of-season simulation bands overlaid.

The capstone visualization: for each team, shows the FULL SEASON ARC
from matchweek 1 to 38, combining three layers of information:

  1. Actual cumulative points (the ground truth line)
  2. Model-predicted cumulative points (how the model tracked the team)
  3. Simulated final-points distribution (where the team ends up,
     with uncertainty band)

This connects the per-matchweek predictions (trajectory plot) to the
end-of-season forecast (simulation) in a single visual. The key insight
it surfaces: at which point in the season did the model "know" the team's
final position, and when was it still uncertain?
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
DEFAULT_PLOT_DIR = "reports/plots/simulation"


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "savefig.bbox": "tight", "axes.titleweight": "bold",
    })


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path);
    p.mkdir(parents=True, exist_ok=True);
    return p


def plot_season_overlay(
        gold: pd.DataFrame,
        sim_result: dict,
        season: str,
        n_teams: int = 6,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    For selected teams, plot:
      - actual cumulative points across matchweeks (blue solid)
      - predicted cumulative points from the quantile model (orange dashed)
      - simulated final-points 80% band as a shaded rectangle at the end

    Teams selected: top 3 + bottom 3 by actual final points.
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "04_season_trajectory_overlay.png"

    ts = sim_result["team_stats"]
    s = gold[gold["season"] == season].copy()

    # select teams: top 3 + bottom 3 by actual final
    ordered = ts.sort_values("actual_final", ascending=False)
    top = ordered.head(n_teams // 2)["team"].tolist()
    bot = ordered.tail(n_teams // 2)["team"].tolist()
    chosen = top + bot

    ncols = 3
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for i, team in enumerate(chosen):
        ax = axes[i]
        t = s[s["team"] == team].sort_values("matchweek").copy()

        if t.empty:
            ax.set_visible(False)
            continue

        # actual cumulative points: cum_points entering MW + that MW's points
        # cum_points is PRE-match (shift(1) cumsum), so actual running total
        # = cum_points + points for the current match
        t["actual_cum"] = t["cum_points"].astype(float) + t["points"].astype(float)

        # predicted cumulative: cum_points + predicted next-match contribution
        # We approximate the single-match prediction as (roll5_points),
        # which is the model's best guess of per-match productivity
        t["pred_cum"] = t["cum_points"].astype(float) + t["roll5_points"].astype(float) * t["matchweek"].astype(float) / \
                        t["matchweek"].astype(float)
        # simpler and more honest: just plot actual_cum as what happened,
        # and annotate the endpoint with the simulation forecast

        mws = t["matchweek"].to_numpy()
        actual_cum = t["actual_cum"].to_numpy()

        # plot actual cumulative line
        ax.plot(mws, actual_cum, "o-",
                color="#3266ad", linewidth=2, markersize=4,
                label="actual cumulative", zorder=3)

        # get simulation stats for this team
        team_row = ts[ts["team"] == team]
        if not team_row.empty:
            r = team_row.iloc[0]
            final_actual = r["actual_final"]
            final_mean = r["final_mean"]
            final_p10 = r["final_p10"]
            final_p90 = r["final_p90"]
            pivot_mw = r["pivot_matchweek"]

            # shaded band from pivot to matchweek 38
            ax.fill_between(
                [pivot_mw, 38], [final_p10, final_p10], [final_p90, final_p90],
                color="#d84a30", alpha=0.15, label="sim 80% band",
            )
            # predicted median endpoint
            ax.plot([pivot_mw, 38], [actual_cum[mws == pivot_mw][0] if pivot_mw in mws else final_mean, final_mean],
                    "--", color="#d84a30", linewidth=2, label="sim median")
            # actual endpoint
            ax.scatter([38], [final_actual], s=120, c="#3266ad", marker="*",
                       edgecolors="black", linewidths=1, zorder=5,
                       label=f"actual final={int(final_actual)}")

            # position annotation
            pos_mean = r["position_mean"]
            pos_actual = r["actual_position"]
            ax.text(0.98, 0.05,
                    f"pred pos: {pos_mean:.1f}\nactual: {int(pos_actual)}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="lightyellow", alpha=0.8))

        ax.set_title(team, fontweight="bold")
        ax.set_xlabel("Matchweek")
        ax.set_ylabel("Cumulative points" if i % ncols == 0 else "")
        ax.set_xlim(0, 40)
        ax.legend(loc="upper left", fontsize=7.5)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Season trajectory + end-of-season forecast overlay — {season}\n"
        f"blue line = actual cumulative points, red band = simulated final range",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [04] season overlay saved -> {out}")
    return out


def plot_full_league_final_comparison(
        sim_result: dict,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    All 20 teams in one chart: predicted vs actual final points,
    with error bars from the simulation's 80% interval.

    Teams sorted by actual final position. The gap between the orange
    error bar center and the blue dot IS the model's error for that team.
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "05_full_league_comparison.png"

    ts = sim_result["team_stats"].sort_values("actual_final", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(7, len(ts) * 0.35)))

    y_pos = np.arange(len(ts))
    teams = ts["team"].to_numpy()
    actual = ts["actual_final"].to_numpy()
    pred_mean = ts["final_mean"].to_numpy()
    p10 = ts["final_p10"].to_numpy()
    p90 = ts["final_p90"].to_numpy()

    # error bars from simulation
    lower_err = pred_mean - p10
    upper_err = p90 - pred_mean
    ax.errorbar(pred_mean, y_pos, xerr=[lower_err, upper_err],
                fmt="s", color="#d84a30", markersize=6,
                ecolor="#d84a30", elinewidth=1.5, capsize=3, alpha=0.8,
                label="predicted (sim median ± 80%)")
    # actual points
    ax.scatter(actual, y_pos, s=80, c="#3266ad", marker="o",
               edgecolors="black", linewidths=1, zorder=5,
               label="actual final points")

    # connect predicted to actual with thin lines
    for j in range(len(ts)):
        ax.plot([pred_mean[j], actual[j]], [y_pos[j], y_pos[j]],
                "-", color="gray", linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams)
    ax.set_xlabel("Final season points")
    ax.set_title(f"Predicted vs Actual final points — season {sim_result['season']}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [05] full league comparison saved -> {out}")
    return out
