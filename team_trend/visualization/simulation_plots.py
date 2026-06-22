"""
visualization/simulation_plots.py — end-of-season simulation visualizations.

Three plots, each answering a different question about the standings forecast:

  1. Position probability heatmap  — where might each team finish?
  2. Final points distribution     — how confident are we in the final tally?
  3. Outcome probabilities         — what's the chance of top-4 / relegation?
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
        "figure.dpi":       150,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "axes.titleweight": "bold",
    })


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


# ─── 1. position probability heatmap ───────────────────────────────────────────

def plot_position_probability_heatmap(
    sim_result: dict,
    save_dir:   str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Heatmap: rows = teams (sorted by mean predicted position), cols = positions 1..N.
    Cell color = probability the team finishes at that position.
    Red dots = actual final positions (for validation).

    What to look for:
      - tight diagonal -> model confident, agrees with actual
      - red dots far off the bright cells -> model wrong about that team
      - diffuse rows -> high uncertainty (often mid-table)
    """
    out_dir = _ensure_dir(save_dir)
    out     = out_dir / "01_position_probability_heatmap.png"

    ts = sim_result["team_stats"]
    teams_orig = sim_result["teams"]
    team_to_idx = {t: i for i, t in enumerate(teams_orig)}
    probs = sim_result["position_probs"]

    # sort by position_mean ascending (best at top)
    ordered = ts.sort_values("position_mean")
    sorted_teams = ordered["team"].to_numpy()
    probs_sorted = np.stack([probs[team_to_idx[t]] for t in sorted_teams])

    n_teams = len(sorted_teams)
    fig, ax = plt.subplots(figsize=(max(10, n_teams * 0.55),
                                    max(8, n_teams * 0.4)))
    sns.heatmap(
        probs_sorted, ax=ax,
        cmap="Blues", vmin=0, vmax=min(0.5, probs_sorted.max()),
        xticklabels=np.arange(1, n_teams + 1),
        yticklabels=sorted_teams,
        cbar_kws={"label": "P(team finishes at position)"},
        linewidths=0.3, linecolor="white",
    )

    actual_map = dict(zip(ts["team"], ts["actual_position"]))
    for i, t in enumerate(sorted_teams):
        ap = actual_map.get(t)
        if pd.notna(ap):
            ax.scatter(
                float(ap) - 0.5, i + 0.5,
                s=110, c="red", edgecolors="black",
                linewidths=1.5, marker="o", zorder=5,
            )

    ax.set_xlabel("Final position")
    ax.set_ylabel("")
    ax.set_title(
        f"Position probability forecast — season {sim_result['season']}\n"
        f"(red dots = actual final positions)"
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [1] saved {out}")
    return out


# ─── 2. final points distribution ──────────────────────────────────────────────

def plot_final_points_distribution(
    sim_result: dict,
    save_dir:   str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Horizontal box plot of simulated final season points per team.
    Red star = actual final points.

    What to look for:
      - actual stars inside the boxes -> well calibrated
      - boxes systematically above or below the stars -> bias
      - very wide boxes -> high uncertainty for that team
    """
    out_dir = _ensure_dir(save_dir)
    out     = out_dir / "02_final_points_distribution.png"

    ts = sim_result["team_stats"]
    teams_orig = sim_result["teams"]
    team_to_idx = {t: i for i, t in enumerate(teams_orig)}
    final_points = sim_result["final_points"]

    ordered = ts.sort_values("final_mean", ascending=True)
    sorted_teams = ordered["team"].to_numpy()
    data_per_team = [final_points[:, team_to_idx[t]] for t in sorted_teams]

    fig, ax = plt.subplots(figsize=(11, max(6, len(sorted_teams) * 0.35)))
    bp = ax.boxplot(
        data_per_team, vert=False, showfliers=False,
        patch_artist=True, widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3266ad")
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("white"); median.set_linewidth(2)

    actual_map = dict(zip(ts["team"], ts["actual_final"]))
    for i, t in enumerate(sorted_teams):
        af = actual_map.get(t)
        if pd.notna(af):
            ax.scatter(
                af, i + 1, s=180, c="red", marker="*",
                edgecolors="black", linewidths=1.2, zorder=5,
            )

    ax.set_yticks(range(1, len(sorted_teams) + 1))
    ax.set_yticklabels(sorted_teams)
    ax.set_xlabel("Final season points")
    ax.set_title(
        f"Simulated final-points distribution — season {sim_result['season']}\n"
        f"(red stars = actual)"
    )
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [2] saved {out}")
    return out


# ─── 3. outcome probabilities ──────────────────────────────────────────────────

def plot_outcome_probabilities(
    sim_result: dict,
    save_dir:   str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Bar chart: P(top 4) in green, P(relegated) in red, per team.
    Sorted by mean predicted position (best -> worst).
    """
    out_dir = _ensure_dir(save_dir)
    out     = out_dir / "03_outcome_probabilities.png"

    ts = sim_result["team_stats"].sort_values("position_mean")
    teams_sorted = ts["team"].to_numpy()
    p_top4  = ts["p_top_4"].to_numpy()
    p_releg = ts["p_relegated"].to_numpy()

    x = np.arange(len(teams_sorted))
    fig, ax = plt.subplots(figsize=(max(11, len(teams_sorted) * 0.5), 6))
    ax.bar(x - 0.2, p_top4,  width=0.4, label="P(top 4)",     color="#5ab27a", alpha=0.85)
    ax.bar(x + 0.2, p_releg, width=0.4, label="P(relegated)", color="#d84a30", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(teams_sorted, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(f"Top-4 vs Relegation probabilities — season {sim_result['season']}")
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [3] saved {out}")
    return out


# ─── orchestrator ──────────────────────────────────────────────────────────────

def run_all_simulation_plots(
    sim_result: dict,
    save_dir:   str = DEFAULT_PLOT_DIR,
) -> list[Path]:
    _setup_style()
    print(f"\n=== SIMULATION PLOTS -> {save_dir} ===")
    paths = [
        plot_position_probability_heatmap(sim_result, save_dir),
        plot_final_points_distribution(sim_result, save_dir),
        plot_outcome_probabilities(sim_result, save_dir),
    ]
    return paths