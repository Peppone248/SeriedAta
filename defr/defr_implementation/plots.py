"""All charts and figures for the DefR analysis report.

Generates five PNG plots:
    1. pitch_zones.png         — 6×4 zone heatmap on a pitch with rate annotations
    2. defr_distribution.png   — histogram of match-level DefR + season-level box
    3. team_rankings.png       — horizontal bar chart of season DefR by team
    4. regression_diagnostics.png — predicted vs actual, residuals, Q-Q, histogram
    5. style_clustering.png    — DefR vs possession scatter with quadrant labels
                                  and footballistic annotations

All plots use a consistent matplotlib style and a shared palette for
positive/negative DefR. Saved at 150 DPI for the HTML report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Rectangle
from scipy import stats as scipy_stats

from . import config

# ─── Shared style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# Palette
COLOR_POS = "#0F6E56"     # green — over-performing
COLOR_NEG = "#C03A2B"     # red   — under-performing
COLOR_NEUTRAL = "#888888"
COLOR_ACCENT = "#1E5A99"
COLOR_PITCH_GRASS = "#F5F2E8"
COLOR_PITCH_LINE = "#2A2A2A"

# Diverging colormap for the zone heatmap
ZONE_CMAP = LinearSegmentedColormap.from_list(
    "defr_zones",
    ["#F0F8FF", "#A8D0E6", "#377EB8", "#1F4D80", "#0A2540"],
    N=256,
)


# ─── Pitch drawing helper ─────────────────────────────────────────────
def _draw_pitch(ax, x_max=100, y_max=100):
    """Draw a simplified football pitch in Wyscout coordinates (0-100)."""
    ax.set_xlim(-2, x_max + 2)
    ax.set_ylim(-2, y_max + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    line_kw = dict(color=COLOR_PITCH_LINE, lw=1.2)

    # Outer rectangle
    ax.add_patch(Rectangle((0, 0), x_max, y_max, fill=False, **line_kw))
    # Halfway line
    ax.plot([x_max / 2, x_max / 2], [0, y_max], **line_kw)
    # Centre circle (radius ~9 in Wyscout-scaled coords)
    ax.add_patch(Circle((x_max / 2, y_max / 2), 9, fill=False, **line_kw))
    ax.add_patch(Circle((x_max / 2, y_max / 2), 0.5, color=COLOR_PITCH_LINE))

    # Penalty boxes: ~16x40 in scaled coords
    pen_w, pen_h = 16, 40
    ax.add_patch(Rectangle((0, (y_max - pen_h) / 2), pen_w, pen_h, fill=False, **line_kw))
    ax.add_patch(Rectangle((x_max - pen_w, (y_max - pen_h) / 2), pen_w, pen_h, fill=False, **line_kw))
    # 6-yard boxes
    six_w, six_h = 6, 18
    ax.add_patch(Rectangle((0, (y_max - six_h) / 2), six_w, six_h, fill=False, **line_kw))
    ax.add_patch(Rectangle((x_max - six_w, (y_max - six_h) / 2), six_w, six_h, fill=False, **line_kw))
    # Goals
    ax.plot([0, 0], [(y_max - 8) / 2, (y_max + 8) / 2], color=COLOR_PITCH_LINE, lw=2.5)
    ax.plot([x_max, x_max], [(y_max - 8) / 2, (y_max + 8) / 2], color=COLOR_PITCH_LINE, lw=2.5)


# ─── Plot 1: pitch zones heatmap ──────────────────────────────────────
def plot_pitch_zones(zone_rates: pd.DataFrame, out_path: Path):
    """Heatmap of baseline defensive rates per zone, overlaid on a pitch."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(COLOR_PITCH_GRASS)
    ax.set_facecolor(COLOR_PITCH_GRASS)

    cell_w = config.PITCH_X_MAX / config.N_ZONE_COLS
    cell_h = config.PITCH_Y_MAX / config.N_ZONE_ROWS

    rates = zone_rates.set_index("zone_id")["baseline_rate"]
    vmin, vmax = float(rates.min()), float(rates.max())

    for _, row in zone_rates.iterrows():
        x0 = row["zone_col"] * cell_w
        y0 = row["zone_row"] * cell_h
        rate = row["baseline_rate"]
        # Normalize for colormap
        norm = (rate - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        color = ZONE_CMAP(norm)
        ax.add_patch(Rectangle(
            (x0, y0), cell_w, cell_h,
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.85,
        ))
        # Annotate value (lighter text on dark cells)
        text_color = "white" if norm > 0.5 else "#0A2540"
        ax.text(
            x0 + cell_w / 2, y0 + cell_h / 2,
            f"{rate:.2f}",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color,
        )

    # Redraw pitch lines on top
    _draw_pitch(ax)

    # Goal labels
    ax.text(-1.5, config.PITCH_Y_MAX / 2, "← Defending team's goal",
            ha="right", va="center", fontsize=9, color="#555", style="italic")
    ax.text(config.PITCH_X_MAX + 1.5, config.PITCH_Y_MAX / 2, "Attacking direction →",
            ha="left", va="center", fontsize=9, color="#555", style="italic")

    ax.set_title(
        "Per-zone baseline defensive rates (Serie A 2017/18)\n"
        f"defensive actions ÷ attacking actions, computed across {380} matches",
        pad=14,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=ZONE_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Baseline defensive rate", fontsize=9)

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


# ─── Plot 2: DefR distribution ────────────────────────────────────────
def plot_defr_distribution(defr_match: pd.DataFrame, season: pd.DataFrame, out_path: Path):
    """Two-panel: histogram of match-level DefR + per-team box plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel 1: histogram of match-level DefR
    data = defr_match["defr_score"].values
    ax1.hist(data, bins=40, color=COLOR_ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="#333", linewidth=1.2, linestyle="--", alpha=0.7, label="Expected = Actual")
    ax1.axvline(data.mean(), color=COLOR_NEG, linewidth=1.4, linestyle="-",
                label=f"Mean = {data.mean():.1f}")
    ax1.set_xlabel("DefR score (actual − expected defensive actions per match)")
    ax1.set_ylabel("Number of team-matches")
    ax1.set_title(f"Distribution of team-match DefR scores  (n = {len(data)})")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Annotation: skew, std
    skew_val = scipy_stats.skew(data)
    ax1.text(0.02, 0.98,
             f"σ = {data.std():.1f}\nskew = {skew_val:+.2f}\nrange = [{data.min():.0f}, {data.max():.0f}]",
             transform=ax1.transAxes, va="top", ha="left",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                    facecolor="white", edgecolor="#ccc", alpha=0.9))

    # Panel 2: per-team box plot, sorted by mean DefR
    teams_order = season.sort_values("avg_defr")["team_name"].tolist()
    box_data = [defr_match.loc[defr_match["team_name"] == t, "defr_score"].values
                for t in teams_order]
    bp = ax2.boxplot(box_data, vert=False, widths=0.6, patch_artist=True,
                     medianprops=dict(color="white", linewidth=1.5),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    # Color boxes by mean DefR
    for patch, team in zip(bp["boxes"], teams_order):
        mean_defr = season.set_index("team_name").loc[team, "avg_defr"]
        patch.set_facecolor(COLOR_POS if mean_defr >= 0 else COLOR_NEG)
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    for cap in bp["caps"] + bp["whiskers"]:
        cap.set_color("#666")
    ax2.set_yticks(range(1, len(teams_order) + 1))
    ax2.set_yticklabels(teams_order, fontsize=9)
    ax2.axvline(0, color="#333", linewidth=1.0, linestyle="--", alpha=0.5)
    ax2.set_xlabel("DefR score per match")
    ax2.set_title("Match-level DefR distribution by team (sorted by season mean)")

    plt.suptitle("DefR score distribution — Serie A 2017/18",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Plot 3: team rankings ────────────────────────────────────────────
def plot_team_rankings(season: pd.DataFrame, out_path: Path):
    """Horizontal bar chart of season-average DefR by team."""
    sorted_ = season.sort_values("avg_defr").reset_index(drop=True)
    n = len(sorted_)

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in sorted_["avg_defr"]]
    bars = ax.barh(
        range(n), sorted_["avg_defr"],
        color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    # Add ±1σ error bars (season-level uncertainty from match-to-match variance)
    errors = sorted_["std_defr"] / np.sqrt(sorted_["matches"])  # SE of mean
    ax.errorbar(
        sorted_["avg_defr"], range(n),
        xerr=errors, fmt="none", ecolor="#444", capsize=2, lw=0.7,
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_["team_name"], fontsize=10)
    ax.axvline(0, color="#333", linewidth=1.0, linestyle="-", alpha=0.6)
    ax.set_xlabel("Season-average DefR score (actual − expected defensive actions)")
    ax.set_title("Serie A 2017/18 — season DefR rankings\n"
                 "error bars = ±1 SE of mean across 38 matches",
                 pad=12)

    # Add value labels at bar ends
    for i, (val, ratio) in enumerate(zip(sorted_["avg_defr"], sorted_["avg_ratio"])):
        xpos = val + (1.5 if val >= 0 else -1.5)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, i, f"{val:+.1f}  (ratio {ratio:.2f})",
                va="center", ha=ha, fontsize=8.5, color="#333")

    ax.set_xlim(sorted_["avg_defr"].min() - 12, sorted_["avg_defr"].max() + 12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Plot 4: regression diagnostics ───────────────────────────────────
def plot_regression_diagnostics(bridge_results: dict, merged: pd.DataFrame, out_path: Path):
    """Four-panel diagnostics for the Ridge bridge regression."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    (ax1, ax2), (ax3, ax4) = axes

    y_true = merged["defr_score"].values
    y_pred = merged["defr_predicted"].values
    resid = merged["defr_residual"].values

    # 4a: predicted vs actual
    ax1.scatter(y_pred, y_true, alpha=0.35, s=14, color=COLOR_ACCENT, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
    ax1.plot(lims, lims, color="#C03A2B", linewidth=1.4, linestyle="--", label="y = x")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Predicted DefR")
    ax1.set_ylabel("Actual DefR")
    ax1.set_title(f"Predicted vs actual  (R² = {bridge_results['full_r2']:.3f},  "
                  f"CV R² = {bridge_results['cv_r2_mean']:.3f} ± {bridge_results['cv_r2_std']:.3f})")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.text(0.02, 0.02,
             f"n = {len(y_true)}\nMAE = {bridge_results['full_mae']:.2f}\n"
             f"CV MAE = {bridge_results['cv_mae_mean']:.2f}",
             transform=ax1.transAxes, va="bottom", ha="left", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#ccc", alpha=0.9))

    # 4b: residuals vs predicted (heteroscedasticity check)
    ax2.scatter(y_pred, resid, alpha=0.35, s=14, color=COLOR_ACCENT, edgecolors="none")
    ax2.axhline(0, color="#C03A2B", linewidth=1.2, linestyle="--")
    # Lowess-ish moving average using rolling bins
    sorted_idx = np.argsort(y_pred)
    yp_s = y_pred[sorted_idx]
    rs_s = resid[sorted_idx]
    window = max(20, len(rs_s) // 30)
    if len(rs_s) >= window:
        mov_avg = pd.Series(rs_s).rolling(window, center=True, min_periods=5).mean()
        ax2.plot(yp_s, mov_avg, color="#0F6E56", linewidth=1.6, label=f"rolling mean (w={window})")
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.set_xlabel("Predicted DefR")
    ax2.set_ylabel("Residual (actual − predicted)")
    ax2.set_title("Residuals vs predicted")

    # 4c: Q-Q plot of residuals
    scipy_stats.probplot(resid, dist="norm", plot=ax3)
    # Restyle the auto-generated plot
    ax3.get_lines()[0].set_marker(".")
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[0].set_color(COLOR_ACCENT)
    ax3.get_lines()[0].set_alpha(0.5)
    ax3.get_lines()[1].set_color("#C03A2B")
    ax3.get_lines()[1].set_linewidth(1.4)
    ax3.set_title("Q-Q plot of residuals  (normality check)")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")

    # 4d: histogram of residuals with normal overlay
    ax4.hist(resid, bins=40, density=True, color=COLOR_ACCENT, alpha=0.65,
             edgecolor="white", linewidth=0.5)
    # Normal overlay
    mu, sigma = resid.mean(), resid.std()
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax4.plot(xs, scipy_stats.norm.pdf(xs, mu, sigma),
             color="#C03A2B", linewidth=1.6, label=f"N({mu:.1f}, {sigma:.1f})")
    ax4.axvline(0, color="#333", linestyle="--", linewidth=1.0, alpha=0.6)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Density")
    ax4.set_title("Residual distribution")
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.suptitle("Bridge regression diagnostics  —  DefR ~ aggregate features",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Plot 5: team style clustering ────────────────────────────────────
def plot_style_clustering(
    season_defr: pd.DataFrame,
    agg: pd.DataFrame,
    defr_match: pd.DataFrame,
    out_path: Path,
):
    """Scatter of season-average DefR vs possession, with annotations
    highlighting known team styles."""
    # Per-team season averages
    poss_by_team = (
        agg.groupby("team_name")
        .agg(avg_poss=("poss_pct", "mean"),
             avg_shots_against=("shots_against", "mean"))
        .reset_index()
    )
    style = season_defr.merge(poss_by_team, on="team_name", how="left")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Quadrant shading
    x_mid = style["avg_poss"].median()
    y_mid = 0
    xlim = (style["avg_poss"].min() - 2, style["avg_poss"].max() + 2)
    ylim = (style["avg_defr"].min() - 6, style["avg_defr"].max() + 8)

    # Quadrant background tint
    ax.axhspan(0, ylim[1], xmin=0, xmax=(x_mid - xlim[0]) / (xlim[1] - xlim[0]),
               color=COLOR_POS, alpha=0.05)
    ax.axhspan(ylim[0], 0, xmin=(x_mid - xlim[0]) / (xlim[1] - xlim[0]), xmax=1,
               color=COLOR_NEG, alpha=0.05)

    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(x_mid, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)

    # Scatter, sized by total defensive volume
    sizes = (style["total_actual"] / style["total_actual"].max()) * 600 + 80
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in style["avg_defr"]]
    ax.scatter(style["avg_poss"], style["avg_defr"],
               s=sizes, c=colors, alpha=0.65,
               edgecolors="white", linewidths=1.4)

    # Team labels with smart offset to avoid overlap
    for _, row in style.iterrows():
        ax.annotate(
            row["team_name"],
            xy=(row["avg_poss"], row["avg_defr"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=9, fontweight="medium",
            color="#222",
        )

    # Quadrant labels
    ax.text(xlim[0] + 0.5, ylim[1] - 1, "AGGRESSIVE PRESS\n(low poss, high DefR)",
            fontsize=10, color=COLOR_POS, alpha=0.85, fontweight="bold",
            va="top", ha="left")
    ax.text(xlim[1] - 0.5, ylim[1] - 1, "ACTIVE DEFENDERS\n(high poss, high DefR)",
            fontsize=10, color=COLOR_POS, alpha=0.85, fontweight="bold",
            va="top", ha="right")
    ax.text(xlim[0] + 0.5, ylim[0] + 1, "PASSIVE BLOCK\n(low poss, low DefR)",
            fontsize=10, color=COLOR_NEG, alpha=0.85, fontweight="bold",
            va="bottom", ha="left")
    ax.text(xlim[1] - 0.5, ylim[0] + 1, "POSSESSION DOMINANT\n(high poss, low DefR)",
            fontsize=10, color=COLOR_NEG, alpha=0.85, fontweight="bold",
            va="bottom", ha="right")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Average possession % per match")
    ax.set_ylabel("Season-average DefR score")
    ax.set_title("Team style clustering — DefR vs possession (Serie A 2017/18)\n"
                 "bubble size ∝ total defensive actions over the season",
                 pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Plot 6: footballistic validation ─────────────────────────────────
def plot_football_validation(
    season: pd.DataFrame,
    agg: pd.DataFrame,
    out_path: Path,
):
    """Annotated validation chart: known team identities → DefR placement.

    This is the central plausibility check of the whole approach. If the
    metric assigns labels that contradict known footballing reality, it's
    not capturing what we think it is.
    """
    poss_by_team = (
        agg.groupby("team_name")["poss_pct"].mean().rename("avg_poss")
    )
    style = season.merge(poss_by_team, on="team_name").sort_values("avg_defr", ascending=False)

    # Footballistic notes for selected teams (2017/18 context)
    annotations = {
        "Atalanta":      "Gasperini's aggressive\nman-marking press",
        "Fiorentina":    "Pioli's energetic\nmid-block",
        "Torino":        "Mihajlović — direct,\nphysical mid-block",
        "Napoli":        "Sarri-ball: dominant\npossession, low pressing demand",
        "Juventus":      "Allegri controlled\npragmatic dominance",
        "Lazio":         "Inzaghi possession\n+ counter-attack",
        "Sampdoria":     "Giampaolo positional\nplay",
        "Crotone":       "Relegated: scramble-\ndefending under siege",
    }

    fig, ax = plt.subplots(figsize=(13, 9))
    n = len(style)
    y_pos = np.arange(n)[::-1]
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in style["avg_defr"]]
    bars = ax.barh(y_pos, style["avg_defr"], color=colors, alpha=0.8,
                    edgecolor="white", linewidth=0.5, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(style["team_name"], fontsize=10)
    ax.axvline(0, color="#333", linewidth=1.0, linestyle="-", alpha=0.5)
    ax.set_xlabel("Season-average DefR score")
    ax.set_title("Footballistic validation — does DefR match known team identities?\n"
                 "Annotated styles based on 2017/18 manager and tactical context",
                 pad=14)

    # Value labels
    for yi, val in zip(y_pos, style["avg_defr"]):
        xpos = val + (1.0 if val >= 0 else -1.0)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=9, color="#333")

    # Annotation callouts for the named teams
    x_min, x_max = ax.get_xlim()
    callout_x = x_max + 8
    ax.set_xlim(x_min, callout_x + 25)

    for yi, team in zip(y_pos, style["team_name"]):
        note = annotations.get(team)
        if note is None:
            continue
        val = style.loc[style["team_name"] == team, "avg_defr"].iloc[0]
        # Connector line
        end_x = val + (1.5 if val >= 0 else -1.5)
        ax.annotate(
            note,
            xy=(end_x, yi),
            xytext=(callout_x, yi),
            fontsize=8.5, color="#333", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.7,
                            connectionstyle="arc3,rad=0"),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E0",
                      edgecolor="#D4A85A", linewidth=0.8, alpha=0.95),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Orchestrator ─────────────────────────────────────────────────────
def make_all_plots(
    zone_rates: pd.DataFrame,
    defr_match: pd.DataFrame,
    season: pd.DataFrame,
    agg: pd.DataFrame,
    bridge_results: dict,
    bridge_dataset: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, Path]:
    """Generate all plots and return a mapping name → path."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pitch_zones": plots_dir / "01_pitch_zones.png",
        "defr_distribution": plots_dir / "02_defr_distribution.png",
        "team_rankings": plots_dir / "03_team_rankings.png",
        "regression_diagnostics": plots_dir / "04_regression_diagnostics.png",
        "style_clustering": plots_dir / "05_style_clustering.png",
        "football_validation": plots_dir / "06_football_validation.png",
    }
    print("  Plotting pitch zones...")
    plot_pitch_zones(zone_rates, paths["pitch_zones"])
    print("  Plotting DefR distribution...")
    plot_defr_distribution(defr_match, season, paths["defr_distribution"])
    print("  Plotting team rankings...")
    plot_team_rankings(season, paths["team_rankings"])
    print("  Plotting regression diagnostics...")
    plot_regression_diagnostics(bridge_results, bridge_dataset, paths["regression_diagnostics"])
    print("  Plotting style clustering...")
    plot_style_clustering(season, agg, defr_match, paths["style_clustering"])
    print("  Plotting footballistic validation...")
    plot_football_validation(season, agg, paths["football_validation"])
    return paths
