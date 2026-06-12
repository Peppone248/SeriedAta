"""
visualization/dataset_plots.py — diagnostic plots for the Gold dataset.

Runs AFTER Gold is built and BEFORE modelling. Output saved to disk so a
run leaves a reproducible artefact, the same way Bronze/Silver/Gold do.

Each function:
  - takes the Gold DataFrame + a save directory
  - produces one figure
  - returns the saved file path
  - prints a one-line insight summary to stdout

The orchestrator run_all_dataset_plots() calls them in sequence and writes
everything to reports/plots/dataset/ by default.

Design choices:
  - matplotlib + seaborn only (no fancy deps)
  - one figure per question — no busy multi-panel charts
  - consistent style set once in _setup_style()
  - resolution 150 dpi (sharp on screen, small file size)
  - non-interactive backend so the script runs headless / in CI
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DEFAULT_PLOT_DIR = "reports/plots/dataset"
TARGET_COL = "next_5_matchweek_points"


# ─── style ─────────────────────────────────────────────────────────────────────

def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── 1. target distribution ────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, save_dir: str = DEFAULT_PLOT_DIR) -> Path:
    """
    Histogram of next_5_matchweek_points (0..15) with mean/median markers.

    Question answered: is the target shape suitable for MSE-based regression?
    Look for:
      - rough symmetry around the mean -> linear regression is reasonable
      - bimodality (e.g. two peaks at ~3 and ~10) -> would suggest a mixture
        model or stratification by team tier
      - hard skew -> Poisson-style loss may fit better than MSE
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "01_target_distribution.png"

    y = df[TARGET_COL].dropna()
    mean = y.mean()
    med = y.median()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(y, bins=16, kde=True, ax=ax, color="#3266ad", alpha=0.7)
    ax.axvline(mean, color="#d84a30", linestyle="--", linewidth=2,
               label=f"mean={mean:.2f}")
    ax.axvline(med, color="#5ab27a", linestyle=":", linewidth=2,
               label=f"median={med:.2f}")
    ax.set_xlabel("Points in next 5 matchweeks")
    ax.set_ylabel("Count of (team, matchweek) rows")
    ax.set_title("Target distribution — next_5_matchweek_points")
    ax.legend()
    plt.savefig(out)
    plt.close(fig)

    print(f"  [01] target: mean={mean:.2f}  median={med:.2f}  "
          f"std={y.std():.2f}  range=[{int(y.min())}, {int(y.max())}]")
    return out


# ─── 2. target by matchweek ────────────────────────────────────────────────────

def plot_target_by_matchweek(df: pd.DataFrame, save_dir: str = DEFAULT_PLOT_DIR) -> Path:
    """
    Boxplot of the target across matchweeks 1..33 (each row pools all teams
    and all seasons for that matchweek).

    Question answered: does the target distribution shift through the season?
    Look for:
      - rising/falling median -> end-of-season effects (pressure, rotations
        in safe teams) the model would need to capture via season_progress
      - widening boxes late in season -> variance grows as stakes diverge,
        making predictions intrinsically harder
      - flat throughout -> the calendar position is not the dominant driver
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "02_target_by_matchweek.png"

    sub = df.dropna(subset=[TARGET_COL, "matchweek"])

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=sub, x="matchweek", y=TARGET_COL,
                ax=ax, color="#3266ad", fliersize=2, linewidth=0.8)
    ax.set_xlabel("Matchweek")
    ax.set_ylabel("Points in next 5 matchweeks")
    ax.set_title("Target by matchweek (all teams, all seasons)")
    # show every other tick to keep the x-axis readable
    for label in ax.get_xticklabels()[1::2]:
        label.set_visible(False)
    plt.savefig(out)
    plt.close(fig)

    medians = sub.groupby("matchweek")[TARGET_COL].median()
    print(f"  [02] target by MW: median range [{medians.min():.1f}, {medians.max():.1f}], "
          f"delta = {medians.max() - medians.min():.2f}")
    return out


# ─── 3. target by team ─────────────────────────────────────────────────────────

def plot_target_by_team(df: pd.DataFrame, save_dir: str = DEFAULT_PLOT_DIR) -> Path:
    """
    Boxplot of the target per team, sorted by median.

    Question answered: how heterogeneous are teams in our population?
    Look for:
      - wide spread across teams -> the model needs team-aware features
        (and we DO encode 'team' indirectly through league_position + cum_*)
      - some teams with tight, low-median boxes -> small clubs that are
        consistently bottom-half: low variance, low values, easy to predict
      - some teams with wide boxes -> volatile clubs: hardest to predict
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "03_target_by_team.png"

    sub = df.dropna(subset=[TARGET_COL, "team"])
    order = sub.groupby("team")[TARGET_COL].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(12, max(5, len(order) * 0.25)))
    sns.boxplot(data=sub, y="team", x=TARGET_COL, order=order,
                ax=ax, color="#5ab27a", fliersize=2, linewidth=0.8)
    ax.set_xlabel("Points in next 5 matchweeks")
    ax.set_ylabel("")
    ax.set_title("Target by team (sorted by median)")
    plt.savefig(out)
    plt.close(fig)

    stats = sub.groupby("team")[TARGET_COL].agg(["median", "std"]).round(2)
    most_variable = stats["std"].idxmax()
    least_variable = stats["std"].idxmin()
    print(f"  [03] team variance: most variable = {most_variable} (std={stats.loc[most_variable, 'std']}), "
          f"least = {least_variable} (std={stats.loc[least_variable, 'std']})")
    return out


# ─── 4. feature correlation heatmap ────────────────────────────────────────────

def plot_feature_correlation(
        df: pd.DataFrame,
        features: list[str] | None = None,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Pearson correlation heatmap over the numeric pre-match features.

    Question answered: which features are redundant by collinearity?
    Look for:
      - dark red blocks far from the diagonal -> highly correlated pairs.
        E.g. roll5_shots & roll5_shots_on_target are correlated by construction.
      - features near-uncorrelated with everything -> either unique signal
        or noise; check their correlation WITH the target (next plot).
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "04_feature_correlation.png"

    if features is None:
        features = [c for c in df.columns
                    if c.startswith("roll5_") or c in (
                        "is_home", "days_rest", "cum_points",
                        "cum_goal_diff", "league_position", "season_progress",
                    )]
    features = [c for c in features if c in df.columns]

    corr = df[features].corr().round(2)

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.45),
                                    max(8, len(features) * 0.4)))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson r"},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor="white")
    ax.set_title("Feature correlation matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.savefig(out)
    plt.close(fig)

    # find redundant pairs (|r| > 0.85, off-diagonal)
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    redundant = (
        upper.stack()
            .abs()
            .sort_values(ascending=False)
            .head(5)
    )
    print(f"  [04] top redundant pairs (|r| > 0.85):")
    for (a, b), r in redundant.items():
        if abs(r) > 0.85:
            print(f"        {a} <-> {b}: r={r:+.2f}")
    return out


# ─── 5. feature-target correlation ─────────────────────────────────────────────

def plot_feature_target_correlation(
        df: pd.DataFrame,
        features: list[str] | None = None,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Bar chart of each feature's Pearson correlation with the target.

    Question answered: which features carry signal toward the prediction?
    Look for:
      - high |r| -> direct linear signal
      - r near 0 -> either non-linear relation hiding (check residual plots
        later) or pure noise
      - sign of the correlation should match intuition (e.g. roll5_points
        positive, league_position negative)
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "05_feature_target_correlation.png"

    if features is None:
        features = [c for c in df.columns
                    if c.startswith("roll5_") or c in (
                        "is_home", "days_rest", "cum_points",
                        "cum_goal_diff", "league_position", "season_progress",
                    )]
    features = [c for c in features if c in df.columns]

    sub = df.dropna(subset=[TARGET_COL])
    corrs = (
        sub[features]
            .apply(lambda c: c.corr(sub[TARGET_COL]))
            .sort_values()
    )

    colors = ["#d84a30" if v < 0 else "#3266ad" for v in corrs.values]
    fig, ax = plt.subplots(figsize=(9, max(5, len(corrs) * 0.3)))
    ax.barh(corrs.index, corrs.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson r vs {TARGET_COL}")
    ax.set_title("Feature correlation with the target")
    plt.savefig(out)
    plt.close(fig)

    top_pos = corrs.tail(3).round(2)
    top_neg = corrs.head(3).round(2)
    print(f"  [05] strongest positive: {', '.join(f'{n} ({v:+.2f})' for n, v in top_pos.items())}")
    print(f"  [05] strongest negative: {', '.join(f'{n} ({v:+.2f})' for n, v in top_neg.items())}")
    return out


# ─── 7. feature vs target scatter (top correlated) ─────────────────────────────

def plot_top_feature_scatters(
        df: pd.DataFrame,
        features: list[str] | None = None,
        top_k: int = 6,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Scatter of each top-correlated feature vs the target, with linear and
    LOWESS (local non-linear) trend lines.

    Question answered: are the feature-target relationships actually LINEAR,
    or does the bar chart hide non-linear structure?
    Look for:
      - linear (red) and LOWESS (green) lines roughly overlap -> linear
        regression is the right family for that feature
      - LOWESS curves while the linear line stays straight -> the feature
        has non-linear signal that ridge regression CAN'T capture but a
        tree-based model would; candidate for piecewise or polynomial
        feature engineering
      - LOWESS saturates at extremes -> feature has diminishing returns
        (e.g., league_position 18-20 may all map to similar low future points)
      - high vertical scatter -> high irreducible noise at that feature value
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "07_top_feature_scatters.png"

    if features is None:
        features = [c for c in df.columns
                    if c.startswith("roll5_") or c in (
                        "is_home", "days_rest", "cum_points",
                        "cum_goal_diff", "league_position", "season_progress",
                    )]
    features = [c for c in features if c in df.columns]

    sub = df.dropna(subset=[TARGET_COL]).copy()
    # pick top_k by |correlation|
    corrs = sub[features].apply(lambda c: c.corr(sub[TARGET_COL])).abs().sort_values(ascending=False)
    top_features = corrs.head(top_k).index.tolist()

    ncols = 3
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        # use a sample if too many points (faster, less overplotting)
        s = sub.sample(min(2000, len(sub)), random_state=42) if len(sub) > 2000 else sub
        ax.scatter(s[feat], s[TARGET_COL], alpha=0.25, s=10, color="#3266ad")

        # linear fit
        x_vals = s[feat].to_numpy(float)
        y_vals = s[TARGET_COL].to_numpy(float)
        mask = ~np.isnan(x_vals)
        x_vals, y_vals = x_vals[mask], y_vals[mask]
        if len(x_vals) > 5:
            coef = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, np.polyval(coef, x_line),
                    color="#d84a30", linewidth=2, label="linear")

            # LOWESS via statsmodels if available, else degree-2 polyfit fallback
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(y_vals, x_vals, frac=0.4, return_sorted=True)
                ax.plot(smoothed[:, 0], smoothed[:, 1],
                        color="#5ab27a", linewidth=2, label="LOWESS")
            except ImportError:
                coef2 = np.polyfit(x_vals, y_vals, 2)
                ax.plot(x_line, np.polyval(coef2, x_line),
                        color="#5ab27a", linewidth=2, linestyle="--",
                        label="quadratic")

        r = corrs[feat]
        ax.set_title(f"{feat}\n(|r|={r:.2f})", fontsize=10)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel(TARGET_COL, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Top features vs target — linearity check", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)

    print(f"  [07] plotted {top_k} top features: "
          f"{', '.join(top_features[:3])} ...")
    return out


# ─── orchestrator ──────────────────────────────────────────────────────────────

def run_all_dataset_plots(
        df: pd.DataFrame,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> list[Path]:
    """Run every dataset plot and return the list of saved paths."""
    _setup_style()
    print(f"\n=== DATASET PLOTS -> {save_dir} ===")
    paths = [
        plot_target_distribution(df, save_dir),
        plot_target_by_matchweek(df, save_dir),
        plot_target_by_team(df, save_dir),
        plot_feature_correlation(df, save_dir=save_dir),
        plot_feature_target_correlation(df, save_dir=save_dir),
        plot_missingness_by_matchweek(df, save_dir=save_dir),
        plot_top_feature_scatters(df, save_dir=save_dir),
    ]
    print(f"=== {len(paths)} plots saved ===")
    return paths
