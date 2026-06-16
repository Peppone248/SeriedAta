"""
visualization/model_plots.py — diagnostic plots for fitted models.

Runs AFTER training. Consumes the per-fold results list returned by
walk_forward_backtest() in train_model.py.

Each plot is a SIDE-BY-SIDE comparison of all models (linear vs xgboost),
because the interesting questions are now about RELATIVE behavior:
  - which model handles tails better
  - which model's errors are more predictable
  - which features each model leans on
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

DEFAULT_PLOT_DIR = "reports/plots/model"
COLORS = {"Linear (ridge)": "#3266ad", "XGBoost": "#d84a30"}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path);
    p.mkdir(parents=True, exist_ok=True);
    return p


def _pooled_predictions(results_by_model: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    """Concatenate per-fold predictions per model for plotting."""
    out = {}
    for label, results in results_by_model.items():
        frames = []
        for r in results:
            df = r["predictions"].copy()
            df["test_season"] = r["test_season"]
            frames.append(df)
        out[label] = pd.concat(frames, ignore_index=True)
    return out


# ─── 1. predicted vs actual ────────────────────────────────────────────────────

def plot_predicted_vs_actual(
        results_by_model: dict[str, list[dict]],
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Scatter of predicted vs actual, per model. Perfect predictions sit on
    the diagonal y=x.

    Question answered: where does each model systematically over- or under-predict?
    Look for:
      - points clustered tightly around the diagonal -> good model
      - cloud tilted off-diagonal -> systematic bias (regression to the mean
        often shows here: low actuals predicted too high, high actuals too low)
      - tails curl away from diagonal -> model can't reach extreme values,
        a common limitation of linear models on truncated targets (0-15 here)
      - horizontal bands in the actual values -> the target is integer-valued,
        causing visible stripes; this is normal, not a model artifact
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "01_predicted_vs_actual.png"

    pooled = _pooled_predictions(results_by_model)
    n = len(pooled)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=True, sharey=True)
    if n == 1: axes = [axes]

    for ax, (label, df) in zip(axes, pooled.items()):
        ax.scatter(df["actual"], df["predicted"],
                   alpha=0.25, s=10, color=COLORS.get(label, "#888"))
        lo, hi = -1, 16
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")

        # least-squares fit through the cloud — divergence from y=x = systematic bias
        coef = np.polyfit(df["actual"], df["predicted"], 1)
        x_line = np.linspace(0, 15, 50)
        ax.plot(x_line, np.polyval(coef, x_line),
                color="#5ab27a", linewidth=2,
                label=f"fit: slope={coef[0]:.2f}")

        from team_trend.models.linear_regression import mae as _mae, r2 as _r2
        ax.set_title(f"{label}\nMAE={_mae(df['actual'], df['predicted']):.2f}  "
                     f"R²={_r2(df['actual'].values, df['predicted'].values):.2f}")
        ax.set_xlabel("Actual next-5 points")
        ax.set_ylabel("Predicted next-5 points")
        ax.set_xlim(lo, hi);
        ax.set_ylim(lo, hi)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Predicted vs Actual — pooled across folds", fontsize=13)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [01] predicted vs actual saved -> {out}")
    return out


# ─── 2. residual distribution ──────────────────────────────────────────────────

def plot_residual_distribution(
        results_by_model: dict[str, list[dict]],
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Histogram of residuals (actual - predicted) per model, overlaid.

    Question answered: are errors centered, symmetric, well-behaved?
    Look for:
      - centred at zero -> model is unbiased on average
      - symmetric, bell-shaped -> errors are random noise (residuals
        looking Gaussian is a good sign)
      - heavy tails / asymmetry -> model misses systematically on certain
        kinds of rows; worth investigating the residual outliers
      - one model's distribution clearly tighter than the other -> that
        model has lower variance regardless of average MAE
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "02_residual_distribution.png"

    pooled = _pooled_predictions(results_by_model)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in pooled.items():
        sns.histplot(df["residual"], bins=40, kde=True,
                     ax=ax, alpha=0.45,
                     color=COLORS.get(label, "#888"), label=label,
                     stat="density")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution — all folds pooled")
    ax.legend()
    plt.savefig(out)
    plt.close(fig)

    for label, df in pooled.items():
        r = df["residual"]
        print(f"  [02] {label}: mean={r.mean():+.3f}  std={r.std():.3f}  "
              f"skew={r.skew():+.2f}")
    return out


# ─── 3. residuals vs feature value (linearity check on residuals) ──────────────

def plot_residuals_vs_features(
        results_by_model: dict[str, list[dict]],
        gold: pd.DataFrame,
        features: list[str],
        save_dir: str = DEFAULT_PLOT_DIR,
        top_k: int = 6,
) -> Path:
    """
    For each of the top-k most-correlated features, scatter the residuals.

    Question answered: is there structure left in the residuals related to
    the features? Patterns here = the model is leaving signal on the table.

    For each feature, compare two models side-by-side. A LOWESS curve through
    the residuals shows whether the average error depends on the feature value.
    Look for:
      - flat horizontal band of residuals -> the model has extracted all the
        linear/non-linear signal that feature offers
      - U-shape or curve -> model misses non-linear structure
      - linear residual model curves where XGBoost is flat -> exactly the
        case for non-linear models to help
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "03_residuals_vs_features.png"

    # pool predictions and join back to feature values via (team, matchweek)
    pooled = _pooled_predictions(results_by_model)
    feat_cols = [c for c in features if c in gold.columns]
    feat_df = gold[["team", "matchweek", "season"] + feat_cols].copy()
    for label in pooled:
        pooled[label] = pooled[label].merge(
            feat_df.rename(columns={"season": "test_season"}),
            on=["team", "matchweek", "test_season"], how="left",
        )

    # rank features by |correlation with the target| (in the gold dataset)
    target_col = "next_5_matchweek_points"
    corrs = (
        gold[feat_cols]
            .apply(lambda c: c.corr(gold[target_col]))
            .abs()
            .sort_values(ascending=False)
    )
    chosen = corrs.head(top_k).index.tolist()

    nrows = len(chosen);
    ncols = len(pooled)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             squeeze=False, sharey="row")

    for i, feat in enumerate(chosen):
        for j, (label, df) in enumerate(pooled.items()):
            ax = axes[i, j]
            if feat not in df.columns or df[feat].isna().all():
                ax.set_visible(False);
                continue
            d = df.dropna(subset=[feat, "residual"])
            ax.scatter(d[feat], d["residual"], alpha=0.20, s=8,
                       color=COLORS.get(label, "#888"))
            ax.axhline(0, color="black", linewidth=0.8)

            # LOWESS curve (or polyfit fallback)
            x, y = d[feat].to_numpy(float), d["residual"].to_numpy(float)
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                s = lowess(y, x, frac=0.4, return_sorted=True)
                ax.plot(s[:, 0], s[:, 1], color="#5ab27a", linewidth=2,
                        label="LOWESS")
            except ImportError:
                coef = np.polyfit(x, y, 2)
                xg = np.linspace(x.min(), x.max(), 100)
                ax.plot(xg, np.polyval(coef, xg), color="#5ab27a",
                        linewidth=2, label="quadratic")

            if i == 0: ax.set_title(label)
            if j == 0: ax.set_ylabel(f"resid\n{feat}", fontsize=9)
            ax.set_xlabel(feat if i == nrows - 1 else "", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)

    plt.suptitle("Residuals vs top features (flat = good)", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [03] residuals vs features saved -> {out}")
    return out


# ─── 4. per-team residual heatmap (drift through season) ───────────────────────

def plot_team_residual_heatmap(
        results_by_model: dict[str, list[dict]],
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Heatmap of mean residual per (team, matchweek-bin), one panel per model.

    Question answered: are there teams or periods of the season where one
    model systematically wins or loses against the other?
    Look for:
      - all-blue or all-red row -> systematic bias for that team (always
        over- or under-predicted)
      - column patterns -> bias at certain stages of the season
      - models with different patterns -> they're learning different things,
        which is the foundation for an ensemble
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "04_team_residual_heatmap.png"

    pooled = _pooled_predictions(results_by_model)

    n = len(pooled)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 9), squeeze=False)

    # discretize matchweek for readability (bins of 5)
    for ax, (label, df) in zip(axes[0], pooled.items()):
        d = df.copy()
        d["mw_bin"] = (d["matchweek"] // 5) * 5

        pivot = (
            d.pivot_table(index="team", columns="mw_bin",
                          values="residual", aggfunc="mean")
                .sort_index()
        )
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-5, vmax=5, cbar_kws={"label": "mean residual"})
        ax.set_title(f"{label}\n(red = under-predicts, blue = over-predicts)")
        ax.set_xlabel("Matchweek (binned)")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [04] team residual heatmap saved -> {out}")
    return out


# ─── 5. team prediction trajectories ───────────────────────────────────────────

def plot_team_trajectories(
        results_by_model: dict[str, list[dict]],
        save_dir: str = DEFAULT_PLOT_DIR,
        n_teams: int = 6,
        focus_model: str = "XGBoost (production)",
) -> Path:
    """
    For the most recent fold, plot the predicted vs actual next_5_points
    trajectory across matchweeks, for selected teams.

    Question answered: how does the model TRACK each team through the season?
    Look for:
      - predicted line shadowing the actual line -> model captures team's
        momentum
      - predicted line flat while actual oscillates -> model defaults to a
        constant per-team prediction (mean compression)
      - predicted line consistently above/below actual -> systematic bias
        for that team (matches the team residual heatmap pattern)

    Teams selected: top 3 + bottom 3 by mean actual in the test fold,
    so the plot spans the full spectrum.
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "05_team_trajectories.png"

    if focus_model not in results_by_model:
        focus_model = list(results_by_model.keys())[0]
    last = results_by_model[focus_model][-1]
    preds = last["predictions"].copy()

    team_mean = preds.groupby("team")["actual"].mean().sort_values(ascending=False)
    top = team_mean.head(n_teams // 2).index.tolist()
    bot = team_mean.tail(n_teams // 2).index.tolist()
    chosen = top + bot

    ncols = 3
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             sharey=True, sharex=True)
    axes = np.array(axes).flatten()

    for i, team in enumerate(chosen):
        ax = axes[i]
        t = preds[preds["team"] == team].sort_values("matchweek")
        ax.plot(t["matchweek"], t["actual"], "o-",
                color="#3266ad", linewidth=2, markersize=5, label="actual")
        ax.plot(t["matchweek"], t["predicted"], "s--",
                color="#d84a30", linewidth=2, markersize=5, label="predicted")
        ax.set_title(team, fontweight="bold")
        ax.set_xlabel("Matchweek")
        ax.set_ylabel("Next-5 points" if i % ncols == 0 else "")
        ax.set_ylim(-1, 16)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Team trajectories — {focus_model} (test={last['test_season']})\n"
        "top 3 + bottom 3 by actual mean",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [05] team trajectories saved -> {out}")
    return out


# ─── 6. team prediction trajectories WITH quantile intervals ───────────────────

def plot_team_trajectories_with_intervals(
        results_by_model: dict[str, list[dict]],
        save_dir: str = DEFAULT_PLOT_DIR,
        n_teams: int = 6,
        quantile_model: str = "XGBoost Quantile",
) -> Path | None:
    """
    Like the standard team trajectory plot but with the quantile prediction
    interval shaded around the median line. Shows for each team:
      - actual next-5 points (blue line + markers)
      - median prediction (orange dashed)
      - 80% interval band (orange shaded region between q_low and q_high)

    Question answered: where does the model report HIGH confidence vs LOW?
    Look for:
      - tight band around volatile teams -> model is overconfident
      - wide band that contains the actuals -> model honestly admits uncertainty
      - actuals consistently outside the band -> miscalibration
    """
    if quantile_model not in results_by_model:
        logger.info("Skipping quantile trajectory plot — '%s' not in results",
                    quantile_model)
        return None

    out_dir = _ensure_dir(save_dir)
    out = out_dir / "06_team_trajectories_quantile.png"

    last = results_by_model[quantile_model][-1]
    preds = last["predictions"].copy()

    if not {"q_low", "q_median", "q_high"}.issubset(preds.columns):
        logger.warning("Quantile columns missing — skipping interval plot")
        return None

    team_mean = preds.groupby("team")["actual"].mean().sort_values(ascending=False)
    chosen = team_mean.head(n_teams // 2).index.tolist() \
             + team_mean.tail(n_teams // 2).index.tolist()

    ncols = 3
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             sharey=True, sharex=True)
    axes = np.array(axes).flatten()

    for i, team in enumerate(chosen):
        ax = axes[i]
        t = preds[preds["team"] == team].sort_values("matchweek")
        ax.fill_between(
            t["matchweek"], t["q_low"], t["q_high"],
            color="#d84a30", alpha=0.20, label="80% interval",
        )
        ax.plot(t["matchweek"], t["q_median"], "s--",
                color="#d84a30", linewidth=1.8, markersize=4, label="median pred")
        ax.plot(t["matchweek"], t["actual"], "o-",
                color="#3266ad", linewidth=2, markersize=5, label="actual")

        # compute per-team coverage and sharpness for the title
        in_band = (
                (t["actual"] >= t["q_low"]) & (t["actual"] <= t["q_high"])
        ).mean()
        width = (t["q_high"] - t["q_low"]).mean()
        ax.set_title(f"{team}   cov={in_band:.0%}   width={width:.1f}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Matchweek")
        ax.set_ylabel("Next-5 points" if i % ncols == 0 else "")
        ax.set_ylim(-1, 16)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Team trajectories with 80% prediction intervals — "
        f"{quantile_model} (test={last['test_season']})",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [06] quantile trajectories saved -> {out}")
    return out


# ─── 7. feature group ablation results ─────────────────────────────────────────

def plot_ablation_results(
        audit_df: pd.DataFrame,
        baseline_mae: float,
        save_dir: str = DEFAULT_PLOT_DIR,
) -> Path:
    """
    Horizontal bar chart of MAE delta per feature group removed.

    Sign convention:
      red bars   (positive delta) -> group is USEFUL (removing hurts)
      green bars (negative delta) -> group is HARMFUL (removing helps)
      gray bars  (~zero delta)    -> group is REDUNDANT (no effect)
    """
    out_dir = _ensure_dir(save_dir)
    out = out_dir / "07_ablation_results.png"

    df = audit_df.sort_values("delta", ascending=True).reset_index(drop=True)
    colors = [
        "#d84a30" if d > 0.005 else
        "#5ab27a" if d < -0.005 else
        "#888888"
        for d in df["delta"]
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.45)))
    bars = ax.barh(df["group_removed"], df["delta"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MAE delta when group is REMOVED  (positive = group was useful)")
    ax.set_title(f"Feature group ablation (baseline MAE = {baseline_mae:.3f})")
    ax.grid(alpha=0.3, axis="x")

    for bar, d in zip(bars, df["delta"]):
        width = bar.get_width()
        offset = 0.002 if width >= 0 else -0.002
        ax.text(
            width + offset, bar.get_y() + bar.get_height() / 2,
            f"{d:+.3f}", va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"  [07] ablation results saved -> {out}")
    return out


# ─── orchestrator ──────────────────────────────────────────────────────────────

def run_all_model_plots(
        results_by_model: dict[str, list[dict]],
        gold: pd.DataFrame,
        features: list[str],
        save_dir: str = DEFAULT_PLOT_DIR,
) -> list[Path]:
    _setup_style()
    print(f"\n=== MODEL PLOTS -> {save_dir} ===")
    paths = [
        plot_predicted_vs_actual(results_by_model, save_dir),
        plot_residual_distribution(results_by_model, save_dir),
        plot_residuals_vs_features(results_by_model, gold, features, save_dir),
        plot_team_residual_heatmap(results_by_model, save_dir),
        plot_team_trajectories(results_by_model, save_dir),
    ]
    p6 = plot_team_trajectories_with_intervals(results_by_model, save_dir)
    if p6 is not None:
        paths.append(p6)
    print(f"=== {len(paths)} plots saved ===")
    return paths
