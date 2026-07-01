"""Validation plots for the DefR injection step.

Generates 5 PNG plots:
    1. bridge_r2_comparison.png   — original 10-feat vs reduced 7-feat bridge
    2. fbref_season_rankings.png  — 2024/25 FBref DefR rankings (quality, not style)
    3. correlation_heatmap.png    — last_5_defr_proxy vs existing features
    4. walkforward_f1.png         — per-fold F1 across conditions
    5. standalone_value.png       — standalone vs in-pipeline marginal contribution

All plots use the same style established in the original analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INJ = ROOT / "output/injection"
VAL = ROOT / "output/validation"
OUT = ROOT / "output/validation/plots"
OUT.mkdir(parents=True, exist_ok=True)

# ─── shared style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLOR_POS = "#0F6E56"
COLOR_NEG = "#C03A2B"
COLOR_NEUTRAL = "#888888"
COLOR_ACCENT = "#1E5A99"
COLOR_AMBER = "#BA7517"


# ─── plot 1: bridge comparison ────────────────────────────────────────
def plot_bridge_comparison():
    with open(ROOT / "output/data/bridge_regression.json") as f:
        orig = json.load(f)
    with open(INJ / "bridge_regression_fbref.json") as f:
        new = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.6]})

    # 1a: R² bars
    metrics = ["CV R²", "Full R²"]
    orig_vals = [orig["cv_r2_mean"], orig["full_r2"]]
    new_vals = [new["cv_r2_mean"], new["full_r2"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w/2, orig_vals, w, label="Original (10 features)",
            color=COLOR_ACCENT, alpha=0.85, edgecolor="white", lw=0.5)
    ax1.bar(x + w/2, new_vals, w, label="FBref-only (7 features)",
            color=COLOR_AMBER, alpha=0.85, edgecolor="white", lw=0.5)
    # value labels
    for i, (o, n) in enumerate(zip(orig_vals, new_vals)):
        ax1.text(i - w/2, o + 0.015, f"{o:.3f}", ha="center", fontsize=9)
        ax1.text(i + w/2, n + 0.015, f"{n:.3f}", ha="center", fontsize=9)
        ax1.text(i, max(o, n) + 0.07, f"Δ = {n-o:+.3f}", ha="center",
                 fontsize=10, color=COLOR_NEG, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(metrics)
    ax1.set_ylabel("R²")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("Bridge regression performance\n(after dropping FBref-incompatible features)")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # 1b: coefficient comparison
    orig_coefs = orig["coefficients"]
    new_coefs = new["coefficients"]
    all_feats = list(orig_coefs.keys())
    orig_vec = [orig_coefs[f] for f in all_feats]
    new_vec = [new_coefs.get(f, 0) for f in all_feats]

    y = np.arange(len(all_feats))
    ax2.barh(y - 0.2, orig_vec, 0.4, label="Original",
             color=COLOR_ACCENT, alpha=0.85, edgecolor="white", lw=0.5)
    ax2.barh(y + 0.2, new_vec, 0.4, label="FBref-only",
             color=COLOR_AMBER, alpha=0.85, edgecolor="white", lw=0.5)
    ax2.axvline(0, color="#333", lw=0.8, ls="-", alpha=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(all_feats, fontsize=9)
    ax2.set_xlabel("Standardized coefficient")
    ax2.set_title("Coefficient drift: what the reduced bridge learns from the same data")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Annotate the dramatic flips
    for feat in ["poss_pct", "n_opp_passes"]:
        if feat in all_feats:
            i = all_feats.index(feat)
            ax2.annotate("",
                xy=(new_coefs.get(feat, 0), i + 0.2),
                xytext=(orig_coefs[feat], i - 0.2),
                arrowprops=dict(arrowstyle="->", color=COLOR_NEG, lw=1.5, alpha=0.6))

    plt.tight_layout()
    plt.savefig(OUT / "01_bridge_comparison.png")
    plt.close()


# ─── plot 2: fbref 2024 season rankings ───────────────────────────────
def plot_fbref_rankings():
    df = pd.read_parquet(INJ / "fbref_with_defr.parquet")
    s = (df[df["season"] == 2024]
         .groupby("team")["defr_proxy"]
         .agg(["mean", "std", "count"])
         .reset_index())
    s = s[s["count"] >= 30].sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in s["mean"]]
    se = s["std"] / np.sqrt(s["count"])
    ax.barh(range(len(s)), s["mean"], xerr=se, color=colors, alpha=0.85,
            edgecolor="white", lw=0.5, error_kw=dict(ecolor="#444", capsize=2, lw=0.7))
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["team"], fontsize=9.5)
    ax.axvline(0, color="#333", lw=1.0, alpha=0.6)
    ax.set_xlabel("FBref-derived DefR proxy (season average)")
    ax.set_title("FBref Serie A 2024/25 — DefR proxy season rankings\n"
                 "(now correlates with team quality, not defensive style)",
                 pad=12)

    for i, v in enumerate(s["mean"]):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)

    # Add a note callout
    ax.text(0.02, 0.97,
            "Note: with n_opp_passes dropped from the bridge,\n"
            "the proxy now scores top teams (Inter, Bologna,\n"
            "Juventus) high and relegation candidates low.\n"
            "Contrast with Wyscout 2017/18 where Atalanta\n"
            "(aggressive press) was high and Napoli\n"
            "(possession-dominant) was lowest.",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E0",
                      edgecolor="#D4A85A", lw=0.7, alpha=0.95))

    plt.tight_layout()
    plt.savefig(OUT / "02_fbref_rankings.png")
    plt.close()


# ─── plot 3: correlation heatmap ──────────────────────────────────────
def plot_correlation():
    # Re-compute correlations on the dataset
    import sys, os
    sys.path.insert(0, str(ROOT.parent / "matches_classification"))
    os.chdir(str(ROOT.parent / "matches_classification"))

    from pipeline import run_pipeline
    from features import add_new_features, add_parity_features

    out = run_pipeline("data/raw/matches_seriea.csv", save=False)
    df = out["raw_df"].copy()
    df = add_new_features(df); df = add_parity_features(df)
    df["date"] = pd.to_datetime(df["date"])

    defr_df = pd.read_parquet(INJ / "fbref_with_defr.parquet")
    defr_df["date"] = pd.to_datetime(defr_df["date"])
    df = df.merge(defr_df[["date", "team", "defr_proxy", "last_5_defr_proxy"]],
                  on=["date", "team"], how="left")

    key_feats = ["cum_avg_points", "cum_avg_xg", "cum_avg_xga",
                 "strength_points_diff", "strength_xg_diff", "strength_xga_diff",
                 "weighted_form", "last_5_points", "last_5_xg",
                 "last_5_goal_diff", "form_consistency"]
    sub = df.dropna(subset=key_feats + ["last_5_defr_proxy"])
    corr = sub[key_feats + ["last_5_defr_proxy"]].corr()["last_5_defr_proxy"].drop("last_5_defr_proxy")
    corr = corr.sort_values()

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in corr.values]
    ax.barh(range(len(corr)), corr.values, color=colors, alpha=0.85,
            edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=10)
    ax.axvline(0, color="#333", lw=0.8, alpha=0.6)
    ax.set_xlabel("Pearson correlation with last_5_defr_proxy")
    ax.set_title("Why the proxy is redundant: it's highly correlated with existing features\n"
                 "(particularly the cumulative strength and form measures)",
                 pad=12)

    for i, v in enumerate(corr.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)

    ax.set_xlim(-0.7, 0.7)
    # Vertical guides
    for thresh in [-0.5, -0.3, 0.3, 0.5]:
        ax.axvline(thresh, color="#aaa", lw=0.4, ls=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUT / "03_correlation.png")
    plt.close()


# ─── plot 4: walk-forward F1 per fold ─────────────────────────────────
def plot_walkforward():
    df = pd.read_csv(VAL / "all_folds.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    pivot = df.pivot(index="test_season", columns="condition", values="f1_macro")
    seasons = pivot.index.astype(str)
    x = np.arange(len(seasons))
    width = 0.2
    conditions = ["baseline", "+proxy_rolling", "+proxy_match", "+proxy_both"]
    colors = [COLOR_ACCENT, COLOR_POS, COLOR_NEG, COLOR_AMBER]
    for i, (cond, col) in enumerate(zip(conditions, colors)):
        ax1.bar(x + (i - 1.5) * width, pivot[cond], width, label=cond,
                color=col, alpha=0.85, edgecolor="white", lw=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(seasons)
    ax1.set_xlabel("Test season")
    ax1.set_ylabel("F1 macro")
    ax1.set_title("Walk-forward F1 macro per fold")
    ax1.set_ylim(0.40, 0.55)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax1.axhline(pivot["baseline"].mean(), color=COLOR_ACCENT, ls="--", lw=0.8, alpha=0.5)

    # right: cumulative delta
    delta_match = pivot["+proxy_match"] - pivot["baseline"]
    delta_rolling = pivot["+proxy_rolling"] - pivot["baseline"]
    ax2.plot(seasons, delta_match, marker="o", color=COLOR_NEG, lw=2,
             label="+proxy_match", markersize=8)
    ax2.plot(seasons, delta_rolling, marker="s", color=COLOR_POS, lw=2,
             label="+proxy_rolling", markersize=8)
    ax2.axhline(0, color="#333", lw=1.0, alpha=0.7)
    ax2.fill_between(seasons, delta_match, 0, where=(delta_match < 0),
                      color=COLOR_NEG, alpha=0.15)
    ax2.set_ylabel("ΔF1 vs baseline")
    ax2.set_xlabel("Test season")
    ax2.set_title("F1 delta per fold vs baseline\n(0 = no change)")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax2.set_ylim(-0.015, 0.015)

    plt.tight_layout()
    plt.savefig(OUT / "04_walkforward.png")
    plt.close()


# ─── plot 5: standalone vs in-pipeline ────────────────────────────────
def plot_standalone():
    """Bar chart showing standalone gain (huge) vs in-pipeline gain (zero).

    The standalone numbers were measured in the sanity check; we replay
    them here statically for visualization.
    """
    standalone_deltas = {2022: +0.0501, 2023: +0.0529, 2024: +0.0763, 2025: +0.0246}
    pipeline_deltas = {2022: 0.0, 2023: 0.0, 2024: 0.0, 2025: 0.0}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    seasons = list(standalone_deltas.keys())
    x = np.arange(len(seasons))
    w = 0.38
    ax.bar(x - w/2, [standalone_deltas[s] for s in seasons], w,
           label="Standalone (vs is_home only)",
           color=COLOR_POS, alpha=0.85, edgecolor="white", lw=0.5)
    ax.bar(x + w/2, [pipeline_deltas[s] for s in seasons], w,
           label="In-pipeline (vs baseline 23 features)",
           color=COLOR_NEG, alpha=0.85, edgecolor="white", lw=0.5)

    ax.axhline(0, color="#333", lw=1.0, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_xlabel("Test season")
    ax.set_ylabel("ΔF1 from adding last_5_defr_proxy")
    ax.set_title("The proxy has real signal — but it's redundant with the existing pipeline\n"
                 "Large standalone gain becomes zero gain once strength/form features are present")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # value labels
    for i, s in enumerate(seasons):
        sv = standalone_deltas[s]
        ax.text(i - w/2, sv + 0.003, f"+{sv:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, 0.003, "0.000", ha="center", fontsize=9, color="#666")

    ax.set_ylim(-0.01, 0.10)

    plt.tight_layout()
    plt.savefig(OUT / "05_standalone.png")
    plt.close()


def main():
    print("Generating validation plots...")
    plot_bridge_comparison(); print("  [1/5] bridge comparison")
    plot_fbref_rankings();    print("  [2/5] fbref rankings")
    plot_correlation();       print("  [3/5] correlation")
    plot_walkforward();       print("  [4/5] walk-forward")
    plot_standalone();        print("  [5/5] standalone")
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
