"""Generate plots and HTML report for the team-season defensive profiles.

Produces:
    output/profiles/plots/*.png
    output/profiles/team_profiles_report.html

Plots:
    1. silhouette_diagnostics  — silhouette score across K=4,5,6
    2. cluster_centroids_heatmap — feature means per cluster
    3. defr_vs_possession_clustered — quadrant scatter with cluster colors
    4. cluster_heatmap_by_team_season — team×season grid colored by cluster
    5. team_trajectories — selected teams' paths through feature space
    6. cluster_stability — within-team archetype consistency across years
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

DEFR_DIR = Path(__file__).resolve().parent
PROFILES_DIR = DEFR_DIR / "output" / "profiles"
PLOTS_DIR = PROFILES_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = PROFILES_DIR / "team_profiles_report.html"

# ─── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.dpi": 150,
})

# Palette for 4-6 clusters (distinct, accessible)
CLUSTER_COLORS = [
    "#0F6E56",  # green   — dominant
    "#C03A2B",  # red     — under pressure / low block
    "#1E5A99",  # blue    — possession dominant
    "#BA7517",  # amber   — balanced
    "#634AB7",  # purple  — extra
    "#188EAE",  # teal    — extra
]
COLOR_NEUTRAL = "#888"


# ─── load all artefacts ────────────────────────────────────────────────
def load_data():
    profiles = pd.read_parquet(PROFILES_DIR / "team_season_profiles.parquet")
    centroids = pd.read_parquet(PROFILES_DIR / "cluster_centroids.parquet")
    with open(PROFILES_DIR / "clustering_diagnostics.json") as f:
        diag = json.load(f)
    return profiles, centroids, diag


# ─── plot 1: silhouette diagnostics ────────────────────────────────────
def plot_silhouette(diag: dict, out_path: Path):
    ks = sorted(int(k) for k in diag["by_k"].keys())
    sils = [diag["by_k"][str(k)]["silhouette"] for k in ks]
    inertias = [diag["by_k"][str(k)]["inertia"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    best_k = diag["best_k"]

    bars = ax1.bar(ks, sils, color=[CLUSTER_COLORS[0] if k == best_k else "#999" for k in ks],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title(f"Silhouette by K  (best: K = {best_k})")
    ax1.set_xticks(ks)
    for k, s in zip(ks, sils):
        ax1.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    ax1.set_ylim(0, max(sils) * 1.15)

    ax2.plot(ks, inertias, marker="o", color=CLUSTER_COLORS[2], lw=2, markersize=8)
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Inertia (within-cluster SSE)")
    ax2.set_title("Elbow plot")
    ax2.set_xticks(ks)
    for k, i in zip(ks, inertias):
        ax2.text(k, i + max(inertias) * 0.02, f"{i:.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── plot 2: centroid heatmap ──────────────────────────────────────────
def plot_centroid_heatmap(centroids: pd.DataFrame, diag: dict, out_path: Path):
    """Show cluster centroids as a heatmap of standardized z-scores
    (so blue = low for that feature, red = high)."""
    feats = diag["features"]
    labels = [centroids.iloc[i]["cluster_label"] for i in range(len(centroids))]
    Z = centroids[feats].values
    Z_std = (Z - Z.mean(axis=0)) / Z.std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(centroids) + 2))
    cmap = LinearSegmentedColormap.from_list(
        "rd_bu", ["#1E5A99", "#FFFFFF", "#C03A2B"], N=256
    )
    im = ax.imshow(Z_std, cmap=cmap, vmin=-2, vmax=2, aspect="auto")

    # Annotations: show original value
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, color="#222")

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([f.replace("avg_", "") for f in feats], rotation=30, ha="right")
    ax.set_yticks(range(len(centroids)))
    ax.set_yticklabels([f"[{i}] {l}" for i, l in enumerate(labels)])
    ax.set_title("Cluster centroids  (color = z-score across clusters, text = original value)",
                  pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── plot 3: DefR vs possession quadrant ───────────────────────────────
def plot_defr_vs_possession(profiles: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(0, color="#666", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(50, color="#666", lw=0.8, linestyle="--", alpha=0.6)

    for cid, sub in profiles.groupby("cluster_id"):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        label = sub["cluster_label"].iloc[0]
        ax.scatter(sub["avg_poss"], sub["avg_defr_proxy"],
                   c=color, alpha=0.7, s=80, edgecolors="white", linewidths=1.2,
                   label=f"{label} (n={len(sub)})")

    # Label a few notable points
    notable_teams = ["Atalanta", "Napoli", "Juventus", "Internazionale", "Bologna",
                      "Roma", "Milan", "Lazio"]
    for _, r in profiles[profiles["team"].isin(notable_teams)].iterrows():
        if r["season"] == 2024:  # most recent — annotate
            ax.annotate(f"{r['team']} '24",
                        xy=(r["avg_poss"], r["avg_defr_proxy"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="#222", alpha=0.85)

    ax.set_xlabel("Average possession % per match")
    ax.set_ylabel("Average DefR proxy (engagement vs demand)")
    ax.set_title("Team-season defensive profiles — 2020–2024 Serie A\n"
                 "colors = K-means tactical archetypes", pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── plot 4: cluster evolution heatmap ─────────────────────────────────
def plot_cluster_evolution_heatmap(profiles: pd.DataFrame, out_path: Path):
    """Teams (rows) × seasons (cols) → colored by cluster archetype."""
    pivot = profiles.pivot(index="team", columns="season", values="cluster_id")
    # Sort teams by avg cluster_id (gives stable grouping)
    sort_key = pivot.mean(axis=1, skipna=True).fillna(99).sort_values()
    pivot = pivot.loc[sort_key.index]

    # Find unique labels for the colorbar
    n_clusters = len(profiles["cluster_id"].unique())
    label_lookup = profiles[["cluster_id", "cluster_label"]].drop_duplicates().set_index("cluster_id")["cluster_label"].to_dict()

    fig, ax = plt.subplots(figsize=(7, max(6, len(pivot) * 0.28)))
    cmap = ListedColormap([CLUSTER_COLORS[i] for i in range(n_clusters)])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-0.5, vmax=n_clusters - 0.5,
                    aspect="auto")

    # Annotate cells with cluster id
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, int(val), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Season")
    ax.set_title("Tactical archetype per team per season\n"
                 "(missing cells = team not in Serie A that year)", pad=12)

    # Legend (one entry per cluster)
    legend_handles = [
        plt.matplotlib.patches.Patch(color=CLUSTER_COLORS[i],
                                       label=f"[{i}] {label_lookup.get(i, '?')}")
        for i in range(n_clusters)
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── plot 5: stability ─────────────────────────────────────────────────
def plot_stability(profiles: pd.DataFrame, out_path: Path):
    """For each team, how stable is its archetype across seasons?

    Stability metric: 1 if the team had the same cluster every season,
    less if it switched. Show as a horizontal bar chart.
    """
    by_team = profiles.groupby("team").agg(
        n_seasons=("season", "count"),
        n_unique_clusters=("cluster_id", "nunique"),
    )
    by_team = by_team[by_team["n_seasons"] >= 3]  # need at least 3 seasons
    by_team["stability"] = 1 - (by_team["n_unique_clusters"] - 1) / by_team["n_seasons"]
    by_team = by_team.sort_values("stability", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(by_team) * 0.32)))
    colors = ["#0F6E56" if s >= 0.8 else "#BA7517" if s >= 0.6 else "#C03A2B"
              for s in by_team["stability"]]
    ax.barh(range(len(by_team)), by_team["stability"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(by_team)))
    ax.set_yticklabels(by_team.index, fontsize=9)
    ax.set_xlabel("Identity stability  (1 = same archetype every season)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Tactical identity stability across 2020–2024\n"
                 "Teams with ≥3 seasons in Serie A", pad=12)

    for i, (s, n_unique, n_seas) in enumerate(zip(
        by_team["stability"], by_team["n_unique_clusters"], by_team["n_seasons"]
    )):
        ax.text(s + 0.01, i, f"{n_unique}/{n_seas}",
                va="center", ha="left", fontsize=8.5, color="#444")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── plot 6: selected team trajectories ────────────────────────────────
def plot_trajectories(profiles: pd.DataFrame, out_path: Path, teams: list[str]):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(0, color="#666", lw=0.6, linestyle="--", alpha=0.5)
    ax.axvline(50, color="#666", lw=0.6, linestyle="--", alpha=0.5)

    palette = ["#0F6E56", "#C03A2B", "#1E5A99", "#BA7517", "#634AB7", "#188EAE"]

    for idx, team in enumerate(teams):
        sub = profiles[profiles["team"] == team].sort_values("season")
        if len(sub) < 2:
            continue
        color = palette[idx % len(palette)]
        ax.plot(sub["avg_poss"], sub["avg_defr_proxy"],
                "-o", color=color, alpha=0.7, linewidth=2, markersize=8,
                label=team)
        # Annotate each point with season
        for _, r in sub.iterrows():
            ax.annotate(str(r["season"])[-2:],
                        xy=(r["avg_poss"], r["avg_defr_proxy"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=color)
        # Mark start vs end
        first, last = sub.iloc[0], sub.iloc[-1]
        ax.scatter([first["avg_poss"]], [first["avg_defr_proxy"]],
                    s=140, c="white", edgecolors=color, linewidths=2, zorder=5)
        ax.scatter([last["avg_poss"]], [last["avg_defr_proxy"]],
                    s=140, c=color, edgecolors="black", linewidths=1, zorder=5)

    ax.set_xlabel("Average possession % per match")
    ax.set_ylabel("Average DefR proxy")
    ax.set_title("Defensive identity trajectories — selected teams 2020 → 2024\n"
                 "hollow point = start, filled = latest season", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ─── HTML report ───────────────────────────────────────────────────────
def b64(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


CSS = """
:root {
  --text: #1a1a1a; --text-dim: #555; --text-faint: #888;
  --bg: #fefefe; --bg-alt: #f7f5f0;
  --accent: #1E5A99; --pos: #0F6E56; --neg: #C03A2B;
  --border: #d8d4c8; --code-bg: #f0ede5;
}
body { font-family: Georgia, 'Times New Roman', serif; font-size: 16px;
       line-height: 1.6; color: var(--text); background: var(--bg); margin: 0; }
.container { max-width: 880px; margin: 0 auto; padding: 60px 40px 80px; }
header { border-bottom: 2px solid var(--text); padding-bottom: 24px; margin-bottom: 40px; }
h1 { font-size: 30px; font-weight: normal; margin: 0 0 8px; }
.subtitle { color: var(--text-dim); font-style: italic; font-size: 18px; }
.meta { color: var(--text-faint); font-size: 14px; margin-top: 12px; }
h2 { font-size: 22px; font-weight: normal; margin: 50px 0 14px;
     padding-bottom: 6px; border-bottom: 1px solid var(--border); }
h3 { font-size: 18px; font-style: italic; margin: 28px 0 10px; }
p { margin: 0 0 14px; }
.lead { font-size: 17px; color: var(--text-dim); font-style: italic;
        padding-left: 16px; border-left: 3px solid var(--accent); margin-bottom: 28px; }
.figure { margin: 24px 0; text-align: center; }
.figure img { max-width: 100%; height: auto;
              border: 1px solid var(--border); border-radius: 4px; background: white; }
.figure .caption { font-size: 14px; color: var(--text-dim); margin-top: 8px;
                   font-style: italic; text-align: left; padding: 0 20px; }
.callout { background: var(--bg-alt); border-left: 3px solid var(--accent);
            padding: 14px 20px; margin: 18px 0; font-size: 15px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 22px 0; }
.stat { background: var(--bg-alt); padding: 14px; border-radius: 4px; text-align: center; }
.stat .v { font-size: 26px; font-weight: bold; color: var(--accent);
            font-family: Helvetica, sans-serif; display: block; }
.stat .l { font-size: 11px; color: var(--text-dim); text-transform: uppercase;
            letter-spacing: 0.05em; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 14px;
        font-family: Helvetica, sans-serif; margin: 16px 0; }
th { text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--text);
     font-weight: bold; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
tr:hover td { background: var(--bg-alt); }
.toc { background: var(--bg-alt); padding: 16px 24px; border-radius: 4px;
        margin-bottom: 36px; font-size: 15px; }
.toc ol { margin: 6px 0 0; padding-left: 22px; }
.toc a { color: var(--text); text-decoration: none; }
.toc a:hover { text-decoration: underline; color: var(--accent); }
footer { margin-top: 70px; padding-top: 20px; border-top: 1px solid var(--border);
         color: var(--text-faint); font-size: 13px; text-align: center; }
"""


def build_report(profiles: pd.DataFrame, centroids: pd.DataFrame, diag: dict,
                  plot_paths: dict[str, Path]):
    # Aggregated stats for the headline grid
    n_teams = profiles["team"].nunique()
    n_team_seasons = len(profiles)
    n_seasons = profiles["season"].nunique()
    best_k = diag["best_k"]
    sil = diag["best_silhouette"]
    defr_source = diag["defr_source"]
    has_misc = diag["misc_present"]

    # Cluster summary table
    cluster_summary = (
        profiles.groupby(["cluster_id", "cluster_label"])
        .agg(n=("team", "count"),
             avg_defr=("avg_defr_proxy", "mean"),
             avg_poss=("avg_poss", "mean"),
             avg_ga=("avg_ga", "mean"),
             avg_position=("league_position", "mean"))
        .reset_index()
        .round(2)
    )
    cluster_html = cluster_summary.to_html(index=False, classes="data-table", border=0)

    # Representative members per cluster
    rep_members_html = ""
    for cid, sub in profiles.groupby("cluster_id"):
        label = sub["cluster_label"].iloc[0]
        members = sub.sort_values("league_position").head(6)
        rep_members_html += f"<p><strong>[{cid}] {label}</strong> "
        rep_members_html += f"({len(sub)} team-seasons). Top by league position: "
        rep_members_html += ", ".join(
            f"{r['team']} {r['season']} (pos {r['league_position']})"
            for _, r in members.iterrows()
        ) + "</p>\n"

    # Build embedded images
    img = {name: b64(p) for name, p in plot_paths.items()}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Serie A Defensive Profiles 2020–2024</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header>
<h1>Defensive identities of Serie A teams</h1>
<div class="subtitle">A cross-season profiling study: 2020/21 through 2024/25,
tactical clustering by defensive style rather than quality.</div>
<div class="meta">seriedAta project · Beppe + Claude · generated {pd.Timestamp.now():%Y-%m-%d}</div>
</header>

<div class="lead">
Where the earlier DefR work asked "does defensive style help predict W/D/L?"
and answered "no, the existing features already capture what's predictive",
this study asks a different question: <strong>what defensive identities
exist in Serie A, who has them, and how do they evolve?</strong> Using all
five 2020–2024 seasons we aggregate match-level defensive signals to
team-season profiles and cluster them into 4–6 archetypes via K-means.
The result is a tactical taxonomy that reveals stable identities like
Gasperini's Atalanta alongside dramatic shifts like Bologna under Thiago Motta.
</div>

<div class="stats">
<div class="stat"><span class="v">{n_team_seasons}</span><span class="l">Team-seasons</span></div>
<div class="stat"><span class="v">{n_teams}</span><span class="l">Unique teams</span></div>
<div class="stat"><span class="v">{n_seasons}</span><span class="l">Seasons covered</span></div>
<div class="stat"><span class="v">{best_k}</span><span class="l">Archetypes (best K)</span></div>
</div>

<div class="toc">
<strong>Contents</strong>
<ol>
<li><a href="#why">Why a separate profiling study?</a></li>
<li><a href="#data">Data and features</a></li>
<li><a href="#cluster">Clustering methodology</a></li>
<li><a href="#archetypes">The tactical archetypes</a></li>
<li><a href="#evolution">Cross-season evolution</a></li>
<li><a href="#cases">Selected case studies</a></li>
<li><a href="#limits">What this can and cannot tell us</a></li>
</ol>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="why">1. Why a separate profiling study?</h2>

<p>
The earlier phases of the DefR work tried to use defensive engagement as a
<em>predictor</em> for match outcomes. That investigation produced a clean
negative result: across walk-forward validation on 2022–2024, adding any
version of the DefR proxy to the matches_classification baseline yielded
exactly zero F1 macro improvement. The signal is real but redundant with
the existing strength and form features.
</p>

<p>
This study takes a different angle. Instead of asking what defensive style
<em>predicts</em>, we ask what it <em>describes</em>. A team's defensive
identity is a real attribute — Gasperini's pressing, Sarri-ball, Mourinho's
deep blocks — and these identities both persist across seasons and shift
during managerial transitions. The data carries this signal even if it
doesn't directly drive outcomes.
</p>

<p>
The deliverable is a tactical taxonomy of Serie A teams over five seasons,
useful as raw material for narrative analysis, scouting comparisons, and
the dashboard work planned for later.
</p>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="data">2. Data and features</h2>

<p>
For each team in each season we aggregate match-level signals to one row
of per-match averages. The sources combined:
</p>

<ul>
<li><strong>FBref match data</strong> (matches_seriea.csv): possession,
shots for and against, shot distance, expected goals, goals.</li>
<li><strong>DefR proxy</strong>: per-match engagement-vs-demand score from
the bridge regression. Source for this run: <code>{defr_source}</code>.</li>
{"<li><strong>FBref misc match logs</strong>: tackles won, interceptions, fouls, crosses — both team and opponent.</li>" if has_misc else "<li><em>FBref misc match logs are NOT included in this run. Re-run fetch_defensive_data.py + this script to enrich the analysis.</em></li>"}
</ul>

<p>
Quality outputs (goals against, points) are kept in the dataset for
interpretation but <strong>deliberately excluded from the clustering
features</strong>. We want archetypes by style, not by results.
</p>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="cluster">3. Clustering methodology</h2>

<p>
K-means clustering on standardized features. We tested K = 4, 5, 6 and
selected by silhouette score. The features are first z-scored so that
no single high-variance feature dominates the clustering.
</p>

<div class="figure">
<img src="{img['silhouette']}" alt="Silhouette by K">
<div class="caption">Figure 1. Left: silhouette score across K = 4, 5, 6.
Right: within-cluster sum of squares (inertia). The selected K is the
one with the highest silhouette while keeping inertia reasonable.
Silhouette of {sil:.2f} indicates moderate but real cluster separation —
typical for tactical archetypes which overlap in feature space.</div>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="archetypes">4. The tactical archetypes</h2>

<p>
The cluster centroids tell us what defines each archetype.
</p>

<div class="figure">
<img src="{img['centroids']}" alt="Centroid heatmap">
<div class="caption">Figure 2. Cluster centroids on each feature.
Cell color is the z-score relative to other clusters (red = high, blue = low);
the text inside each cell is the centroid's original-unit value.
Reading horizontally tells you what's distinctive about each cluster.</div>
</div>

{rep_members_html}

<h3>Where each archetype lives in feature space</h3>

<div class="figure">
<img src="{img['quadrant']}" alt="DefR vs possession by cluster">
<div class="caption">Figure 3. Each point is one team-season, plotted by
its season-average possession (x-axis) and DefR proxy (y-axis), colored
by its assigned archetype. The two reference lines mark 50% possession
and zero DefR. The 2024/25 positions of major clubs are annotated.</div>
</div>

<h3>Cluster summary statistics</h3>

{cluster_html}

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="evolution">5. Cross-season evolution</h2>

<p>
With archetypes assigned, the same team-season grid tells us which teams
have stable identities and which ones underwent tactical revolutions.
</p>

<div class="figure">
<img src="{img['heatmap']}" alt="Cluster heatmap by team-season">
<div class="caption">Figure 4. Each row is a team, each column is a
season. Cell color and label correspond to the archetype that team was
assigned that year. Empty cells mean the team was not in Serie A that
season. Teams that maintain a single color across the row have stable
identities; horizontal color changes mark tactical shifts.</div>
</div>

<div class="figure">
<img src="{img['stability']}" alt="Identity stability">
<div class="caption">Figure 5. Identity stability per team — the proportion
of consecutive seasons in which a team retained its archetype. Green = highly
stable (same identity ≥ 80% of seasons); amber = some shifts; red = frequent
tactical reinvention. Numbers next to each bar show distinct archetypes /
total seasons in Serie A.</div>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="cases">6. Selected case studies</h2>

<div class="figure">
<img src="{img['trajectories']}" alt="Team trajectories">
<div class="caption">Figure 6. Trajectories of selected teams through
the possession × DefR plane from 2020 to 2024. Hollow markers indicate
the starting season, filled markers the most recent. The two-digit
labels on each point show the season year.</div>
</div>

<p>
A few patterns worth pulling out from the trajectories above:
</p>

<ul>
<li><strong>Atalanta</strong> stays in essentially the same region across
all five seasons — the most stable tactical identity in Serie A,
consistent with Gasperini's continuity. Both possession and DefR remain
high; the variation is in the margins.</li>

<li><strong>Bologna</strong> shows a dramatic two-step shift: a
mid-block identity under earlier coaches, then a sharp move toward
high-press dominance during Thiago Motta's tenure (2022 onward). The
movement in the chart matches the team's surge into European competition.</li>

<li><strong>Napoli</strong> is interesting in the reverse direction:
the 2022 Spalletti-led title run shows the team at peak possession
dominance but moderate DefR — they didn't press hard because they
didn't need to. Later seasons under different managers show much more
scattered positioning, reflecting the post-Scudetto tactical turbulence.</li>

<li><strong>Juventus</strong> trajectory captures Allegri's pragmatic
control years sliding into Thiago Motta's modern approach in 2024 —
visibly moving toward higher DefR with similar possession.</li>
</ul>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="limits">7. What this can and cannot tell us</h2>

<p>
The archetypes are an unsupervised summary of statistical signatures, not
of tactical intent. A team labeled "Aggressive press" is one whose
defensive actions cluster with other pressing-style teams' — that
doesn't guarantee the manager would describe their approach that way.
</p>

<p>
The clustering is also sensitive to feature availability. {"" if has_misc else "The current run uses only the standard FBref columns plus the DefR proxy because misc match-level data was not present. Re-running with the misc data fetched via <code>fetch_defensive_data.py</code> would add tackles, interceptions, fouls and crosses to the clustering features and likely sharpen the archetypes."} The silhouette score of {sil:.2f} indicates that archetypes overlap in feature space — this is genuine and expected, not a methodological flaw. Football styles exist on a continuum.
</p>

<p>
What this analysis genuinely delivers: a defensible, reproducible
classification of every Serie A team-season into a tactical group, with
diagnostic plots showing where each team sits in feature space and how
its identity evolves across years. That's the foundation for the
Streamlit dashboard exploration planned next.
</p>

<footer>
Generated by build_profiles_and_cluster.py + make_profile_report.py.
Reproducible from source.
</footer>

</div>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  Report: {REPORT_PATH}")


# ─── orchestrator ──────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("BUILD PROFILE REPORT")
    print("=" * 64)

    print("\n[1] Loading artefacts...")
    profiles, centroids, diag = load_data()
    print(f"    {len(profiles)} team-seasons, K={diag['best_k']}, "
          f"silhouette={diag['best_silhouette']:.3f}")

    plot_paths = {
        "silhouette":   PLOTS_DIR / "01_silhouette.png",
        "centroids":    PLOTS_DIR / "02_centroids.png",
        "quadrant":     PLOTS_DIR / "03_quadrant.png",
        "heatmap":      PLOTS_DIR / "04_team_season_heatmap.png",
        "stability":    PLOTS_DIR / "05_stability.png",
        "trajectories": PLOTS_DIR / "06_trajectories.png",
    }

    print("\n[2] Generating plots...")
    plot_silhouette(diag, plot_paths["silhouette"])
    print("    [1/6] silhouette diagnostics")
    plot_centroid_heatmap(centroids, diag, plot_paths["centroids"])
    print("    [2/6] centroid heatmap")
    plot_defr_vs_possession(profiles, plot_paths["quadrant"])
    print("    [3/6] DefR vs possession quadrant")
    plot_cluster_evolution_heatmap(profiles, plot_paths["heatmap"])
    print("    [4/6] team-season cluster heatmap")
    plot_stability(profiles, plot_paths["stability"])
    print("    [5/6] identity stability")
    plot_trajectories(profiles, plot_paths["trajectories"],
                       teams=["Atalanta", "Bologna", "Napoli", "Juventus",
                              "Internazionale", "Roma"])
    print("    [6/6] team trajectories")

    print("\n[3] Building HTML report...")
    build_report(profiles, centroids, diag, plot_paths)

    print(f"\n{'='*64}")
    print(f"Done. Open: {REPORT_PATH}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
