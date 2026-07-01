"""Build team-season defensive profiles and cluster into tactical archetypes.

Aggregates all available defensive signals to one row per team per season,
then runs K-means clustering (K=4..6) to identify defensive archetypes.

Data sources (in order of preference, fall back gracefully if missing):
    1. matches_classification/data/raw/matches_seriea.csv   (always available)
    2. output/injection/fbref_with_defr_full.parquet         (rescue-bridge proxy)
       OR output/injection/fbref_with_defr.parquet           (reduced fallback)
    3. output/injection/fbref_defensive_data.parquet         (misc match stats)
    4. output/profiles/fbref_defense_season.parquet          (optional enrichment)

Outputs:
    output/profiles/team_season_profiles.parquet  — features per team-season
    output/profiles/cluster_assignments.parquet   — cluster label per team-season
    output/profiles/cluster_centroids.parquet     — feature means per cluster
    output/profiles/clustering_diagnostics.json   — silhouette, inertia, K choice
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEFR_DIR.parent
FBREF_CSV = REPO_ROOT / "matches_classification" / "data" / "raw" / "matches_seriea.csv"
PROFILES_DIR = DEFR_DIR / "output" / "profiles"
INJECTION_DIR = DEFR_DIR / "output" / "injection"

SEASONS_TO_USE = [2020, 2021, 2022, 2023, 2024]  # exclude partial 2025
MIN_MATCHES_PER_SEASON = 30  # require near-full season


# ─── Load data ─────────────────────────────────────────────────────────
def load_base() -> pd.DataFrame:
    df = pd.read_csv(FBREF_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["season"].isin(SEASONS_TO_USE)].copy()


def merge_defr(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Merge DefR proxy. Prefer full (rescue), fall back to reduced."""
    full = INJECTION_DIR / "fbref_with_defr_full.parquet"
    reduced = INJECTION_DIR / "fbref_with_defr.parquet"

    if full.exists():
        src = full
        cols = ["date", "team", "defr_proxy_full", "last_5_defr_proxy_full"]
        rename = {"defr_proxy_full": "defr_proxy", "last_5_defr_proxy_full": "last_5_defr_proxy"}
        label = "full bridge (rescue, R²=0.83)"
    elif reduced.exists():
        src = reduced
        cols = ["date", "team", "defr_proxy", "last_5_defr_proxy"]
        rename = {}
        label = "reduced bridge (R²=0.14)"
    else:
        print(f"  WARNING: no DefR proxy parquet found. Proceeding without it.")
        return df, "none"

    p = pd.read_parquet(src)
    p["date"] = pd.to_datetime(p["date"])
    if rename:
        p = p.rename(columns=rename)
    merged = df.merge(p[cols if not rename else ["date", "team", "defr_proxy", "last_5_defr_proxy"]],
                      on=["date", "team"], how="left")
    return merged, label


def merge_misc(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Merge misc match data (tackles, interceptions, fouls, crosses)."""
    misc_path = INJECTION_DIR / "fbref_defensive_data.parquet"
    if not misc_path.exists():
        return df, False

    misc = pd.read_parquet(misc_path)
    misc["date"] = pd.to_datetime(misc["date"])
    keep_cols = [c for c in misc.columns if c in (
        "date", "team", "tklw", "int", "fls", "fld", "crs", "crdy",
        "opp_tklw", "opp_int", "opp_fls", "opp_crs",
    )]
    merged = df.merge(misc[keep_cols], on=["date", "team"], how="left")
    return merged, True


# ─── Build team-season profiles ───────────────────────────────────────
def aggregate_to_season(df: pd.DataFrame, has_misc: bool) -> pd.DataFrame:
    """One row per (team, season) with all aggregated features."""
    # Filter to teams with enough matches in each season
    counts = df.groupby(["team", "season"]).size()
    valid = counts[counts >= MIN_MATCHES_PER_SEASON].index
    df = df[df.set_index(["team", "season"]).index.isin(valid)].copy()

    # Build aggregations
    agg_dict = {
        # Quality (kept for reference but EXCLUDED from clustering features)
        "ga": "mean",
        "gf": "mean",
        "xga": "mean",
        "xg": "mean",
        # Style (used in clustering)
        "poss": "mean",
        "sh": "mean",
        "sot": "mean",
        "dist": "mean",
    }
    grouped = df.groupby(["team", "season"]).agg(agg_dict).reset_index()

    # Rename for clarity
    grouped.columns = ["team", "season",
                       "avg_ga", "avg_gf", "avg_xga", "avg_xg",
                       "avg_poss", "avg_sh", "avg_sot", "avg_shot_dist"]

    # Opponent shot stats (from the mirror rows we already have via the merge)
    # df already has opp_dist from inject_defr_full step. If not, compute it.
    if "opp_sh" not in df.columns:
        # Pair opponents on the fly
        lookup = df.set_index(["date", "team"])[["sh", "sot", "dist"]]
        idx = pd.MultiIndex.from_arrays([df["date"], df["opponent"]],
                                         names=["date", "team"])
        opp = lookup.reindex(idx).reset_index(drop=True)
        df["opp_sh"] = opp["sh"].values
        df["opp_sot"] = opp["sot"].values
        df["opp_dist"] = opp["dist"].values

    opp_agg = df.groupby(["team", "season"]).agg(
        avg_sh_against=("opp_sh", "mean"),
        avg_sot_against=("opp_sot", "mean"),
        avg_opp_shot_dist=("opp_dist", "mean"),
    ).reset_index()
    grouped = grouped.merge(opp_agg, on=["team", "season"])

    # DefR proxy aggregation
    if "defr_proxy" in df.columns:
        defr_agg = df.groupby(["team", "season"])["defr_proxy"].mean().reset_index()
        defr_agg.columns = ["team", "season", "avg_defr_proxy"]
        grouped = grouped.merge(defr_agg, on=["team", "season"], how="left")

    # Misc match stats (if available)
    if has_misc:
        misc_cols = [c for c in df.columns if c in ("tklw", "int", "fls", "fld", "crs", "crdy",
                                                      "opp_tklw", "opp_int", "opp_fls", "opp_crs")]
        if misc_cols:
            misc_agg = df.groupby(["team", "season"])[misc_cols].mean().reset_index()
            misc_agg.columns = ["team", "season"] + [f"avg_{c}" for c in misc_cols]
            grouped = grouped.merge(misc_agg, on=["team", "season"], how="left")

    # Compute points and final position
    df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})
    pts = df.groupby(["team", "season"])["points"].sum().reset_index()
    pts.columns = ["team", "season", "season_points"]
    grouped = grouped.merge(pts, on=["team", "season"])

    # League position (rank by points)
    grouped["league_position"] = (
        grouped.groupby("season")["season_points"]
        .rank(method="dense", ascending=False).astype(int)
    )

    return grouped


# ─── Clustering ────────────────────────────────────────────────────────
def select_clustering_features(df: pd.DataFrame) -> list[str]:
    """Pick STYLE features for clustering (exclude pure quality/output)."""
    candidates = [
        "avg_defr_proxy",          # defensive engagement vs demand
        "avg_poss",                # possession share
        "avg_sh_against",          # defensive load
        "avg_opp_shot_dist",       # how deep do you let opponents shoot
        "avg_sh",                  # attacking volume
        "avg_shot_dist",           # do you shoot from far (counter) or close (sustained)
    ]
    # Add misc features if present
    misc_candidates = ["avg_tklw", "avg_int", "avg_fls", "avg_crs",
                       "avg_opp_tklw", "avg_opp_crs"]
    candidates += [c for c in misc_candidates if c in df.columns]

    # Filter to features actually present and well-populated
    valid = [c for c in candidates if c in df.columns and df[c].notna().mean() > 0.95]
    return valid


def run_clustering(df: pd.DataFrame, features: list[str], k_range=(4, 5, 6)) -> dict:
    """Fit K-means for several K, pick best by silhouette."""
    X = df[features].values
    X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0)[None, :])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clean)

    results = {"by_k": {}, "best_k": None, "best_silhouette": -1.0}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        results["by_k"][k] = {
            "silhouette": float(sil),
            "inertia": float(km.inertia_),
            "labels": labels.tolist(),
            "centroids_scaled": km.cluster_centers_.tolist(),
        }
        if sil > results["best_silhouette"]:
            results["best_silhouette"] = sil
            results["best_k"] = k
            results["best_labels"] = labels
            results["best_model"] = km
            results["best_centroids_scaled"] = km.cluster_centers_

    # Convert centroids back to original scale for interpretation
    centroids_orig = scaler.inverse_transform(results["best_centroids_scaled"])
    results["features"] = features
    results["scaler"] = scaler
    results["centroids_original"] = centroids_orig

    return results


def label_clusters(centroids: np.ndarray, features: list[str]) -> dict[int, str]:
    """Label clusters using percentile ranking among centroids.

    Assigns a tactical archetype to each cluster based on its relative
    position vs the other clusters on key dimensions. The labels
    follow the article's framework but adapted for team-level (not
    player-level) data.
    """
    n_clusters = len(centroids)
    df = pd.DataFrame(centroids, columns=features)

    # Compute percentile rank for each feature (0=low, 1=high)
    pct = df.rank(pct=True)

    labels = {}
    used = set()
    # Process clusters in a deterministic order: highest avg_defr first
    if "avg_defr_proxy" in features:
        order = df["avg_defr_proxy"].rank(ascending=False).astype(int) - 1
    else:
        order = list(range(n_clusters))

    # Build a profile signature per cluster
    for i in range(n_clusters):
        p_defr = pct.iloc[i]["avg_defr_proxy"] if "avg_defr_proxy" in features else 0.5
        p_poss = pct.iloc[i]["avg_poss"] if "avg_poss" in features else 0.5
        p_sh_against = pct.iloc[i]["avg_sh_against"] if "avg_sh_against" in features else 0.5
        p_opp_dist = pct.iloc[i]["avg_opp_shot_dist"] if "avg_opp_shot_dist" in features else 0.5

        # Decide label using relative positions
        if p_defr >= 0.75 and p_poss >= 0.6:
            name = "Dominant active defender"
        elif p_defr >= 0.6 and p_sh_against <= 0.4:
            name = "Aggressive press"
        elif p_defr <= 0.25 and p_sh_against >= 0.6:
            name = "Low block under pressure"
        elif p_defr <= 0.35 and p_poss >= 0.5:
            name = "Possession dominant"
        elif p_poss <= 0.35:
            name = "Reactive deep defender"
        elif p_opp_dist >= 0.6:
            name = "Compact mid-block"
        else:
            name = "Balanced / transitional"

        # Ensure uniqueness across clusters with duplicates getting suffixes
        base = name
        counter = 2
        while name in used:
            name = f"{base} ({counter})"
            counter += 1
        used.add(name)
        labels[i] = name

    return labels


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("TEAM-SEASON PROFILE BUILDER + TACTICAL CLUSTERING")
    print("=" * 64)

    print("\n[1] Loading FBref CSV...")
    df = load_base()
    print(f"    {len(df):,} team-match rows, seasons {SEASONS_TO_USE}")

    print("\n[2] Merging DefR proxy...")
    df, defr_label = merge_defr(df)
    print(f"    DefR source: {defr_label}")

    print("\n[3] Merging misc match stats...")
    df, has_misc = merge_misc(df)
    print(f"    Misc data: {'present ✓' if has_misc else 'missing (will use base features only)'}")

    print("\n[4] Aggregating to team-season level...")
    profiles = aggregate_to_season(df, has_misc)
    print(f"    {len(profiles)} team-seasons, {profiles['team'].nunique()} unique teams")
    print(f"    Features: {list(profiles.columns)}")

    print("\n[5] Selecting clustering features...")
    feats = select_clustering_features(profiles)
    print(f"    Using {len(feats)} style features:")
    for f in feats:
        print(f"      - {f}")

    print("\n[6] Running K-means clustering (K=4, 5, 6)...")
    cluster_results = run_clustering(profiles, feats)
    print(f"\n    Silhouette scores:")
    for k, info in cluster_results["by_k"].items():
        marker = "  ← best" if k == cluster_results["best_k"] else ""
        print(f"      K={k}: silhouette = {info['silhouette']:.4f}  "
              f"inertia = {info['inertia']:.1f}{marker}")
    print(f"\n    Best K = {cluster_results['best_k']}")

    # Apply best labels
    profiles["cluster_id"] = cluster_results["best_labels"]
    cluster_labels = label_clusters(cluster_results["centroids_original"], feats)
    profiles["cluster_label"] = profiles["cluster_id"].map(cluster_labels)

    print(f"\n[7] Cluster archetypes:")
    for cid in sorted(cluster_labels.keys()):
        name = cluster_labels[cid]
        members = profiles[profiles["cluster_id"] == cid]
        print(f"    [{cid}] {name}  (n={len(members)})")
        sample = members[["team", "season"]].head(5)
        for _, r in sample.iterrows():
            print(f"        {r['team']} {r['season']}")

    print("\n[8] Saving artefacts...")
    profiles.to_parquet(PROFILES_DIR / "team_season_profiles.parquet", index=False)

    centroids_df = pd.DataFrame(
        cluster_results["centroids_original"],
        columns=feats,
    )
    centroids_df["cluster_id"] = range(len(centroids_df))
    centroids_df["cluster_label"] = centroids_df["cluster_id"].map(cluster_labels)
    centroids_df.to_parquet(PROFILES_DIR / "cluster_centroids.parquet", index=False)

    diagnostics = {
        "best_k": int(cluster_results["best_k"]),
        "best_silhouette": float(cluster_results["best_silhouette"]),
        "by_k": {str(k): {"silhouette": v["silhouette"], "inertia": v["inertia"]}
                  for k, v in cluster_results["by_k"].items()},
        "features": feats,
        "cluster_labels": cluster_labels,
        "defr_source": defr_label,
        "misc_present": has_misc,
        "n_team_seasons": int(len(profiles)),
    }
    with open(PROFILES_DIR / "clustering_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"    Saved to {PROFILES_DIR}/")
    print(f"\n{'='*64}")
    print(f"Done. Next: python make_profile_report.py")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
