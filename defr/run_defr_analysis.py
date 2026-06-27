"""End-to-end DefR analysis pipeline.

Run with:
    python run_defr_analysis.py

Steps:
    1. Ensure Wyscout Serie A 2017/18 data is downloaded
    2. Parse events into a DataFrame
    3. Assign zones and classify events
    4. Compute baseline rates and team-match DefR scores
    5. Compute aggregate features
    6. Fit the bridge regression
    7. Save artefacts (parquet + JSON)
    8. Generate all plots
    9. Build the standalone HTML report

The pipeline is idempotent: re-running it will reuse downloaded data
and overwrite output artefacts. Total wall-clock time is dominated by
the initial download (~2 minutes on a fast connection); subsequent
runs complete in under 60 seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from defr_implementation import bridge, config, data, model, plots, report


def main():
    t0 = time.time()
    print("=" * 64)
    print("DefR ANALYSIS PIPELINE — Serie A 2017/18 (Wyscout)")
    print("=" * 64)

    # Ensure output directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: data ─────────────────────────────────────────────────
    print("\n[1/9] Ensure Wyscout data...")
    data.ensure_data()

    # ─── Step 2: parse ────────────────────────────────────────────────
    print("\n[2/9] Parse events...")
    events = data.parse_all_matches()

    # ─── Step 3: zones & classification ───────────────────────────────
    print("\n[3/9] Assign zones and classify events...")
    events = model.assign_zones(events)
    events = model.classify_events(events)
    events = model.add_opponent(events)
    n_atk = int(events["is_attacking"].sum())
    n_def = int(events["is_defensive"].sum())
    print(f"  Attacking events: {n_atk:,}  ({100 * n_atk / len(events):.1f}%)")
    print(f"  Defensive events: {n_def:,}  ({100 * n_def / len(events):.1f}%)")

    # ─── Step 4: zone baselines + team-match DefR ────────────────────
    print("\n[4/9] Compute zone baselines and team-match DefR...")
    zone_rates, attacking, defensive = model.compute_zone_baselines(events)
    defr_match = model.compute_team_defr(attacking, defensive, zone_rates, events)
    season = model.aggregate_season(defr_match)

    print(f"  DefR records: {len(defr_match)}")
    print(f"  DefR range: [{defr_match['defr_score'].min():.1f}, "
          f"{defr_match['defr_score'].max():.1f}]")
    print("\n  Top 5 season DefR:")
    for _, row in season.head(5).iterrows():
        print(f"    {row['rank']:2d}. {row['team_name']:18s}  {row['avg_defr']:+6.2f}")
    print("\n  Bottom 5 season DefR:")
    for _, row in season.tail(5).iterrows():
        print(f"    {row['rank']:2d}. {row['team_name']:18s}  {row['avg_defr']:+6.2f}")

    # ─── Step 5: aggregate features ──────────────────────────────────
    print("\n[5/9] Compute aggregate features...")
    agg = bridge.compute_aggregate_features(events)
    print(f"  Aggregate records: {len(agg)}")

    # ─── Step 6: bridge regression ───────────────────────────────────
    print("\n[6/9] Fit bridge regression...")
    bridge_results, model_obj, scaler_obj, merged = bridge.fit_bridge_regression(defr_match, agg)
    print(f"  CV R²:    {bridge_results['cv_r2_mean']:.4f} "
          f"± {bridge_results['cv_r2_std']:.4f}")
    print(f"  CV MAE:   {bridge_results['cv_mae_mean']:.4f} "
          f"± {bridge_results['cv_mae_std']:.4f}")
    print(f"  Full R²:  {bridge_results['full_r2']:.4f}")
    print(f"  Full MAE: {bridge_results['full_mae']:.4f}")

    print("\n  Feature importance (|coefficient| on standardised data):")
    for feat in bridge_results["feature_ranking"]:
        direction = "+" if feat["coef"] > 0 else "−"
        print(f"    {direction} {feat['name']:25s}  "
              f"coef={feat['coef']:+.4f}  |coef|={feat['abs_coef']:.4f}")

    # ─── Step 7: save artefacts ──────────────────────────────────────
    print("\n[7/9] Save data artefacts...")
    events.to_parquet(config.ARTIFACTS_DIR / "wyscout_events.parquet", index=False)
    defr_match.to_parquet(config.ARTIFACTS_DIR / "team_match_defr.parquet", index=False)
    season.to_parquet(config.ARTIFACTS_DIR / "team_season_defr.parquet", index=False)
    zone_rates.to_parquet(config.ARTIFACTS_DIR / "zone_rates.parquet", index=False)
    agg.to_parquet(config.ARTIFACTS_DIR / "aggregate_features.parquet", index=False)
    merged.to_parquet(config.ARTIFACTS_DIR / "bridge_dataset.parquet", index=False)
    with open(config.ARTIFACTS_DIR / "bridge_regression.json", "w") as f:
        json.dump(bridge_results, f, indent=2)
    print(f"  Wrote 7 files to {config.ARTIFACTS_DIR}")

    # ─── Step 8: plots ────────────────────────────────────────────────
    print("\n[8/9] Generate plots...")
    plot_paths = plots.make_all_plots(
        zone_rates, defr_match, season, agg, bridge_results, merged,
        config.PLOTS_DIR,
    )
    print(f"  Wrote {len(plot_paths)} plots to {config.PLOTS_DIR}")

    # ─── Step 9: HTML report ──────────────────────────────────────────
    print("\n[9/9] Build HTML report...")
    report.build_report(
        plot_paths, zone_rates, defr_match, season, bridge_results,
        n_events=len(events), n_matches=events["match_id"].nunique(),
        n_teams=events["team_name"].nunique(),
        out_path=config.REPORT_PATH,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"DONE in {elapsed:.1f} seconds")
    print(f"Report: {config.REPORT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
