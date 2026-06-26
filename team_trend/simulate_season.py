"""
simulate_season.py — end-of-season standings simulation.

Fits the quantile model on all seasons except the most recent, then
runs Monte Carlo simulation of the held-out season's remaining matches
to forecast the final table. Validates against actual outcomes.

Usage:
    python simulate_season.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from team_trend.models.xgboost_quantile import XGBoostQuantileRegressor
from team_trend.models.season_simulation import (
    simulate_end_of_season, print_standings_forecast, validate_simulation,
)
from team_trend.visualization.simulation_plots import run_all_simulation_plots
from team_trend.visualization.trajectory_simulation_overlay import (
    plot_season_overlay,
    plot_full_league_final_comparison,
)

# reuse the same feature definitions as training
from config import FEATURES_CLEAN as FEATURES, TARGET as TARGET_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLD_PATH = "data/gold/squad_momentum.parquet"
PLOT_DIR = "reports/plots/simulation"
N_SIMULATIONS = 10_000


def main() -> None:
    if not Path(GOLD_PATH).exists():
        raise FileNotFoundError(f"{GOLD_PATH} not found. Run pipeline.py first.")

    gold = pd.read_parquet(GOLD_PATH)
    available_features = [f for f in FEATURES if f in gold.columns]

    seasons = sorted(gold["season"].unique())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons (train + simulate).")
    train_seasons = seasons[:-1]
    test_season = seasons[-1]
    logger.info("Train seasons: %s   Simulate season: %s",
                train_seasons, test_season)

    # ── train the quantile model ──────────────────────────────────────────
    train_df = gold[gold["season"].isin(train_seasons)].dropna(
        subset=available_features + [TARGET_NAME]
    )
    X_tr = train_df[available_features].to_numpy(float)
    y_tr = train_df[TARGET_NAME].to_numpy(float)
    logger.info("Training quantile model on %d rows", len(X_tr))

    model = XGBoostQuantileRegressor().fit(
        X_tr, y_tr, feature_names=available_features,
    )

    # ── predict quantiles on test season ──────────────────────────────────
    test_df = gold[gold["season"] == test_season].dropna(
        subset=available_features + [TARGET_NAME]
    )
    X_te = test_df[available_features].to_numpy(float)
    qpreds = model.predict_quantiles(X_te)

    predictions = pd.DataFrame({
        "team": test_df["team"].to_numpy(),
        "matchweek": test_df["matchweek"].to_numpy(),
        "q_low": qpreds[0.1],
        "q_median": qpreds[0.5],
        "q_high": qpreds[0.9],
    })

    # ── simulate ──────────────────────────────────────────────────────────
    logger.info("Running %d Monte Carlo simulations...", N_SIMULATIONS)
    sim = simulate_end_of_season(
        gold=gold,
        quantile_predictions=predictions,
        season=test_season,
        target_col=TARGET_NAME,
        n_simulations=N_SIMULATIONS,
    )

    print_standings_forecast(sim)
    validate_simulation(sim)
    run_all_simulation_plots(sim, save_dir=PLOT_DIR)

    # step 4: trajectory + simulation overlay
    from visualization.trajectory_simulation_overlay import _setup_style
    _setup_style()
    plot_season_overlay(gold, sim, test_season, save_dir=PLOT_DIR)
    plot_full_league_final_comparison(sim, save_dir=PLOT_DIR)


if __name__ == "__main__":
    main()
