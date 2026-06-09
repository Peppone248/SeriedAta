"""
team_trend/main.py — entry point for Squad Momentum Prediction.

Task:
    Given squad-level stats from the last N matchweeks, predict the number
    of points the team will accumulate over the next PREDICTION_HORIZON
    matchweeks.

    This is a regression problem at squad level, with temporal cross-validation.

Workflow:
  1. ETL pipeline (Bronze → Silver → Gold)
  2. Target engineering (next N matchweek points)
  3. Feature analysis
  4. Model training + evaluation
  5. Walk-forward backtesting
  6. Squad trajectory plots

Status: SKELETON — implement after ETL is validated end-to-end.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import (
    SERIE_A_SEASONS, NUM_FEATURES, TARGET_COL, PREDICTION_HORIZON,
)
from team_trend.pipeline import run_pipeline

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── HELPER ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# ─── SECTION 1 — ETL ─────────────────────────────────────────────────────────

def build_dataset(
    seasons:    list[str]    = SERIE_A_SEASONS,
    start_from: str          = "bronze",
    force:      bool         = False,
) -> pd.DataFrame:
    """
    Run the full ETL pipeline and return the Gold DataFrame.

    Args:
        seasons:    Seasons to include.
        start_from: ETL entry point ("bronze" | "silver" | "gold").
        force:      Recompute all layers even if files exist.
    """
    _section("1 — ETL PIPELINE")

    df = run_pipeline(
        seasons    = seasons,
        force      = force,
        start_from = start_from,
    )

    logger.info("Gold dataset: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


# ─── SECTION 2 — TARGET ENGINEERING ──────────────────────────────────────────

def build_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Create the prediction target: sum of points over the next `horizon` matches.

    Uses shift(-1) to look forward — target is future, not current.
    Rows near end of season where horizon is incomplete are dropped.

    Args:
        df:      Gold DataFrame with 'team', 'date', 'points' columns.
        horizon: Number of future matchweeks to sum.

    Returns:
        DataFrame with TARGET_COL added, NaN rows dropped.
    """
    df = df.copy().sort_values(["team", "date"])

    df[TARGET_COL] = (
        df.groupby("team")["points"]
        .transform(
            lambda x: x.shift(-1).rolling(horizon, min_periods=horizon).sum()
        )
    )

    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before - len(df)
    logger.info(
        "Target '%s': %d rows, %d dropped (incomplete horizon)",
        TARGET_COL, len(df), dropped
    )
    return df


# ─── SECTION 3 — FEATURE ANALYSIS ────────────────────────────────────────────

def run_feature_analysis(df: pd.DataFrame) -> None:
    """
    Leakage audit + correlation analysis on Gold features.
    TODO: implement after Gold dataset is validated.
    """
    _section("3 — FEATURE ANALYSIS")
    logger.info("Feature analysis: %d features", len(NUM_FEATURES))
    # TODO: call feature_selection.audit_leakage(df, NUM_FEATURES, target_col=TARGET_COL)
    # TODO: call feature_selection.run_feature_analysis(df, NUM_FEATURES)


# ─── SECTION 4 — MODEL TRAINING ───────────────────────────────────────────────

def run_models(df: pd.DataFrame):
    """
    Train regression models on the Gold feature set.
    TODO: implement after feature analysis is complete.

    Candidates:
        - Linear Regression (baseline)
        - XGBoost Regressor
        - LightGBM Regressor
    """
    _section("4 — MODEL TRAINING")
    logger.info("Model training: target=%s, features=%d", TARGET_COL, len(NUM_FEATURES))
    # TODO: implement run_regression, run_xgboost_regressor, run_lgbm_regressor


# ─── SECTION 5 — BACKTESTING ─────────────────────────────────────────────────

def run_backtesting_analysis(df: pd.DataFrame) -> None:
    """
    Walk-forward backtesting on regression models.
    TODO: adapt backtesting.py for regression task.
    """
    _section("5 — WALK-FORWARD BACKTESTING")
    # TODO: implement temporal cross-validation for regression


# ─── SECTION 6 — SQUAD TRAJECTORY PLOTS ──────────────────────────────────────

def plot_squad_trajectories(df: pd.DataFrame) -> None:
    """
    Visualise predicted vs actual points trajectory for each squad.
    TODO: implement after model training.
    """
    _section("6 — SQUAD TRAJECTORY VISUALISATION")
    # TODO: implement per-team trajectory plot with confidence interval


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. ETL ────────────────────────────────────────────────────────────
    gold_df = build_dataset(seasons=SERIE_A_SEASONS, start_from="bronze")

    # ── 2. target ─────────────────────────────────────────────────────────
    model_df = build_target(gold_df, horizon=PREDICTION_HORIZON)

    # ── 3. feature analysis ───────────────────────────────────────────────
    run_feature_analysis(model_df)

    # ── 4. models ─────────────────────────────────────────────────────────
    run_models(model_df)

    # ── 5. backtesting ────────────────────────────────────────────────────
    run_backtesting_analysis(model_df)

    # ── 6. visualisation ──────────────────────────────────────────────────
    plot_squad_trajectories(model_df)

    _section("DONE")
    print("  Pipeline completed.")


if __name__ == "__main__":
    main()
