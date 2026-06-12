"""
train_model.py — train and evaluate the squad momentum regressor.

What it does:
  1. Load the Gold dataset from data/gold/squad_momentum.parquet
  2. Walk-forward backtest by season:
        fold k: train on seasons [s_1 .. s_k], test on season s_{k+1}
     With 5 seasons, this produces 4 folds. Each fold simulates real
     deployment: the model only ever sees the past.
  3. For each fold, train the from-scratch linear regression, predict
     on the held-out season, report MAE, RMSE, R^2 and a per-team summary.
  4. Compare against two baselines:
        - mean baseline:  predict y_train.mean() everywhere
        - persistence:    predict roll5_points * 5 (the team's recent form
                          extrapolated linearly)
  5. Aggregate stability across folds (mean and std of each metric).

Walk-forward backtesting is the right evaluation for time-series tasks:
random k-fold would leak future data into training and produce inflated R^2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from linear_regression import (
    LinearRegressionScratch,
    Standardizer,
    mae,
    rmse,
    r2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


GOLD_PATH = "data/gold/squad_momentum.parquet"
TARGET    = "next_5_matchweek_points"

# Pre-match features only. Anything from the current match (points,
# goals_for, possession on its own, etc.) would leak the target.
ROLL_FEATURES = [
    "roll5_points", "roll5_goal_diff",
    "roll5_goals_for", "roll5_goals_against",
    "roll5_possession",
    "roll5_shots", "roll5_shots_on_target", "roll5_shots_on_target_pct",
    "roll5_goals_per_shot",
    "roll5_save_pct", "roll5_saves",
    "roll5_players_used", "roll5_starters_used",
    "roll5_minutes_std", "roll5_squad_age_mean",
    "roll5_sum_tackles_won", "roll5_sum_interceptions",
    "roll5_sum_yellow_cards", "roll5_sum_fouls",
]
CONTEXT_FEATURES = [
    "is_home", "days_rest",
    "cum_points", "cum_goal_diff", "league_position",
    "season_progress",
]
FEATURES = ROLL_FEATURES + CONTEXT_FEATURES


# ─── walk-forward fold generator ───────────────────────────────────────────────

def season_folds(seasons_sorted: list[str]) -> list[tuple[list[str], str]]:
    """
    [(train_seasons, test_season), ...] with expanding train window.
    Needs at least 2 seasons; with 5 seasons -> 4 folds.
    """
    if len(seasons_sorted) < 2:
        raise ValueError("Need at least 2 seasons for walk-forward CV.")
    return [(seasons_sorted[:i], seasons_sorted[i])
            for i in range(1, len(seasons_sorted))]


# ─── one fold ──────────────────────────────────────────────────────────────────

def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    features: list[str],
    target:   str,
    l2:       float = 1.0,
) -> dict:
    """Fit on train, predict on test, return metrics and predictions."""
    X_tr, y_tr = train_df[features].to_numpy(float), train_df[target].to_numpy(float)
    X_te, y_te = test_df [features].to_numpy(float), test_df [target].to_numpy(float)

    scaler = Standardizer().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LinearRegressionScratch(solver="normal", l2=l2).fit(X_tr_s, y_tr)

    pred_tr = model.predict(X_tr_s)
    pred_te = model.predict(X_te_s)

    # baselines computed inside the fold so they respect the same temporal split
    baseline_mean = np.full_like(y_te, y_tr.mean())
    # persistence: extrapolate roll5_points to a 5-match horizon
    persistence = (test_df["roll5_points"].to_numpy(float) * 5).clip(0, 15)

    return {
        "n_train":          len(train_df),
        "n_test":           len(test_df),
        "train_mae":        mae (y_tr, pred_tr),
        "train_rmse":       rmse(y_tr, pred_tr),
        "train_r2":         r2  (y_tr, pred_tr),
        "test_mae":         mae (y_te, pred_te),
        "test_rmse":        rmse(y_te, pred_te),
        "test_r2":          r2  (y_te, pred_te),
        "baseline_mae":     mae (y_te, baseline_mean),
        "baseline_rmse":    rmse(y_te, baseline_mean),
        "persistence_mae":  mae (y_te, persistence),
        "persistence_rmse": rmse(y_te, persistence),
        "predictions":      pd.DataFrame({
            "team":      test_df["team"].values,
            "matchweek": test_df["matchweek"].values,
            "actual":    y_te,
            "predicted": pred_te,
            "residual":  y_te - pred_te,
        }),
        "model":            model,
        "scaler":           scaler,
    }


# ─── full walk-forward ─────────────────────────────────────────────────────────

def walk_forward_backtest(
    gold:     pd.DataFrame,
    features: list[str],
    target:   str,
    l2:       float = 1.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns (fold_summary_df, fold_results_list).
    fold_summary_df has one row per fold for quick comparison;
    fold_results_list keeps full per-fold output including predictions.
    """
    seasons = sorted(gold["season"].unique())
    folds   = season_folds(seasons)
    logger.info("Walk-forward CV over %d folds across seasons %s",
                len(folds), seasons)

    rows, results = [], []
    for train_seasons, test_season in folds:
        train_df = gold[gold["season"].isin(train_seasons)].dropna(subset=features + [target])
        test_df  = gold[gold["season"] == test_season].dropna(subset=features + [target])
        logger.info("Fold: train=%s (%d rows) -> test=%s (%d rows)",
                    train_seasons, len(train_df), test_season, len(test_df))

        res = train_and_evaluate_fold(train_df, test_df, features, target, l2=l2)
        res["test_season"]   = test_season
        res["train_seasons"] = train_seasons
        results.append(res)

        rows.append({k: v for k, v in res.items()
                     if k not in ("predictions", "model", "scaler")})

    summary = pd.DataFrame(rows)
    return summary, results


# ─── report ────────────────────────────────────────────────────────────────────

def print_report(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"{'WALK-FORWARD BACKTEST — SQUAD MOMENTUM':^78}")
    print("=" * 78)

    show = summary[[
        "test_season", "n_train", "n_test",
        "train_mae", "test_mae", "test_rmse", "test_r2",
        "baseline_mae", "persistence_mae",
    ]].copy()
    for c in show.columns[3:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n" + "-" * 78)
    print("STABILITY (mean ± std across folds):")
    print("-" * 78)
    for m in ["test_mae", "test_rmse", "test_r2"]:
        print(f"  {m:14s} {summary[m].mean():.3f}  ±  {summary[m].std():.3f}")

    print("\n" + "-" * 78)
    print("VS BASELINES (averaged across folds):")
    print("-" * 78)
    print(f"  model MAE        : {summary['test_mae'].mean():.3f}")
    print(f"  mean-baseline    : {summary['baseline_mae'].mean():.3f}   "
          f"gain {summary['baseline_mae'].mean() - summary['test_mae'].mean():+.3f}")
    print(f"  persistence base : {summary['persistence_mae'].mean():.3f}   "
          f"gain {summary['persistence_mae'].mean() - summary['test_mae'].mean():+.3f}")
    print("=" * 78)


def per_team_diagnostics(results: list[dict], top_n: int = 5) -> None:
    """For the last fold, show worst-predicted teams (largest residuals)."""
    last = results[-1]
    preds = last["predictions"].copy()
    preds["abs_residual"] = preds["residual"].abs()

    worst = (
        preds.groupby("team")["abs_residual"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    best = (
        preds.groupby("team")["abs_residual"]
        .mean()
        .sort_values()
        .head(top_n)
    )

    print(f"\nFold test season: {last['test_season']}")
    print(f"  Hardest to predict (highest mean |residual|):")
    print(worst.round(2).to_string())
    print(f"  Easiest to predict:")
    print(best.round(2).to_string())


def top_coefficients(results: list[dict], top_n: int = 12) -> None:
    """Show the strongest coefficients from the final (most data) model."""
    last = results[-1]
    coefs = last["model"].coefficients(FEATURES)
    print(f"\nTop {top_n} coefficients (model trained on {len(last['train_seasons'])} seasons, standardised scale):")
    for name, w in coefs[:top_n]:
        print(f"  {name:32s} {w:+.3f}")


# ─── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(GOLD_PATH).exists():
        raise FileNotFoundError(
            f"{GOLD_PATH} not found. Run pipeline.py first."
        )

    gold = pd.read_parquet(GOLD_PATH)
    logger.info("Loaded Gold: %s, seasons=%s",
                gold.shape, sorted(gold['season'].unique()))

    summary, results = walk_forward_backtest(
        gold, features=FEATURES, target=TARGET, l2=1.0,
    )

    print_report(summary)
    per_team_diagnostics(results)
    top_coefficients(results)


if __name__ == "__main__":
    main()