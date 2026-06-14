"""
train_model.py — train and evaluate squad-momentum regressors.

Now supports both:
  - LinearRegressionScratch (numpy, ridge, our from-scratch baseline)
  - XGBoostRegressorWrapper  (xgboost, non-linear, library-backed)

Walk-forward CV by season. Models are trained on identical features so the
comparison isolates the effect of MODEL FAMILY, not feature engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.linear_regression import (
    LinearRegressionScratch, Standardizer, mae, rmse, r2,
)
from models.xgboost_regressor import XGBoostRegressorWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLD_PATH = "data/gold/squad_momentum.parquet"
TARGET = "next_5_matchweek_points"

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
    # volatility (new)
    "roll5_points_std", "roll5_goal_diff_std",
]
CONTEXT_FEATURES = [
    "is_home", "days_rest",
    "cum_points", "cum_goal_diff", "league_position",
    "season_progress",
]
FEATURES = ROLL_FEATURES + CONTEXT_FEATURES


# ─── walk-forward fold generator ───────────────────────────────────────────────

def season_folds(seasons_sorted: list[str]) -> list[tuple[list[str], str]]:
    if len(seasons_sorted) < 2:
        raise ValueError("Need at least 2 seasons for walk-forward CV.")
    return [(seasons_sorted[:i], seasons_sorted[i])
            for i in range(1, len(seasons_sorted))]


# ─── one fold ──────────────────────────────────────────────────────────────────

def train_and_evaluate_fold(
        model_factory,
        needs_scaling: bool,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        target: str,
) -> dict:
    """
    Fit one model on train, predict on test, return metrics.

    Args:
        model_factory:  callable returning a fresh, unfitted model.
        needs_scaling:  True for linear, False for trees.
    """
    X_tr, y_tr = train_df[features].to_numpy(float), train_df[target].to_numpy(float)
    X_te, y_te = test_df[features].to_numpy(float), test_df[target].to_numpy(float)

    if needs_scaling:
        scaler = Standardizer().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    model = model_factory()
    if hasattr(model, "fit") and "feature_names" in model.fit.__code__.co_varnames:
        model.fit(X_tr, y_tr, feature_names=features)
    else:
        model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    baseline_mean = np.full_like(y_te, y_tr.mean())
    persistence = (test_df["roll5_points"].to_numpy(float) * 5).clip(0, 15)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_mae": mae(y_tr, pred_tr),
        "test_mae": mae(y_te, pred_te),
        "test_rmse": rmse(y_te, pred_te),
        "test_r2": r2(y_te, pred_te),
        "baseline_mae": mae(y_te, baseline_mean),
        "persistence_mae": mae(y_te, persistence),
        "predictions": pd.DataFrame({
            "team": test_df["team"].values,
            "matchweek": test_df["matchweek"].values,
            "actual": y_te,
            "predicted": pred_te,
            "residual": y_te - pred_te,
        }),
        "model": model,
    }


# ─── full walk-forward ─────────────────────────────────────────────────────────

def walk_forward_backtest(
        gold: pd.DataFrame,
        model_factory,
        needs_scaling: bool,
        features: list[str],
        target: str,
        label: str,
) -> tuple[pd.DataFrame, list[dict]]:
    seasons = sorted(gold["season"].unique())
    folds = season_folds(seasons)
    logger.info("[%s] walk-forward CV over %d folds across seasons %s",
                label, len(folds), seasons)

    rows, results = [], []
    for train_seasons, test_season in folds:
        train_df = gold[gold["season"].isin(train_seasons)].dropna(subset=features + [target])
        test_df = gold[gold["season"] == test_season].dropna(subset=features + [target])
        logger.info("[%s] fold: train=%s (%d) -> test=%s (%d)",
                    label, train_seasons, len(train_df), test_season, len(test_df))

        res = train_and_evaluate_fold(
            model_factory, needs_scaling, train_df, test_df, features, target,
        )
        res["test_season"] = test_season
        res["train_seasons"] = train_seasons
        res["label"] = label
        results.append(res)

        rows.append({k: v for k, v in res.items()
                     if k not in ("predictions", "model")})

    return pd.DataFrame(rows), results


# ─── report ────────────────────────────────────────────────────────────────────

def print_report(label: str, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"{label.upper():^78}")
    print("=" * 78)
    show = summary[[
        "test_season", "n_train", "n_test",
        "train_mae", "test_mae", "test_rmse", "test_r2",
        "baseline_mae", "persistence_mae",
    ]].copy()
    for c in show.columns[3:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print(f"\nstability   test_mae = {summary['test_mae'].mean():.3f} ± {summary['test_mae'].std():.3f}")
    print(f"            test_r2  = {summary['test_r2'].mean():.3f} ± {summary['test_r2'].std():.3f}")


def print_comparison(
        summaries: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 78)
    print(f"{'MODEL COMPARISON — averaged across folds':^78}")
    print("=" * 78)
    rows = []
    for label, s in summaries.items():
        rows.append({
            "model": label,
            "test_mae_mean": round(s["test_mae"].mean(), 3),
            "test_mae_std": round(s["test_mae"].std(), 3),
            "test_rmse_mean": round(s["test_rmse"].mean(), 3),
            "test_r2_mean": round(s["test_r2"].mean(), 3),
            "vs_baseline": f"+{s['baseline_mae'].mean() - s['test_mae'].mean():.3f}",
            "vs_persistence": f"+{s['persistence_mae'].mean() - s['test_mae'].mean():.3f}",
        })
    out = pd.DataFrame(rows).sort_values("test_mae_mean")
    print(out.to_string(index=False))


def per_team_diagnostics(results: list[dict], label: str, top_n: int = 5) -> None:
    last = results[-1]
    preds = last["predictions"].copy()
    preds["abs_residual"] = preds["residual"].abs()
    worst = preds.groupby("team")["abs_residual"].mean().sort_values(ascending=False).head(top_n)
    best = preds.groupby("team")["abs_residual"].mean().sort_values().head(top_n)
    print(f"\n[{label}] last fold (test={last['test_season']})")
    print(f"  hardest:")
    print(worst.round(2).to_string())
    print(f"  easiest:")
    print(best.round(2).to_string())


def top_features(results: list[dict], label: str, top_n: int = 12) -> None:
    last = results[-1]
    pairs = last["model"].coefficients(FEATURES)
    header = "top coefficients" if label.startswith("Linear") else "top feature importances"
    print(f"\n[{label}] {header}:")
    for name, w in pairs[:top_n]:
        print(f"  {name:32s} {w:+.4f}")


# ─── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(GOLD_PATH).exists():
        raise FileNotFoundError(f"{GOLD_PATH} not found. Run pipeline.py first.")
    gold = pd.read_parquet(GOLD_PATH)
    logger.info("Gold: %s, seasons=%s", gold.shape, sorted(gold['season'].unique()))

    models = {
        "Linear (ridge)": dict(
            factory=lambda: LinearRegressionScratch(solver="normal", l2=1.0),
            needs_scaling=True,
        ),
        "XGBoost": dict(
            factory=lambda: XGBoostRegressorWrapper(),
            needs_scaling=False,
        ),
    }

    summaries: dict[str, pd.DataFrame] = {}
    all_results: dict[str, list[dict]] = {}

    for label, cfg in models.items():
        summary, results = walk_forward_backtest(
            gold,
            model_factory=cfg["factory"],
            needs_scaling=cfg["needs_scaling"],
            features=FEATURES,
            target=TARGET,
            label=label,
        )
        print_report(label, summary)
        per_team_diagnostics(results, label)
        top_features(results, label)
        summaries[label] = summary
        all_results[label] = results

    print_comparison(summaries)

    # ── model behavior plots ──────────────────────────────────────────────
    try:
        from visualization.model_plots import run_all_model_plots
        run_all_model_plots(
            results_by_model=all_results,
            gold=gold,
            features=FEATURES,
            save_dir="reports/plots/model",
        )
    except ImportError as exc:
        logger.warning("Skipping model plots: %s", exc)

    return summaries, all_results


if __name__ == "__main__":
    main()
