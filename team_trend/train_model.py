"""
train_model.py — train and evaluate squad-momentum regressors.

Production model:  XGBoost (better on test, handles non-linearity).
Reference model:   Linear (ridge) — kept as sanity baseline only.

Two targets supported, selectable via TARGET_NAME:
    next_5_matchweek_points  → absolute future points (team baseline + momentum)
    momentum_change_5        → deviation from current rolling form

Walk-forward CV by season. Both models train on identical features so the
comparison isolates model family vs feature signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from team_trend.models.linear_regression import (
    LinearRegressionScratch, Standardizer, mae, rmse, r2,
)
from team_trend.models.xgboost_regressor import XGBoostRegressorWrapper
from team_trend.models.xgboost_quantile import XGBoostQuantileRegressor, quantile_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLD_PATH = "data/gold/squad_momentum.parquet"

# choose which target to predict
# "next_5_matchweek_points"  → absolute points (the original target)
# "momentum_change_5"        → deviation from current form (isolates momentum)
TARGET_NAME = "next_5_matchweek_points"

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
    "roll5_points_std", "roll5_goal_diff_std",
]
CONTEXT_FEATURES = [
    "is_home", "days_rest",
    "cum_points", "cum_goal_diff", "league_position",
    "season_progress",
]
# new squad-quality features from player history (Phase A)
SQUAD_QUALITY_FEATURES = [
    "squad_quality_goals_per_90",
    "squad_quality_assists_per_90",
    "squad_quality_xg_proxy_per_90",
    "squad_quality_shots_per_90",
    "starter_avg_experience",
    "n_starters",
    "top_scorer_present",
    "top_assister_present",
]
# opponent-aware features (Phase B): avg pre-match form of the next 5 opponents
OPPONENT_FEATURES = [
    "opp5_avg_roll5_goals_against",
    "opp5_avg_roll5_save_pct",
    "opp5_avg_roll5_points",
    "opp5_avg_roll5_goal_diff",
    "opp5_avg_league_position",
    "opp5_avg_cum_points",
]
FEATURES = ROLL_FEATURES + CONTEXT_FEATURES + SQUAD_QUALITY_FEATURES + OPPONENT_FEATURES


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
    # persistence baseline is meaningful only for absolute-points target
    if target == "next_5_matchweek_points":
        persistence = (test_df["roll5_points"].to_numpy(float) * 5).clip(0, 15)
    else:
        # for momentum_change the persistence baseline is "no change" = 0
        persistence = np.zeros_like(y_te)

    out = {
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

    # if the model exposes quantile predictions, capture them and the diagnostics
    if hasattr(model, "predict_quantiles"):
        qpred_te = model.predict_quantiles(X_te)
        taus = sorted(qpred_te.keys())
        out["predictions"]["q_low"] = qpred_te[taus[0]]
        out["predictions"]["q_median"] = qpred_te[taus[1]]
        out["predictions"]["q_high"] = qpred_te[taus[2]]
        out["quantile_diagnostics"] = quantile_diagnostics(y_te, qpred_te)

    return out


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
    logger.info("[%s] walk-forward CV over %d folds, target=%s",
                label, len(folds), target)

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


def print_comparison(summaries: dict[str, pd.DataFrame]) -> None:
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


def top_features(results: list[dict], label: str, top_n: int = 15) -> None:
    last = results[-1]
    pairs = last["model"].coefficients(FEATURES)
    header = "top coefficients" if label.startswith("Linear") else "top feature importances"
    print(f"\n[{label}] {header}:")
    for name, w in pairs[:top_n]:
        marker = "  ← squad quality" if name in SQUAD_QUALITY_FEATURES else ""
        print(f"  {name:32s} {w:+.4f}{marker}")


# ─── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(GOLD_PATH).exists():
        raise FileNotFoundError(f"{GOLD_PATH} not found. Run pipeline.py first.")
    gold = pd.read_parquet(GOLD_PATH)
    logger.info("Gold: %s, seasons=%s", gold.shape, sorted(gold['season'].unique()))
    logger.info("Target: %s", TARGET_NAME)

    # filter out features not in this gold (squad quality may be absent for older runs)
    available_features = [f for f in FEATURES if f in gold.columns]
    missing = [f for f in FEATURES if f not in gold.columns]
    if missing:
        logger.warning("Features missing from Gold (will be skipped): %s", missing)
    logger.info("Active features: %d", len(available_features))

    # XGBoost is the production model
    # Linear is kept as a REFERENCE baseline (don't trust its absolute numbers,
    # use it as a sanity check — if XGBoost gains nothing over linear, something
    # is wrong; if linear suddenly beats XGBoost across folds, something changed)
    models = {
        "XGBoost (production)": dict(
            factory=lambda: XGBoostRegressorWrapper(),
            needs_scaling=False,
        ),
        "XGBoost Quantile": dict(
            factory=lambda: XGBoostQuantileRegressor(),
            needs_scaling=False,
        ),
        "Linear (reference)": dict(
            factory=lambda: LinearRegressionScratch(solver="normal", l2=1.0),
            needs_scaling=True,
        ),
    }

    summaries: dict[str, pd.DataFrame] = {}
    all_results: dict[str, list[dict]] = {}

    for label, cfg in models.items():
        summary, results = walk_forward_backtest(
            gold,
            model_factory=cfg["factory"],
            needs_scaling=cfg["needs_scaling"],
            features=available_features,
            target=TARGET_NAME,
            label=label,
        )
        print_report(label, summary)
        per_team_diagnostics(results, label)
        top_features(results, label)
        summaries[label] = summary
        all_results[label] = results

    print_comparison(summaries)

    # ── quantile-specific diagnostics ─────────────────────────────────────
    quant_label = "XGBoost Quantile"
    if quant_label in all_results:
        print("\n" + "=" * 78)
        print(f"{'QUANTILE DIAGNOSTICS — ' + quant_label:^78}")
        print("=" * 78)
        rows = []
        for r in all_results[quant_label]:
            d = r.get("quantile_diagnostics", {})
            if d:
                rows.append({
                    "test_season": r["test_season"],
                    "nominal_coverage": d["nominal_coverage"],
                    "actual_coverage": round(d["actual_coverage"], 3),
                    "sharpness": round(d["sharpness"], 2),
                    "pinball_low": round(d["pinball_low"], 3),
                    "pinball_median": round(d["pinball_median"], 3),
                    "pinball_high": round(d["pinball_high"], 3),
                })
        if rows:
            qdf = pd.DataFrame(rows)
            print(qdf.to_string(index=False))
            avg_cov = qdf["actual_coverage"].mean()
            avg_sharp = qdf["sharpness"].mean()
            print(f"\n  mean actual_coverage: {avg_cov:.3f} (target: {rows[0]['nominal_coverage']:.2f})")
            print(f"  mean sharpness:       {avg_sharp:.2f} points wide")
            if avg_cov < rows[0]["nominal_coverage"] - 0.05:
                print("  -> intervals are TOO NARROW (overconfident)")
            elif avg_cov > rows[0]["nominal_coverage"] + 0.05:
                print("  -> intervals are TOO WIDE (underconfident)")
            else:
                print("  -> well calibrated")

    # ── permutation importance audit on the production model ─────────────
    try:
        from team_trend.models.permutation_audit import permutation_importance, report_audit
        last = all_results["XGBoost (production)"][-1]
        test_season = last["test_season"]
        # rebuild the test arrays from the same gold rows for this fold
        test_df = gold[gold["season"] == test_season].dropna(subset=available_features + [TARGET_NAME])
        X_test = test_df[available_features].to_numpy(float)
        y_test = test_df[TARGET_NAME].to_numpy(float)
        audit = permutation_importance(
            model=last["model"],
            X_test=X_test,
            y_test=y_test,
            feature_names=available_features,
            n_repeats=10,
        )
        drop_candidates = report_audit(
            audit, drop_thresh=0.005, label=f"XGBoost on test={test_season}",
        )
    except Exception as exc:
        logger.warning("Permutation audit skipped: %s", exc)

    # ── feature-group ablation ────────────────────────────────────────────
    try:
        from team_trend.models.ablation import run_group_ablation, print_ablation_report

        # disjoint partition of features into conceptual groups
        feature_groups = {
            "core_form": ["roll5_points", "roll5_goal_diff",
                          "roll5_goals_for", "roll5_goals_against"],
            "possession": ["roll5_possession"],
            "shooting": ["roll5_shots", "roll5_shots_on_target",
                         "roll5_shots_on_target_pct", "roll5_goals_per_shot"],
            "keeper": ["roll5_save_pct", "roll5_saves"],
            "rotation": ["roll5_players_used", "roll5_starters_used",
                         "roll5_minutes_std", "roll5_squad_age_mean"],
            "defense": ["roll5_sum_tackles_won", "roll5_sum_interceptions"],
            "discipline": ["roll5_sum_yellow_cards", "roll5_sum_fouls"],
            "volatility": ["roll5_points_std", "roll5_goal_diff_std"],
            "context": ["is_home", "days_rest"],
            "standings": ["cum_points", "cum_goal_diff",
                          "league_position", "season_progress"],
            "squad_quality": SQUAD_QUALITY_FEATURES,
            "opponent": OPPONENT_FEATURES,
        }
        # restrict each group to features actually in gold
        feature_groups = {
            k: [f for f in v if f in gold.columns]
            for k, v in feature_groups.items()
        }

        audit_df, baseline_mae = run_group_ablation(
            gold=gold,
            target=TARGET_NAME,
            feature_groups=feature_groups,
            model_factory=lambda: XGBoostRegressorWrapper(),
        )
        print_ablation_report(audit_df, baseline_mae)

        # save plot
        from team_trend.visualization.model_plots import plot_ablation_results
        plot_ablation_results(audit_df, baseline_mae, save_dir="reports/plots/model")
    except Exception as exc:
        logger.warning("Ablation skipped: %s", exc)

    # model behavior plots
    try:
        from team_trend.visualization.model_plots import run_all_model_plots
        run_all_model_plots(
            results_by_model=all_results,
            gold=gold,
            features=available_features,
            save_dir="reports/plots/model",
        )
    except ImportError as exc:
        logger.warning("Skipping model plots: %s", exc)

    return summaries, all_results


if __name__ == "__main__":
    main()
