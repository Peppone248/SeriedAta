"""
backtesting.py — walk-forward season-by-season backtesting.

Scheme (expanding window):
    Fold 1: train [s1]             → test [s2]
    Fold 2: train [s1, s2]         → test [s3]
    Fold 3: train [s1, s2, s3]     → test [s4]
    ...

The training set grows by one season at every fold, exactly replicating
real deployment: the model never sees the future.

Design:
    Model-agnostic — receives build/train functions as parameters instead
    of importing a specific pipeline. This allows Logistic, XGBoost and
    LightGBM to be compared on identical folds with a single call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder


# ─── OUTPUT CONTRACTS ────────────────────────────────────────────────────────

@dataclass
class BacktestFold:
    """Results for a single seasonal fold."""
    model_name:    str
    test_season:   str
    train_seasons: list[str]
    n_train:       int
    n_test:        int
    accuracy:      float
    f1_macro:      float
    f1_L:          float
    f1_D:          float
    f1_W:          float
    log_loss:      float


@dataclass
class BacktestResult:
    """Aggregated results for the full backtest."""
    model_name:    str
    folds:         list[BacktestFold] = field(default_factory=list)

    # ── stability metrics (populated by summarize()) ──────────────────────
    mean_accuracy: float = 0.0
    std_accuracy:  float = 0.0
    mean_f1_macro: float = 0.0
    std_f1_macro:  float = 0.0
    mean_f1_D:     float = 0.0
    std_f1_D:      float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.folds])

    def summarize(self) -> "BacktestResult":
        df = self.to_dataframe()
        self.mean_accuracy = round(df["accuracy"].mean(), 4)
        self.std_accuracy  = round(df["accuracy"].std(),  4)
        self.mean_f1_macro = round(df["f1_macro"].mean(), 4)
        self.std_f1_macro  = round(df["f1_macro"].std(),  4)
        self.mean_f1_D     = round(df["f1_D"].mean(),     4)
        self.std_f1_D      = round(df["f1_D"].std(),      4)
        return self


# ─── CORE ────────────────────────────────────────────────────────────────────

def backtest_by_season(
    df:                   pd.DataFrame,
    build_pipeline_fn:    Callable,
    train_fn:             Callable,
    num_features:         list[str],
    cat_features:         list[str],
    model_name:           str   = "Model",
    needs_label_encoding: bool  = False,
    min_train_seasons:    int   = 2,
    min_test_samples:     int   = 200,
    best_params:          dict | None = None,
) -> BacktestResult:
    """
    Walk-forward season-by-season backtest.

    Args:
        df:                   DataFrame with 'season' column and built features.
        build_pipeline_fn:    Returns an unfitted sklearn Pipeline.
                              e.g. models.xgboost_pipeline.build_model_pipeline
        train_fn:             (X, y, pipeline) → fitted GridSearchCV.
                              e.g. models.xgboost_pipeline.train_model
        num_features:         Numerical feature list (from config.py).
        cat_features:         Categorical feature list (from config.py).
        model_name:           Label used in plots and reports.
        needs_label_encoding: True for XGBoost (requires integer y).
        min_train_seasons:    Minimum seasons before testing begins.
        min_test_samples:     Minimum rows in test set (skips partial seasons).
        best_params:          If provided, bypasses GridSearch and uses these
                              parameters directly — much faster per fold.
                              e.g. grid.best_params_ from the main training run.

    Returns:
        BacktestResult with one BacktestFold per tested season.
    """
    if "season" not in df.columns:
        raise ValueError("DataFrame must have a 'season' column.")

    seasons = sorted(df["season"].unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(
            f"Need at least {min_train_seasons + 1} seasons "
            f"({min_train_seasons} train + 1 test). Found: {len(seasons)}."
        )

    result      = BacktestResult(model_name=model_name)
    all_features = num_features + cat_features

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"] == test_season].copy()

        train_df = train_df.dropna(subset=all_features + ["result"])
        test_df  = test_df.dropna(subset=all_features + ["result"])

        if len(test_df) < min_test_samples:
            print(f"  [skip] season {test_season} — test set too small "
                  f"({len(test_df)} rows < {min_test_samples}). "
                  f"Season likely incomplete.")
            continue

        X_train = train_df[all_features]
        y_train = train_df["result"]
        X_test  = test_df[all_features]
        y_test  = test_df["result"]

        print(f"\n  Fold: train {list(train_seasons)} → test {test_season}")
        print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # ── label encoding (XGBoost only) ─────────────────────────────────
        le = LabelEncoder()
        if needs_label_encoding:
            y_train_fit = le.fit_transform(y_train)
        else:
            le.fit(y_train)
            y_train_fit = y_train

        # ── training ──────────────────────────────────────────────────────
        pipeline = build_pipeline_fn()

        if best_params is not None:
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train_fit)
            fitted = pipeline
        else:
            fitted = train_fn(X_train, y_train_fit, pipeline)

        # ── predictions ───────────────────────────────────────────────────
        y_pred_raw = fitted.predict(X_test)
        y_proba    = fitted.predict_proba(X_test)

        if needs_label_encoding:
            y_pred = le.inverse_transform(y_pred_raw.astype(int))
        else:
            y_pred = y_pred_raw

        # ── metrics ───────────────────────────────────────────────────────
        classes   = ["L", "D", "W"]
        f1_values = f1_score(y_test, y_pred, average=None, labels=classes,
                             zero_division=0)

        try:
            ll = log_loss(y_test, y_proba, labels=list(le.classes_))
        except Exception:
            ll = float("nan")

        fold = BacktestFold(
            model_name    = model_name,
            test_season   = str(test_season),
            train_seasons = [str(s) for s in train_seasons],
            n_train       = len(X_train),
            n_test        = len(X_test),
            accuracy      = round(float(accuracy_score(y_test, y_pred)), 4),
            f1_macro      = round(float(f1_score(y_test, y_pred, average="macro",
                                                  zero_division=0)), 4),
            f1_L          = round(float(f1_values[0]), 4),
            f1_D          = round(float(f1_values[1]), 4),
            f1_W          = round(float(f1_values[2]), 4),
            log_loss      = round(ll, 4),
        )

        result.folds.append(fold)
        _print_fold_summary(fold)

    result.summarize()
    return result


# ─── MULTI-MODEL ─────────────────────────────────────────────────────────────

def compare_models_backtest(
    df:            pd.DataFrame,
    model_configs: list[dict],
) -> list[BacktestResult]:
    """
    Run backtest for multiple models on the same folds.

    Args:
        df:             DataFrame with built features.
        model_configs:  List of dicts, each with keys:
                        {
                          "model_name":           str,
                          "build_pipeline_fn":    callable,
                          "train_fn":             callable,
                          "num_features":         list[str],
                          "cat_features":         list[str],
                          "needs_label_encoding": bool,
                          "best_params":          dict | None,
                        }

    Returns:
        List of BacktestResult, one per model.
    """
    results = []
    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"  BACKTEST — {cfg['model_name']}")
        print(f"{'=' * 60}")
        r = backtest_by_season(df, **cfg)
        results.append(r)
    return results


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_backtest_results(results: list[BacktestResult]) -> None:
    """
    4 panels:
      1. F1 macro over time per model
      2. F1 per class (L / D / W) for the first model
      3. Accuracy over time
      4. Stability summary (mean ± std as error bars)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors    = ["#3266ad", "#d84a30", "#5ab27a", "#d48a2b"]

    # ── 1. F1 macro over time ─────────────────────────────────────────────
    ax = axes[0, 0]
    for r, color in zip(results, colors):
        df_r = r.to_dataframe()
        ax.plot(df_r["test_season"], df_r["f1_macro"],
                marker="o", label=r.model_name, color=color, linewidth=2)
        ax.axhline(r.mean_f1_macro, linestyle="--", color=color,
                   alpha=0.4, linewidth=1)
    ax.set_title("F1 macro per season")
    ax.set_xlabel("Test season")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. F1 per class (first model) ────────────────────────────────────
    ax   = axes[0, 1]
    r0   = results[0]
    df_r = r0.to_dataframe()
    for col, color in [("f1_L", "#d84a30"), ("f1_D", "#d48a2b"), ("f1_W", "#5ab27a")]:
        ax.plot(df_r["test_season"], df_r[col],
                marker="s", label=col.replace("f1_", ""),
                color=color, linewidth=2)
    ax.axhline(0.33, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
    ax.set_title(f"F1 per class — {r0.model_name}")
    ax.set_xlabel("Test season")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Class")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 3. Accuracy over time ─────────────────────────────────────────────
    ax = axes[1, 0]
    for r, color in zip(results, colors):
        df_r = r.to_dataframe()
        ax.plot(df_r["test_season"], df_r["accuracy"],
                marker="^", label=r.model_name, color=color, linewidth=2)
    ax.set_title("Accuracy per season")
    ax.set_xlabel("Test season")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 4. Stability summary (mean ± std) ─────────────────────────────────
    ax       = axes[1, 1]
    names    = [r.model_name    for r in results]
    mean_f1  = [r.mean_f1_macro for r in results]
    std_f1   = [r.std_f1_macro  for r in results]
    x        = np.arange(len(results))

    bars = ax.bar(x, mean_f1, color=colors[:len(results)], alpha=0.85)
    ax.errorbar(x, mean_f1, yerr=std_f1, fmt="none",
                color="black", capsize=5, linewidth=1.5)

    for bar, mean, std in zip(bars, mean_f1, std_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Stability — F1 macro (mean ± std)")
    ax.set_ylabel("F1 macro")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Walk-forward Backtest — season by season", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─── TEXT REPORT ─────────────────────────────────────────────────────────────

def print_backtest_summary(results: list[BacktestResult]) -> None:
    """Print the stability leaderboard and fold-by-fold detail."""
    print("\n" + "═" * 70)
    print(f"{'BACKTEST SUMMARY':^70}")
    print("═" * 70)

    rows = [{
        "model":         r.model_name,
        "mean_f1_macro": r.mean_f1_macro,
        "std_f1_macro":  r.std_f1_macro,
        "mean_accuracy": r.mean_accuracy,
        "std_accuracy":  r.std_accuracy,
        "mean_f1_D":     r.mean_f1_D,
        "std_f1_D":      r.std_f1_D,
    } for r in results]

    lb = (pd.DataFrame(rows)
          .sort_values("mean_f1_macro", ascending=False)
          .reset_index(drop=True))
    lb.index += 1

    print("\n  Stability over time (mean ± std across all valid folds):\n")
    print(lb.to_string())

    for r in results:
        print(f"\n  {'─' * 60}")
        print(f"  {r.model_name} — fold detail")
        print(f"  {'─' * 60}")
        df_r = r.to_dataframe()[
            ["test_season", "n_train", "n_test",
             "accuracy", "f1_macro", "f1_L", "f1_D", "f1_W"]
        ]
        print(df_r.to_string(index=False))

    print("\n" + "═" * 70)
    _print_stability_insight(results)


def _print_fold_summary(fold: BacktestFold) -> None:
    print(f"  → accuracy={fold.accuracy:.4f}  f1_macro={fold.f1_macro:.4f}  "
          f"[L={fold.f1_L:.3f}  D={fold.f1_D:.3f}  W={fold.f1_W:.3f}]")


def _print_stability_insight(results: list[BacktestResult]) -> None:
    """Auto-generated stability comment per model."""
    for r in results:
        n_folds   = len(r.folds)
        stability = (
            "stable"              if r.std_f1_macro < 0.05 else
            "moderately variable" if r.std_f1_macro < 0.10 else
            "unstable — possible drift between seasons"
        )
        print(f"  {r.model_name}: mean f1_macro={r.mean_f1_macro:.3f} "
              f"(std={r.std_f1_macro:.3f}, {n_folds} valid folds) → {stability}")