"""
backtesting.py — walk-forward backtesting stagione per stagione.

Schema:
    Fold 1: train [s1]         → test [s2]
    Fold 2: train [s1, s2]     → test [s3]
    Fold 3: train [s1, s2, s3] → test [s4]
    ...

Ad ogni fold il training set cresce (expanding window), esattamente
come avviene nel deployment reale: il modello non vede mai il futuro.

Design:
    Il modulo è model-agnostic — riceve le funzioni di build/train
    come parametri invece di importare una pipeline specifica.
    Questo permette di confrontare Logistic, XGBoost e LightGBM
    sugli stessi fold con una sola chiamata a compare_models_backtest().
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


# ─── OUTPUT CONTRACT ─────────────────────────────────────────────────────────

@dataclass
class BacktestFold:
    """Risultati di un singolo fold stagionale."""
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
    """Risultati aggregati dell'intero backtest."""
    model_name:  str
    folds:       list[BacktestFold]    = field(default_factory=list)

    # ── statistiche di stabilità (calcolate da summarize()) ──────────────
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
    df:                    pd.DataFrame,
    build_pipeline_fn:     Callable,
    train_fn:              Callable,
    num_features:          list[str],
    cat_features:          list[str],
    model_name:            str   = "Model",
    needs_label_encoding:  bool  = False,
    min_train_seasons:     int   = 2,
    best_params:           dict | None = None,
) -> BacktestResult:
    """
    Walk-forward backtesting stagione per stagione.

    Args:
        df:                   DataFrame con colonna "season" e feature già costruite.
        build_pipeline_fn:    funzione che ritorna un Pipeline sklearn non fittato.
                              Es: models.xgboost_pipeline.build_model_pipeline
        train_fn:             funzione (X, y, pipeline) → GridSearchCV fittato.
                              Es: models.xgboost_pipeline.train_model
        num_features:         lista feature numeriche (da config.py).
        cat_features:         lista feature categoriali (da config.py).
        model_name:           nome del modello per i plot e i report.
        needs_label_encoding: True per XGBoost (richiede y numerico).
        min_train_seasons:    numero minimo di stagioni di training prima
                              di iniziare il test. Default=2.
        best_params:          se fornito, bypassa il GridSearch e usa questi
                              parametri direttamente — molto più veloce.
                              Es: grid.best_params_ dal training principale.

    Returns:
        BacktestResult con un BacktestFold per ogni stagione testata.
    """
    if "season" not in df.columns:
        raise ValueError("Il DataFrame deve avere una colonna 'season'.")

    seasons = sorted(df["season"].unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(
            f"Servono almeno {min_train_seasons + 1} stagioni "
            f"({min_train_seasons} train + 1 test). Trovate: {len(seasons)}."
        )

    result = BacktestResult(model_name=model_name)
    all_features = num_features + cat_features

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"] == test_season].copy()

        # dropna solo sulle feature usate
        train_df = train_df.dropna(subset=all_features + ["result"])
        test_df  = test_df.dropna(subset=all_features + ["result"])

        if test_df.empty:
            print(f"  [skip] stagione {test_season} — nessuna riga valida nel test.")
            continue

        X_train = train_df[all_features]
        y_train = train_df["result"]
        X_test  = test_df[all_features]
        y_test  = test_df["result"]

        print(f"\n  Fold: train {list(train_seasons)} → test {test_season}")
        print(f"  Train: {len(X_train)} righe | Test: {len(X_test)} righe")

        # ── label encoding (solo XGBoost) ────────────────────────────────
        le = LabelEncoder()
        if needs_label_encoding:
            y_train_fit = le.fit_transform(y_train)
        else:
            le.fit(y_train)
            y_train_fit = y_train

        # ── training ─────────────────────────────────────────────────────
        pipeline = build_pipeline_fn()

        if best_params is not None:
            # usa i parametri ottimali già trovati — evita grid search per fold
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train_fit)
            fitted = pipeline
        else:
            fitted = train_fn(X_train, y_train_fit, pipeline)

        # ── predizioni ───────────────────────────────────────────────────
        y_pred_raw = fitted.predict(X_test)
        y_proba    = fitted.predict_proba(X_test)

        if needs_label_encoding:
            y_pred = le.inverse_transform(y_pred_raw.astype(int))
        else:
            y_pred = y_pred_raw

        # ── metriche ─────────────────────────────────────────────────────
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
        _print_fold(fold)

    result.summarize()
    return result


# ─── MULTI-MODEL ─────────────────────────────────────────────────────────────

def compare_models_backtest(
    df: pd.DataFrame,
    model_configs: list[dict],
) -> list[BacktestResult]:
    """
    Esegue il backtest per più modelli sugli stessi fold.

    Args:
        df:             DataFrame con feature già costruite.
        model_configs:  lista di dict, ognuno con le chiavi:
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
        Lista di BacktestResult, uno per modello.
    """
    results = []
    for cfg in model_configs:
        print(f"\n{'='*60}")
        print(f"  BACKTEST — {cfg['model_name']}")
        print(f"{'='*60}")
        r = backtest_by_season(df, **cfg)
        results.append(r)
    return results


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_backtest_results(results: list[BacktestResult]) -> None:
    """
    4 pannelli:
      1. F1 macro nel tempo per ogni modello
      2. F1 per classe (L / D / W) — solo primo modello o modello scelto
      3. Accuracy nel tempo
      4. Stability summary (mean ± std come errorbar)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = ["#3266ad", "#d84a30", "#5ab27a", "#d48a2b"]

    # ── 1. F1 macro nel tempo ─────────────────────────────────────────────
    ax = axes[0, 0]
    for r, color in zip(results, colors):
        df_r = r.to_dataframe()
        ax.plot(df_r["test_season"], df_r["f1_macro"],
                marker="o", label=r.model_name, color=color, linewidth=2)
        ax.axhline(r.mean_f1_macro, linestyle="--", color=color,
                   alpha=0.4, linewidth=1)
    ax.set_title("F1 macro per stagione")
    ax.set_xlabel("Stagione di test")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. F1 per classe (primo modello) ─────────────────────────────────
    ax = axes[0, 1]
    r0   = results[0]
    df_r = r0.to_dataframe()
    class_colors = {"f1_L": "#d84a30", "f1_D": "#d48a2b", "f1_W": "#5ab27a"}
    for col, color in class_colors.items():
        ax.plot(df_r["test_season"], df_r[col],
                marker="s", label=col.replace("f1_", ""),
                color=color, linewidth=2)
    ax.axhline(0.33, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
    ax.set_title(f"F1 per classe — {r0.model_name}")
    ax.set_xlabel("Stagione di test")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Classe")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 3. Accuracy nel tempo ─────────────────────────────────────────────
    ax = axes[1, 0]
    for r, color in zip(results, colors):
        df_r = r.to_dataframe()
        ax.plot(df_r["test_season"], df_r["accuracy"],
                marker="^", label=r.model_name, color=color, linewidth=2)
    ax.set_title("Accuracy per stagione")
    ax.set_xlabel("Stagione di test")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # ── 4. Stability summary (mean ± std) ─────────────────────────────────
    ax = axes[1, 1]
    model_names  = [r.model_name    for r in results]
    mean_f1      = [r.mean_f1_macro for r in results]
    std_f1       = [r.std_f1_macro  for r in results]

    x = np.arange(len(results))
    bars = ax.bar(x, mean_f1, color=colors[:len(results)], alpha=0.85)
    ax.errorbar(x, mean_f1, yerr=std_f1, fmt="none",
                color="black", capsize=5, linewidth=1.5)

    for bar, mean, std in zip(bars, mean_f1, std_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Stabilità — F1 macro  (mean ± std)")
    ax.set_ylabel("F1 macro")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Walk-forward Backtest — stagione per stagione", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─── REPORT TESTUALE ─────────────────────────────────────────────────────────

def print_backtest_summary(results: list[BacktestResult]) -> None:
    """
    Stampa la leaderboard di stabilità e il dettaglio fold per fold.
    """
    print("\n" + "═" * 70)
    print(f"{'BACKTEST SUMMARY':^70}")
    print("═" * 70)

    # leaderboard stabilità
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

    print("\n  Stabilità nel tempo (mean ± std su tutti i fold):\n")
    print(lb.to_string())

    # dettaglio fold per fold
    for r in results:
        print(f"\n  {'─'*60}")
        print(f"  {r.model_name} — dettaglio per stagione")
        print(f"  {'─'*60}")
        df_r = r.to_dataframe()[
            ["test_season", "n_train", "n_test",
             "accuracy", "f1_macro", "f1_L", "f1_D", "f1_W"]
        ]
        print(df_r.to_string(index=False))

    print("\n" + "═" * 70)
    _print_stability_insight(results)


def _print_fold(fold: BacktestFold) -> None:
    print(f"  → accuracy={fold.accuracy:.4f}  f1_macro={fold.f1_macro:.4f}  "
          f"[L={fold.f1_L:.3f}  D={fold.f1_D:.3f}  W={fold.f1_W:.3f}]")


def _print_stability_insight(results: list[BacktestResult]) -> None:
    """Commento automatico sulla stabilità."""
    for r in results:
        stability = "stabile" if r.std_f1_macro < 0.05 else \
                    "moderatamente variabile" if r.std_f1_macro < 0.10 else \
                    "instabile — potrebbe esserci drift tra stagioni"

        print(f"  {r.model_name}: f1_macro medio={r.mean_f1_macro:.3f} "
              f"(std={r.std_f1_macro:.3f}) → {stability}")