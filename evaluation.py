"""
evaluation.py — confronto unificato di modelli.

Funziona su ClassificationResult e RegressionResult da models/base.py.
Poiché tutti i modelli restituiscono lo stesso contratto, non ci sono
if/else né chiavi hardcoded: si itera direttamente sulla lista di risultati.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.base import ClassificationResult, RegressionResult


# ─── CLASSIFICATION — LEADERBOARD ────────────────────────────────────────────

def classification_leaderboard(results: list[ClassificationResult]) -> pd.DataFrame:
    """
    Tabella comparativa ordinata per f1_macro.

    Ogni riga corrisponde a un ClassificationResult.
    Nessun accesso diretto a dict interni — tutto via attributi del dataclass.
    """
    rows = [
        {
            "model":    r.model_name,
            "accuracy": round(r.accuracy, 4),
            "f1_macro": round(r.f1_macro, 4),
            "f1_L":     round(r.f1_per_class.get("L", float("nan")), 4),
            "f1_D":     round(r.f1_per_class.get("D", float("nan")), 4),
            "f1_W":     round(r.f1_per_class.get("W", float("nan")), 4),
            "log_loss": round(r.log_loss, 4) if r.log_loss is not None else None,
        }
        for r in results
    ]
    lb = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    lb.index = range(1, len(lb) + 1)
    return lb


def print_classification_leaderboard(results: list[ClassificationResult]) -> None:
    lb = classification_leaderboard(results)
    print("\n" + "═" * 70)
    print(f"{'CLASSIFICATION LEADERBOARD':^70}")
    print("═" * 70)
    print(lb.to_string())
    print("═" * 70)

    best         = lb.iloc[0]
    best_draw    = lb.loc[lb["f1_D"].idxmax()]
    print(f"\n  Miglior f1_macro : {best['model']} ({best['f1_macro']:.4f})")
    print(f"  Miglior F1 draw  : {best_draw['model']} ({best_draw['f1_D']:.4f})")
    print("  (i draw sono la classe più difficile — monitorarla separatamente)")


# ─── CLASSIFICATION — PLOTS ──────────────────────────────────────────────────

def plot_classification_comparison(results: list[ClassificationResult]) -> None:
    """
    3 pannelli affiancati:
      1. f1_macro + accuracy per modello
      2. f1 per classe (L / D / W)
      3. log_loss (pannello vuoto se nessun modello lo calcola)
    """
    lb = classification_leaderboard(results).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. f1_macro vs accuracy ───────────────────────────────────────────
    x, w = np.arange(len(lb)), 0.35
    axes[0].bar(x - w / 2, lb["f1_macro"], w, label="F1 macro", alpha=0.9)
    axes[0].bar(x + w / 2, lb["accuracy"], w, label="Accuracy", alpha=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lb["model"], rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("F1 macro vs Accuracy")
    axes[0].legend()

    # ── 2. f1 per classe ─────────────────────────────────────────────────
    bar_w  = 0.25
    class_colors = {"f1_L": "#d84a30", "f1_D": "#d48a2b", "f1_W": "#5ab27a"}
    for i, (col, color) in enumerate(class_colors.items()):
        axes[1].bar(x + (i - 1) * bar_w, lb[col], bar_w,
                    label=col.replace("f1_", ""), color=color, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lb["model"], rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.33, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("F1 per classe (L / D / W)")
    axes[1].legend(title="Classe")

    # ── 3. log_loss ───────────────────────────────────────────────────────
    ll_data = lb.dropna(subset=["log_loss"])
    if not ll_data.empty:
        axes[2].bar(ll_data["model"], ll_data["log_loss"], alpha=0.85)
        axes[2].set_title("Log Loss (↓ meglio)")
        axes[2].tick_params(axis="x", rotation=15)
    else:
        axes[2].text(0.5, 0.5, "log_loss non disponibile",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Log Loss")

    plt.suptitle("Model Comparison — Classificazione", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results: list[ClassificationResult]) -> None:
    """Confusion matrix side-by-side per ogni ClassificationResult."""
    n     = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes      = np.array(axes).flatten()

    for i, r in enumerate(results):
        cm = confusion_matrix(r.y_test, r.predictions, labels=["L", "D", "W"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["L", "D", "W"], yticklabels=["L", "D", "W"],
            ax=axes[i],
        )
        axes[i].set_title(f"{r.model_name}\nf1_macro={r.f1_macro:.3f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion matrices — tutti i modelli", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─── REGRESSION — LEADERBOARD ────────────────────────────────────────────────

def regression_leaderboard(results: list[RegressionResult]) -> pd.DataFrame:
    rows = [
        {
            "model":   r.model_name,
            "mae":     round(r.mae,     4),
            "rmse":    round(r.rmse,    4),
            "r2":      round(r.r2,      4),
            "cv_mae":  round(r.cv_mae,  4),
            "cv_rmse": round(r.cv_rmse, 4),
            "cv_r2":   round(r.cv_r2,   4),
        }
        for r in results
    ]
    lb = pd.DataFrame(rows).sort_values("r2", ascending=False)
    lb.index = range(1, len(lb) + 1)
    return lb


def print_regression_leaderboard(results: list[RegressionResult]) -> None:
    lb = regression_leaderboard(results)
    print("\n" + "═" * 60)
    print(f"{'REGRESSION LEADERBOARD':^60}")
    print("═" * 60)
    print(lb.to_string())
    print("═" * 60)
