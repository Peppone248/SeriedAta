"""
model_comparison.py
Confronto sistematico di più classificatori sullo stesso split temporale.

Uso minimo in main.py:
    from model_comparison import run_model_comparison, plot_comparison_results
    results = run_model_comparison(raw_df)
    plot_comparison_results(results["leaderboard"])

Modifica rispetto alla versione precedente:
    - NUM_FEATURES / CAT_FEATURES ora importati da config.py
      (prima: da models.classification_pipeline — accoppiamento fragile)
    - clean_data / build_preprocessor ora importati da models.logistic_pipeline
    Nessun'altra modifica alla logica.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import add_new_features

# ── MODIFICA: import da config.py invece di models.classification_pipeline ──
from config import NUM_FEATURES, CAT_FEATURES
from models.logistic_pipeline import clean_data, build_preprocessor
from models.xgboost_pipeline import temporal_train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────
# DEFINIZIONE MODELLI E GRIGLIE
# ─────────────────────────────────────────────

def get_model_configs() -> list[dict]:
    """
    Restituisce la lista dei modelli da confrontare con le rispettive griglie.
    Ogni entry: { name, model, param_grid, needs_label_encoding }

    needs_label_encoding=True  → XGBoost con multi:softmax richiede y numerico
    needs_label_encoding=False → sklearn models accettano stringhe W/D/L

    Modelli inclusi:
      - LR Baseline : replica classification_model.py (max_iter=1000, no tuning).
                      Adattata al temporal split per confronto onesto.
                      Risponde a: "quanto vale il preprocessing da solo?"
      - LR Tuned    : stessa LR con GridSearch su C, solver, class_weight.
                      Risponde a: "quanto vale il tuning sopra al preprocessing?"
      - Random Forest, XGBoost, LightGBM: modelli alternativi.
    """
    return [
        # Baseline: specchia classification_model.py
        # max_iter=1000 come nell'originale, griglia fissa (nessun tuning).
        # Usa però temporal split + StandardScaler + OneHotEncoder.
        {
            "name": "LR Baseline",
            "model": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            "param_grid": {
                "model__C":            [1],
                "model__class_weight": [None],
            },
            "needs_label_encoding": False,
        },
        # LR Tuned: classification_pipeline.py con griglia estesa
        {
            "name": "LR Tuned",
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "param_grid": {
                "model__C":            [0.01, 0.1, 1, 5, 10, 50],
                "model__penalty":      ["l2"],
                "model__solver":       ["lbfgs", "saga"],
                "model__class_weight": [None, "balanced"],
            },
            "needs_label_encoding": False,
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [100, 300],
                "model__max_depth":    [None, 10, 20],
                "model__class_weight": [None, "balanced"],
            },
            "needs_label_encoding": False,
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                base_score=0.5,
                random_state=42,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators":    [100, 300],
                "model__max_depth":       [3, 5],
                "model__learning_rate":   [0.05, 0.1],
                "model__subsample":       [0.8, 1.0],
                "model__colsample_bytree":[0.8, 1.0],
            },
            "needs_label_encoding": True,
        },
        {
            "name": "LightGBM",
            "model": LGBMClassifier(
                objective="multiclass",
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "param_grid": {
                "model__n_estimators":  [100, 300],
                "model__max_depth":     [3, 5, -1],
                "model__learning_rate": [0.05, 0.1],
                "model__num_leaves":    [31, 63],
            },
            "needs_label_encoding": False,
        },
    ]


# ─────────────────────────────────────────────
# TRAINING SINGOLO MODELLO
# ─────────────────────────────────────────────

def _train_single(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    tscv:    TimeSeriesSplit,
    le:      LabelEncoder,
) -> dict:
    """
    Allena e valuta un singolo modello.
    Restituisce un dizionario con tutte le metriche.
    """
    name = config["name"]
    print(f"\n{'─'*50}")
    print(f"[{name}] training...")
    t0 = time.time()

    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", config["model"]),
    ])

    # XGBoost vuole y numerico
    if config["needs_label_encoding"]:
        y_tr = le.transform(y_train)
    else:
        y_tr = y_train

    grid = GridSearchCV(
        pipeline,
        config["param_grid"],
        cv=tscv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_tr)
    elapsed = time.time() - t0

    print(f"[{name}] best CV f1_macro: {grid.best_score_:.4f}  ({elapsed:.0f}s)")
    print(f"[{name}] best params: {grid.best_params_}")

    # predizioni sul test set
    y_pred_raw = grid.predict(X_test)

    if config["needs_label_encoding"]:
        y_pred = le.inverse_transform(y_pred_raw.astype(int))
    else:
        y_pred = y_pred_raw

    y_proba = grid.predict_proba(X_test)

    # metriche
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=["L", "D", "W"])

    metrics = {
        "model":         name,
        "accuracy":      round(accuracy_score(y_test, y_pred),             4),
        "f1_macro":      round(f1_score(y_test, y_pred, average="macro"),  4),
        "f1_weighted":   round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_L":          round(float(f1_per_class[0]),                     4),
        "f1_D":          round(float(f1_per_class[1]),                     4),
        "f1_W":          round(float(f1_per_class[2]),                     4),
        "log_loss":      round(log_loss(y_test, y_proba),                  4),
        "cv_f1_macro":   round(float(grid.best_score_),                    4),
        "train_time_s":  round(elapsed, 1),
        "best_params":   grid.best_params_,
    }

    print(f"[{name}] test → accuracy={metrics['accuracy']} | f1_macro={metrics['f1_macro']} "
          f"| f1 [L={metrics['f1_L']} D={metrics['f1_D']} W={metrics['f1_W']}]")

    return {
        "metrics":  metrics,
        "grid":     grid,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "report":   classification_report(y_test, y_pred, output_dict=True),
    }


# ─────────────────────────────────────────────
# PIPELINE CONFRONTO COMPLETO
# ─────────────────────────────────────────────

def run_model_comparison(
    df:         pd.DataFrame,
    test_ratio: float      = 0.2,
    cv_folds:   int        = 5,
    models:     list | None = None,
) -> dict:
    """
    Allena e confronta tutti i modelli sullo stesso split temporale.

    Args:
        df:          DataFrame con le feature già costruite (output di pipeline.py)
        test_ratio:  % del dataset usata come test set (split temporale)
        cv_folds:    numero di fold per TimeSeriesSplit nella CV
        models:      lista di nomi da eseguire (None = tutti)
                     es. ["LR Tuned", "XGBoost"]

    Returns:
        {
          "leaderboard":   pd.DataFrame  — tabella ordinata per f1_macro
          "results":       dict          — risultati per modello
          "y_test":        pd.Series
          "X_test":        pd.DataFrame
          "label_encoder": LabelEncoder
        }
    """
    df = add_new_features(df)
    df = clean_data(df)

    train_df, test_df = temporal_train_test_split(df, test_ratio=test_ratio)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df["result"]
    X_test  = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test  = test_df["result"]

    print(f"Train: {len(X_train)} righe | Test: {len(X_test)} righe")
    print(f"Distribuzione test: {y_test.value_counts().to_dict()}")

    # LabelEncoder condiviso — fit su train, usato solo per XGBoost
    le = LabelEncoder()
    le.fit(y_train)

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    configs = get_model_configs()
    if models:
        configs = [c for c in configs if c["name"] in models]

    all_results      = {}
    leaderboard_rows = []

    for config in configs:
        result = _train_single(config, X_train, y_train, X_test, y_test, tscv, le)
        all_results[config["name"]] = result
        leaderboard_rows.append(result["metrics"])

    leaderboard = (
        pd.DataFrame(leaderboard_rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.index += 1  # rank da 1

    print("\n" + "═"*60)
    print("LEADERBOARD")
    print("═"*60)
    print(leaderboard[["model","accuracy","f1_macro","f1_L","f1_D","f1_W",
                        "log_loss","cv_f1_macro","train_time_s"]].to_string())

    return {
        "leaderboard":   leaderboard,
        "results":       all_results,
        "y_test":        y_test,
        "X_test":        X_test,
        "label_encoder": le,
    }


# ─────────────────────────────────────────────
# VISUALIZZAZIONI
# ─────────────────────────────────────────────

def plot_comparison_results(leaderboard: pd.DataFrame):
    """
    Produce 3 grafici affiancati:
      1. f1_macro + accuracy per modello
      2. f1 per classe (L / D / W)
      3. log_loss vs tempo di training
    """
    models = leaderboard["model"].tolist()
    colors = ["#3266ad","#d84a30","#5ab27a","#d48a2b","#9b59b6"][:len(models)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. f1_macro + accuracy ──
    x     = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, leaderboard["f1_macro"], width, label="F1 macro", color=colors, alpha=0.9)
    axes[0].bar(x + width/2, leaderboard["accuracy"], width, label="Accuracy", color=colors, alpha=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylim(0, 1); axes[0].set_title("F1 macro vs Accuracy")
    axes[0].legend(); axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    # ── 2. f1 per classe ──
    bar_w = 0.25
    for i, (cls, col) in enumerate(zip(["f1_L","f1_D","f1_W"], ["#d84a30","#d48a2b","#5ab27a"])):
        axes[1].bar(x + (i-1)*bar_w, leaderboard[cls], bar_w,
                    label=cls.replace("f1_",""), color=col, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylim(0, 1); axes[1].set_title("F1 per classe (L / D / W)")
    axes[1].legend(title="Classe")
    axes[1].axhline(0.33, color="gray", linestyle="--", linewidth=0.8)

    # ── 3. log_loss vs train time ──
    for i, (_, row) in enumerate(leaderboard.iterrows()):
        axes[2].scatter(row["train_time_s"], row["log_loss"],
                        color=colors[i % len(colors)], s=120, zorder=5, label=row["model"])
        axes[2].annotate(row["model"], (row["train_time_s"], row["log_loss"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes[2].set_xlabel("Tempo di training (s)")
    axes[2].set_ylabel("Log loss (↓ meglio)")
    axes[2].set_title("Qualità probabilità vs Tempo")
    axes[2].legend(fontsize=8)

    plt.suptitle("Model Comparison — Serie A match prediction", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results: dict, y_test: pd.Series):
    """
    Plotta la confusion matrix per ogni modello su una griglia.
    """
    n     = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"], labels=["L","D","W"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["L","D","W"], yticklabels=["L","D","W"],
            ax=axes[i],
        )
        f1 = res["metrics"]["f1_macro"]
        axes[i].set_title(f"{name}\nf1_macro={f1:.3f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion matrices — tutti i modelli", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def print_leaderboard(leaderboard: pd.DataFrame):
    """Stampa la leaderboard formattata a console."""
    cols   = ["model","accuracy","f1_macro","f1_L","f1_D","f1_W","log_loss","cv_f1_macro","train_time_s"]
    labels = ["Modello","Accuracy","F1 macro","F1 Loss","F1 Draw","F1 Win","Log Loss","CV F1","Tempo (s)"]

    print("\n" + "═"*80)
    print(f"{'LEADERBOARD — ordinata per f1_macro':^80}")
    print("═"*80)
    display = leaderboard[cols].copy()
    display.columns = labels
    print(display.to_string(index=True))
    print("═"*80)

    best       = leaderboard.iloc[0]
    best_draw  = leaderboard.loc[leaderboard["f1_D"].idxmax()]
    print(f"\nMiglior f1_macro:     {best['model']} ({best['f1_macro']:.4f})")
    print(f"Miglior F1 sui draw:  {best_draw['model']} ({best_draw['f1_D']:.4f})")
    print(f"  (I draw sono la classe più difficile — monitorarla separatamente)")
