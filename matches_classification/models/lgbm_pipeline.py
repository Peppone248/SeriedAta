"""
models/lgbm_pipeline.py — pipeline LightGBM con split temporale.

Restituisce ClassificationResult per confronto uniforme con Logistic e XGBoost.

Differenze rispetto a xgboost_pipeline:
  - LGBMClassifier accetta label stringa (W/D/L) → nessun LabelEncoder
  - param_grid orientato ai parametri tipici di LGBM (num_leaves, min_child_samples)
  - label_encoder = None nel ClassificationResult risultante
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, log_loss,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

from matches_classification.config import LGBM_NUM_FEATURES, CAT_FEATURES
from matches_classification.models.base import ClassificationResult

NUM_FEATURES = LGBM_NUM_FEATURES


# ─── DATA PREP ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date", ascending=True)
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["result"])
    return df


def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    cutoff = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff], df.iloc[cutoff:]


# ─── PREPROCESSING ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def build_model_pipeline() -> Pipeline:
    model = LGBMClassifier(
        objective    = "multiclass",
        num_class    = 3,
        random_state = 42,
        n_jobs       = -1,
        verbose      = -1,
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model",        model),
    ])


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, pipeline) -> GridSearchCV:
    param_grid = {
        "model__n_estimators":      [100, 300],
        "model__max_depth":         [3, 5, -1],
        "model__learning_rate":     [0.05, 0.1],
        "model__num_leaves":        [31, 63],
        "model__min_child_samples": [20, 50],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv      = TimeSeriesSplit(n_splits=5),
        scoring = "f1_macro",
        n_jobs  = -1,
        verbose = 1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params:      {grid.best_params_}")
    print(f"  Best CV f1_macro: {grid.best_score_:.4f}")
    return grid


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def _compute_metrics(model, X_test, y_test) -> dict:
    """
    Metriche unificate — stessa struttura di logistic e xgboost _compute_metrics().
    LGBM restituisce le probabilità in ordine alfabetico delle classi (D, L, W),
    coerente con come log_loss si aspetta i label.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = ["L", "D", "W"]

    # ordine classi nel predict_proba di LGBM — dipende dal LabelEncoder interno
    lgbm_classes = model.classes_
    f1_values    = f1_score(y_test, y_pred, average=None, labels=classes)
    ll           = log_loss(y_test, y_proba, labels=list(lgbm_classes))

    return {
        "accuracy":      float(accuracy_score(y_test, y_pred)),
        "f1_macro":      float(f1_score(y_test, y_pred, average="macro")),
        "f1_per_class":  dict(zip(classes, f1_values.tolist())),
        "log_loss":      float(ll),
        "report":        classification_report(y_test, y_pred, output_dict=True),
        "predictions":   y_pred,
        "probabilities": y_proba,
    }


# ─── TABELLE OUTPUT ──────────────────────────────────────────────────────────

def _build_prediction_table(model, X_test, y_test) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    out = X_test.copy()
    out["actual"]    = y_test.values
    out["predicted"] = y_pred
    out["correct"]   = out["actual"] == out["predicted"]
    return out


def _build_probability_table(model, X_test, y_test) -> pd.DataFrame:
    probs   = model.predict_proba(X_test)
    classes = model.classes_        # ordine interno LGBM (alfabetico: D, L, W)
    out = X_test.copy()
    out["actual"] = y_test.values
    for i, cls in enumerate(classes):
        out[f"P_{cls}"] = probs[:, i]
    out["confidence"] = probs.max(axis=1)
    return out


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=["L", "D", "W"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["L", "D", "W"], yticklabels=["L", "D", "W"])
    plt.title("Confusion Matrix — LightGBM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def run_classification_pipeline(
    df: pd.DataFrame,
    run_eda_flag: bool = False,
) -> ClassificationResult:
    """
    Pipeline completa LightGBM con split temporale.
    Restituisce ClassificationResult — stesso contratto di Logistic e XGBoost.
    """
    all_features = NUM_FEATURES + CAT_FEATURES
    duplicates   = [f for f in all_features if all_features.count(f) > 1]
    if duplicates:
        raise ValueError(
            f"Feature duplicate in LGBM_NUM_FEATURES: {sorted(set(duplicates))}"
        )

    df = clean_data(df)

    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df["result"]
    X_test  = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test  = test_df["result"]

    print(f"  Train: {len(X_train)} righe | Test: {len(X_test)} righe")

    pipeline = build_model_pipeline()
    grid     = train_model(X_train, y_train, pipeline)

    best    = grid.best_estimator_
    metrics = _compute_metrics(best, X_test, y_test)

    return ClassificationResult(
        model_name        = "LightGBM",
        accuracy          = metrics["accuracy"],
        f1_macro          = metrics["f1_macro"],
        f1_per_class      = metrics["f1_per_class"],
        log_loss          = metrics["log_loss"],
        report            = metrics["report"],
        predictions       = metrics["predictions"],
        probabilities     = metrics["probabilities"],
        y_test            = y_test,
        X_test            = X_test,
        prediction_table  = _build_prediction_table(best, X_test, y_test),
        probability_table = _build_probability_table(best, X_test, y_test),
        model             = grid,
        label_encoder     = None,   # LGBM gestisce le label stringa internamente
    )