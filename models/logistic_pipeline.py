"""
models/logistic_pipeline.py — pipeline Logistic Regression.

Restituisce ClassificationResult per confronto con XGBoost.

Cambiamenti rispetto alla versione precedente:
  - Feature list importata da config.py (non più definita inline)
  - evaluate_model aggiunge f1_macro / f1_per_class (prima assenti)
  - run_classification_pipeline restituisce ClassificationResult invece di dict
  - Funzioni interne rinominate per chiarire che non fanno parte dell'API pubblica
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, log_loss,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import LOGISTIC_NUM_FEATURES, CAT_FEATURES
from models.base import ClassificationResult

# alias locale — usato da model_comparison.py che importa NUM_FEATURES da qui
NUM_FEATURES = LOGISTIC_NUM_FEATURES


# ─── DATA PREP ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date", ascending=True)
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["result"])
    return df


# ─── PREPROCESSING ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def build_model_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(max_iter=5000)),
    ])


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, pipeline) -> GridSearchCV:
    param_grid = {
        "model__C": [0.01, 0.1, 1, 5, 10, 50],
        "model__solver": ["lbfgs", "saga"],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params:      {grid.best_params_}")
    print(f"  Best CV log_loss: {grid.best_score_:.4f}")
    return grid


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def _compute_metrics(model, X_test, y_test) -> dict:
    """
    Calcola tutte le metriche in modo uniforme.
    Stessa struttura restituita da xgboost_pipeline._compute_metrics().
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = ["L", "D", "W"]

    f1_values = f1_score(y_test, y_pred, average=None, labels=classes)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_per_class": dict(zip(classes, f1_values.tolist())),
        "log_loss": float(log_loss(y_test, y_proba)),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred,
        "probabilities": y_proba,
    }


# ─── TABELLE OUTPUT ──────────────────────────────────────────────────────────

def _build_prediction_table(model, X_test, y_test) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    out = X_test.copy()
    out["actual"] = y_test.values
    out["predicted"] = y_pred
    out["correct"] = out["actual"] == out["predicted"]
    return out


def _build_probability_table(model, X_test, y_test) -> pd.DataFrame:
    probs = model.predict_proba(X_test)
    classes = model.classes_
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
    plt.title("Confusion Matrix — Logistic Regression")
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
    Pipeline completa Logistic Regression.

    Args:
        df:           DataFrame con feature già costruite (output di features.py).
        run_eda_flag: se True esegue EDA preliminare (unused per ora).

    Returns:
        ClassificationResult — stesso output restituito da xgboost_pipeline.
    """
    df = clean_data(df)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["result"]

    # Logistic usa stratified split (dati non strettamente temporali dopo OHE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = build_model_pipeline()
    grid = train_model(X_train, y_train, pipeline)
    metrics = _compute_metrics(grid, X_test, y_test)
    best = grid.best_estimator_

    return ClassificationResult(
        model_name="Logistic Regression",
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        f1_per_class=metrics["f1_per_class"],
        log_loss=metrics["log_loss"],
        report=metrics["report"],
        predictions=metrics["predictions"],
        probabilities=metrics["probabilities"],
        y_test=y_test,
        X_test=X_test,
        prediction_table=_build_prediction_table(best, X_test, y_test),
        probability_table=_build_probability_table(best, X_test, y_test),
        model=grid,
    )
