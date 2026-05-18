"""
models/xgboost_pipeline.py — pipeline XGBoost con split temporale.

Restituisce ClassificationResult per confronto con Logistic.

Cambiamenti rispetto alla versione precedente:
  - Feature list importata da config.py (non più definita inline)
  - _compute_metrics() aggiunge log_loss (calcolato su label stringa via le.classes_)
  - run_classification_pipeline restituisce ClassificationResult invece di dict
  - Funzioni backwards-compat (build_prediction_table, build_probability_table)
    rimosse: le tabelle sono ora costruite inline e accessibili via result.prediction_table
  - temporal_train_test_split e build_preprocessor rimangono pubblici
    perché usati da model_comparison.py
"""

from __future__ import annotations

import numpy as np
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import XGBOOST_NUM_FEATURES, CAT_FEATURES
from models.base import ClassificationResult

NUM_FEATURES = XGBOOST_NUM_FEATURES


# ─── DATA PREP ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date", ascending=True)
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["result"])
    return df


def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split temporale: le ultime `test_ratio` righe (per data) vanno nel test.
    Evita data leakage rispetto a train_test_split con stratify.
    """
    cutoff = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff], df.iloc[cutoff:]


# ─── PREPROCESSING ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Usata da model_comparison.py.
    """
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def build_model_pipeline() -> Pipeline:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        base_score=0.5,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, pipeline) -> GridSearchCV:
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params:      {grid.best_params_}")
    print(f"  Best CV f1_macro: {grid.best_score_:.4f}")
    return grid


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def _compute_metrics(y_test, y_pred, y_proba, le: LabelEncoder) -> dict:
    """
    Metriche unificate — stessa struttura di logistic_pipeline._compute_metrics().

    log_loss: label order = le.classes_ (D, L, W alfabetico),
              coincide con le colonne di y_proba → calcolo corretto.
    """
    classes = ["L", "D", "W"]
    f1_values = f1_score(y_test, y_pred, average=None, labels=classes)
    ll = log_loss(y_test, y_proba, labels=list(le.classes_))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_per_class": dict(zip(classes, f1_values.tolist())),
        "log_loss": float(ll),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred,
        "probabilities": y_proba,
    }


# ─── TABELLE OUTPUT ──────────────────────────────────────────────────────────

def _build_prediction_table(X_test, y_test, y_pred) -> pd.DataFrame:
    out = X_test.copy()
    out["actual"] = y_test.values
    out["predicted"] = y_pred
    out["correct"] = out["actual"] == out["predicted"]
    return out


def _build_probability_table(X_test, y_test, y_proba, le: LabelEncoder) -> pd.DataFrame:
    out = X_test.copy()
    out["actual"] = y_test.values
    for i, cls in enumerate(le.classes_):  # D, L, W (ordine LabelEncoder)
        out[f"P_{cls}"] = y_proba[:, i]
    out["confidence"] = y_proba.max(axis=1)
    return out


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=["L", "D", "W"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["L", "D", "W"], yticklabels=["L", "D", "W"])
    plt.title("Confusion Matrix — XGBoost")
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
    Pipeline completa XGBoost con split temporale.

    Args:
        df:           DataFrame con feature già costruite (output di features.py).
        run_eda_flag: non usato, mantenuto per compatibilità firma con logistic.

    Returns:
        ClassificationResult — stesso contratto restituito da logistic_pipeline.
    """
    df = clean_data(df)

    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df["result"]
    X_test = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test = test_df["result"]

    print(f"  Train: {len(X_train)} righe | Test: {len(X_test)} righe")
    print(f"  Distribuzione train: {y_train.value_counts().to_dict()}")

    # XGBoost richiede label numeriche — D→0 L→1 W→2 (fit solo su train)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    pipeline = build_model_pipeline()
    grid = train_model(X_train, y_train_enc, pipeline)

    y_pred_enc = grid.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc.astype(int))
    y_proba = grid.predict_proba(X_test)

    metrics = _compute_metrics(y_test, y_pred, y_proba, le)

    return ClassificationResult(
        model_name="XGBoost",
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        f1_per_class=metrics["f1_per_class"],
        log_loss=metrics["log_loss"],
        report=metrics["report"],
        predictions=metrics["predictions"],
        probabilities=metrics["probabilities"],
        y_test=y_test,
        X_test=X_test,
        prediction_table=_build_prediction_table(X_test, y_test, y_pred),
        probability_table=_build_probability_table(X_test, y_test, y_proba, le),
        model=grid,
        label_encoder=le,
    )
