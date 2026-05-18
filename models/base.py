"""
models/base.py — include gli output condivisi tra tutti i modelli.

ClassificationResult   → logistic_pipeline e xgboost_pipeline
RegressionResult       → linear_regression_model (via wrapper in main)

Avere lo stesso dataclass garantisce che evaluation.py possa confrontare
i modelli senza if/else o chiavi hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ClassificationResult:
    """
    Output unificato per tutti i modelli di classificazione.

    Metriche obbligatorie (sempre presenti):
        accuracy, f1_macro, f1_per_class

    Metriche opzionali (None se non calcolate):
        log_loss   — None nei modelli che non producono probabilità calibrate

    Uso tipico:
        result = run_logistic(df)       # oppure run_xgboost(df)
        leaderboard = classification_leaderboard([result_a, result_b])
    """

    model_name: str

    # ── metriche fondamentali ─────────────────────────────────────────────
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]  # {"L": ..., "D": ..., "W": ...}
    log_loss: float | None = None

    # ── report sklearn completo ───────────────────────────────────────────
    report: dict = field(default_factory=dict)

    # ── array di predizione ───────────────────────────────────────────────
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    y_test: pd.Series = field(default_factory=pd.Series)
    X_test: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── tabelle di comodo per analisi ─────────────────────────────────────
    prediction_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    probability_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── modello fittato ───────────────────────────────────────────────────
    model: Any = None  # GridSearchCV wrappato su Pipeline
    label_encoder: Any = None  # solo XGBoost (LabelEncoder D→0 L→1 W→2)


@dataclass
class RegressionResult:
    """
    Output unificato per i modelli di regressione.

    Uso tipico:
        raw = run_regression_pipeline(df, features, target)
        result = RegressionResult(model_name="Linear", **raw["test_metrics"], ...)
        leaderboard = regression_leaderboard([result])
    """

    model_name: str

    # ── metriche test ─────────────────────────────────────────────────────
    mae: float
    rmse: float
    r2: float

    # ── cross-validation ─────────────────────────────────────────────────
    cv_mae: float
    cv_rmse: float
    cv_r2: float

    # ── predizioni (include actual, predicted, residual) ──────────────────
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── modello fittato ───────────────────────────────────────────────────
    model: Any = None
