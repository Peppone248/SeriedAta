"""
models/xgboost_pipeline.py — pipeline XGBoost con split temporale.

Restituisce ClassificationResult per confronto uniforme con Logistic.
Sostituisce models/classification_pipeline_xgboost.py.

La logica SHAP è in classification_model_interpretation_xgboost.py (invariata).

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
    Pubblica perché usata da model_comparison.py.
    """
    cutoff = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff], df.iloc[cutoff:]


# ─── PREPROCESSING ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Pubblica perché usata da model_comparison.py.
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
    """
    Nota sul bilanciamento delle classi:
      scale_pos_weight è ignorato da XGBoost con objective='multi:softprob'
      (funziona solo per binary:logistic). Il bilanciamento in multiclasse
      si ottiene con sample_weight nel fit(), calcolato da compute_sample_weight.
    """
    from sklearn.utils.class_weight import compute_sample_weight

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

    # sample_weight bilancia W / D / L proporzionalmente alla loro frequenza
    sample_weights = compute_sample_weight("balanced", y_train)
    grid.fit(X_train, y_train, model__sample_weight=sample_weights)

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


def find_optimal_draw_threshold(
        y_val: pd.Series,
        y_proba_val: np.ndarray,
        le: LabelEncoder,
        search_range: tuple = (0.25, 0.45),  # range più conservativo
        step: float = 0.01,
        min_f1_win: float = 0.40,  # floor su W
        min_f1_loss: float = 0.40,  # floor su L
) -> float:
    """
    Cerca il threshold che massimizza F1-macro sul validation set,
    con il vincolo che F1-W >= min_f1_win e F1-L >= min_f1_loss.
    Evita che alzare il recall sui draw distrugga le altre classi.

    Se nessun threshold rispetta i vincoli, ritorna 0.33 (argmax standard).
    """
    draw_idx = list(le.classes_).index("D")
    thresholds = np.arange(search_range[0], search_range[1], step)

    best_thresh = 0.33
    best_f1macro = 0.0

    results = []

    for t in thresholds:
        y_pred = _predict_with_draw_threshold(y_proba_val, le.classes_, draw_idx, t)

        f1_w = f1_score(y_val, y_pred, labels=["W"], average="macro", zero_division=0)
        f1_l = f1_score(y_val, y_pred, labels=["L"], average="macro", zero_division=0)
        f1_d = f1_score(y_val, y_pred, labels=["D"], average="macro", zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

        results.append((t, f1_macro, f1_l, f1_d, f1_w))

        # accetta solo se non distrugge W e L
        if f1_w < min_f1_win or f1_l < min_f1_loss:
            continue

        if f1_macro > best_f1macro:
            best_f1macro = f1_macro
            best_thresh = t

    # stampa tabella di ricerca per diagnostica
    print(f"\n  {'threshold':>10}  {'f1_macro':>8}  {'f1_L':>6}  {'f1_D':>6}  {'f1_W':>6}  {'valid':>6}")
    for t, fm, fl, fd, fw in results:
        valid = "✓" if fw >= min_f1_win and fl >= min_f1_loss else "✗"
        marker = " ←" if abs(t - best_thresh) < 0.005 else ""
        print(f"  {t:>10.2f}  {fm:>8.4f}  {fl:>6.3f}  {fd:>6.3f}  {fw:>6.3f}  {valid:>6}{marker}")

    print(f"\n  Threshold scelto: {best_thresh:.2f}  (F1-macro val={best_f1macro:.4f})")
    return best_thresh


def _predict_with_draw_threshold(
        y_proba: np.ndarray,
        classes: np.ndarray,
        draw_idx: int,
        threshold: float,
) -> np.ndarray:
    """
    Predice D se P(D) >= threshold, altrimenti argmax standard.
    """
    preds = []
    for row in y_proba:
        if row[draw_idx] >= threshold:
            preds.append("D")
        else:
            best = int(np.argmax(row))
            preds.append(classes[best])
    return np.array(preds)


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
        tune_draw_threshold: bool = True,  # ← nuovo parametro
) -> ClassificationResult:
    df = clean_data(df)
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

    # ── split val dal training (ultimi 15% del train) ─────────────────────
    val_cutoff = int(len(train_df) * 0.85)
    fit_df = train_df.iloc[:val_cutoff]
    val_df = train_df.iloc[val_cutoff:]

    X_fit = fit_df[NUM_FEATURES + CAT_FEATURES]
    y_fit = fit_df["result"]
    X_val = val_df[NUM_FEATURES + CAT_FEATURES]
    y_val = val_df["result"]
    X_test = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test = test_df["result"]

    le = LabelEncoder()
    y_fit_enc = le.fit_transform(y_fit)

    pipeline = build_model_pipeline()
    grid = train_model(X_fit, y_fit_enc, pipeline)

    draw_idx = list(le.classes_).index("D")
    y_proba = grid.predict_proba(X_test)

    # ── threshold tuning su val, applicato su test ────────────────────────
    if tune_draw_threshold:
        y_proba_val = grid.predict_proba(X_val)
        threshold = find_optimal_draw_threshold(y_val, y_proba_val, le)
        y_pred = _predict_with_draw_threshold(y_proba, le.classes_, draw_idx, threshold)
    else:
        y_pred_enc = grid.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc.astype(int))

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
