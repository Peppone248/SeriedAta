"""
models/cascaded_pipeline.py — classificatore a cascata per W / D / L.

Due classificatori binari XGBoost in sequenza:

  Step 1 — W vs {D, L}
    Segnale forte: xG, strength_diff, cum_avg_points.
    scale_pos_weight calibrato su n_nonW / n_W.

  Step 2 — D vs L  (addestrato sui soli campioni non-W del training)
    Segnale utile: parity features, h2h_draw_rate, both_defensive.
    scale_pos_weight calibrato su n_L / n_D.

Assemblaggio delle probabilità finali (somma = 1 per costruzione):
    P(W)  = clf_1.predict_proba[:, 1]
    P(D)  = (1 - P(W)) * clf_2.predict_proba[:, 1]
    P(L)  = (1 - P(W)) * (1 - clf_2.predict_proba[:, 1])

Predizione finale: argmax sulle tre probabilità assemblate.

Restituisce ClassificationResult — stesso contratto degli altri modelli.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from matches_classification.config import XGBOOST_NUM_FEATURES, CAT_FEATURES
from matches_classification.models.base import ClassificationResult

NUM_FEATURES = XGBOOST_NUM_FEATURES


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


def _build_binary_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    Pipeline binaria XGBoost.
    scale_pos_weight funziona correttamente con binary:logistic
    (a differenza di multi:softprob dove era ignorato).
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


# ─── TRAINING ────────────────────────────────────────────────────────────────

def _train_binary(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_pos_weight: float,
        label: str,
) -> GridSearchCV:
    """
    Grid search per un classificatore binario.
    scale_pos_weight viene incluso nel param_grid per permettere
    al grid di scegliere se usarlo o no.
    """
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__scale_pos_weight": [1.0, round(scale_pos_weight, 2)],
    }

    pipeline = _build_binary_pipeline()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1",  # f1 binario sulla classe positiva
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"  [{label}] best params:  {grid.best_params_}")
    print(f"  [{label}] best CV f1:   {grid.best_score_:.4f}")
    return grid


# ─── PROBABILITY ASSEMBLY ────────────────────────────────────────────────────

def _assemble_probabilities(
        p_win: np.ndarray,
        p_draw_given_notwin: np.ndarray,
) -> np.ndarray:
    """
    Assembla le probabilità finali da due classificatori binari.
    La somma è 1 per costruzione.

    Returns:
        Array (n_samples, 3) con colonne [P(L), P(D), P(W)]
        nell'ordine LabelEncoder-compatibile con le.classes_ = [D, L, W].
        → colonna 0 = P(D), colonna 1 = P(L), colonna 2 = P(W)

    Nota: l'ordine rispetta le.classes_ = ["D", "L", "W"] (alfabetico)
    per coerenza con xgboost_pipeline e _compute_metrics.
    """
    p_notwin = 1.0 - p_win
    p_draw = p_notwin * p_draw_given_notwin
    p_loss = p_notwin * (1.0 - p_draw_given_notwin)

    # stack in ordine [D, L, W] — alfabetico come le.classes_
    return np.column_stack([p_draw, p_loss, p_win])


def _predict_from_proba(
        proba: np.ndarray,
        class_order: list[str] = ["D", "L", "W"],
) -> np.ndarray:
    """Predice la classe con probabilità massima."""
    idx = np.argmax(proba, axis=1)
    labels = np.array(class_order)
    return labels[idx]


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def _compute_metrics(
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
) -> dict:
    """
    Stessa struttura di xgboost_pipeline._compute_metrics().
    y_proba ha colonne [D, L, W] — log_loss usa labels coerenti.
    """
    classes = ["L", "D", "W"]
    f1_values = f1_score(
        y_test, y_pred, average=None, labels=classes, zero_division=0
    )
    ll = log_loss(y_test, y_proba, labels=["D", "L", "W"])

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
        "f1_per_class": dict(zip(classes, f1_values.tolist())),
        "log_loss": float(ll),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred,
        "probabilities": y_proba,
    }


# ─── OUTPUT TABLES ───────────────────────────────────────────────────────────

def _build_prediction_table(X_test, y_test, y_pred) -> pd.DataFrame:
    out = X_test.copy()
    out["actual"] = y_test.values
    out["predicted"] = y_pred
    out["correct"] = out["actual"] == out["predicted"]
    return out


def _build_probability_table(X_test, y_test, y_proba) -> pd.DataFrame:
    out = X_test.copy()
    out["actual"] = y_test.values
    out["P_D"] = y_proba[:, 0]
    out["P_L"] = y_proba[:, 1]
    out["P_W"] = y_proba[:, 2]
    out["confidence"] = y_proba.max(axis=1)
    return out


# ─── DIAGNOSTICA ─────────────────────────────────────────────────────────────

def plot_cascade_probabilities(y_proba: np.ndarray, y_test: pd.Series) -> None:
    """
    Distribuzioni di P(W), P(D), P(L) per classe reale.
    Permette di verificare che il modello separi bene le distribuzioni.
    """
    df_plot = pd.DataFrame({
        "P_W": y_proba[:, 2],
        "P_D": y_proba[:, 0],
        "P_L": y_proba[:, 1],
        "label": y_test.values,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    proba_cols = [("P_W", "W", "#5ab27a"),
                  ("P_D", "D", "#d48a2b"),
                  ("P_L", "L", "#d84a30")]

    for ax, (col, label, color) in zip(axes, proba_cols):
        for true_class in ["W", "D", "L"]:
            subset = df_plot[df_plot["label"] == true_class][col]
            subset.plot.kde(ax=ax, label=f"reale={true_class}", alpha=0.7)
        ax.set_title(f"Distribuzione {col} per classe reale")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        ax.axvline(0.33, color="gray", linestyle="--", linewidth=0.8)

    plt.suptitle("Cascaded Classifier — separazione probabilità per classe",
                 fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=["L", "D", "W"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["L", "D", "W"], yticklabels=["L", "D", "W"])
    plt.title("Confusion Matrix — Cascaded Classifier")
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
    Pipeline a cascata: W vs non-W → D vs L.

    Args:
        df: DataFrame con feature già costruite (output di features.py).

    Returns:
        ClassificationResult — stesso contratto degli altri modelli.
    """
    df = clean_data(df)
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

    all_features = NUM_FEATURES + CAT_FEATURES

    X_train = train_df[all_features]
    y_train = train_df["result"]
    X_test = test_df[all_features]
    y_test = test_df["result"]

    print(f"  Train: {len(X_train)} righe | Test: {len(X_test)} righe")
    print(f"  Distribuzione train: {y_train.value_counts().to_dict()}")

    # ── step 1: W vs {D, L} ──────────────────────────────────────────────
    print("\n  ── Step 1: W vs non-W ──")

    y_train_1 = (y_train == "W").astype(int)

    n_W = y_train_1.sum()
    n_nonW = len(y_train_1) - n_W
    spw_1 = n_nonW / n_W  # scala i non-W per bilanciare

    clf_1 = _train_binary(X_train, y_train_1, spw_1, label="W vs non-W")
    p_win = clf_1.predict_proba(X_test)[:, 1]

    # ── step 2: D vs L (solo sui non-W del training) ─────────────────────
    print("\n  ── Step 2: D vs L (su campioni non-W) ──")

    mask_nonW = y_train != "W"
    X_train_2 = X_train[mask_nonW]
    y_train_2 = (y_train[mask_nonW] == "D").astype(int)

    n_D = y_train_2.sum()
    n_L = len(y_train_2) - n_D
    spw_2 = n_L / n_D  # scala D (minoritaria tra non-W)

    # clf_2 applicato a TUTTI i test sample per ottenere P(D|non-W)
    clf_2 = _train_binary(X_train_2, y_train_2, spw_2, label="D vs L")
    p_draw_given_notwin = clf_2.predict_proba(X_test)[:, 1]

    # ── assemblaggio probabilità finali ───────────────────────────────────
    y_proba = _assemble_probabilities(p_win, p_draw_given_notwin)
    y_pred = _predict_from_proba(y_proba, class_order=["D", "L", "W"])

    metrics = _compute_metrics(y_test, y_pred, y_proba)

    print(f"\n  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    fpc = metrics["f1_per_class"]
    print(f"  F1 [L={fpc['L']:.3f}  D={fpc['D']:.3f}  W={fpc['W']:.3f}]")

    return ClassificationResult(
        model_name="Cascaded (W|DL → D|L)",
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
        probability_table=_build_probability_table(X_test, y_test, y_proba),
        model={"clf_1": clf_1, "clf_2": clf_2},  # entrambi accessibili
        label_encoder=None,
    )
