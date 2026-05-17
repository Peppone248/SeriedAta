import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


NUM_FEATURES = [
    # --- feature originali ---
    "xg",
    "xga",
    "poss",
    "sot",
    "shot_accuracy",
    "is_home",
    "strength_points_diff",
    "strength_xg_diff",
    "strength_xga_diff",
    #"finishing_efficiency",
    #"defensive_efficiency",
    "last_5_points",
    "last_5_goal_diff",
    "last_5_xg",
    "xg_trend",
    "points_trend",
    "days_rest",
    "dist",                # distanza media tiro
    "formation_changed",   # cambio modulo rispetto alla gara precedente
    "cum_avg_points",
    "cum_avg_xg",
    "cum_avg_xga",
    "h2h_win_rate",
    #"matches_last_14d",
    "weighted_form",
    "form_consistency"
]

CAT_FEATURES = [
    "team",
    "opponent",
    "venue",
]


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by="date", ascending=True)  # fix: era "data"
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["result"])
    return df


# ─────────────────────────────────────────────
# PREPROCESSOR & PIPELINE
# ─────────────────────────────────────────────

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )


def build_model_pipeline():
    preprocessor = build_preprocessor()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        base_score=0.5,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ─────────────────────────────────────────────
# TRAINING con TimeSeriesSplit coerente
# ─────────────────────────────────────────────

def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split temporale: le ultime `test_ratio` righe (per data) vanno nel test.
    Evita il data leakage che si aveva con train_test_split + stratify.
    """
    cutoff = int(len(df) * (1 - test_ratio))
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]
    return train, test


def train_model(X_train, y_train, pipeline):
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring="f1_macro",   # obiettivo: predire la classe corretta W/D/L
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    return grid


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model_decoded(y_test, y_pred, y_proba):
    """Valuta il modello con label stringa (W/D/L) già decodificate."""
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_per_class": f1_score(y_test, y_pred, average=None, labels=["L", "D", "W"]).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"\nAccuracy:  {results['accuracy']:.4f}")
    print(f"F1 macro:  {results['f1_macro']:.4f}")
    print(f"F1 [L,D,W]: {[round(f, 3) for f in results['f1_per_class']]}")

    return {
        "metrics": results,
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_test": y_test,
    }


def evaluate_model(model, X_test, y_test):
    """Kept for backwards compatibility — richiede y_test con label numeriche."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluate_model_decoded(y_test, y_pred, y_proba)


# ─────────────────────────────────────────────
# PIPELINE COMPLETA
# ─────────────────────────────────────────────

def run_classification_pipeline(df: pd.DataFrame, run_eda_flag: bool = False):
    # feature engineering aggiuntivo
    df = clean_data(df)

    # split temporale coerente
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df["result"]
    X_test  = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test  = test_df["result"]

    print(f"Train: {len(X_train)} righe | Test: {len(X_test)} righe")
    print(f"Distribuzione train:\n{y_train.value_counts()}")
    print(f"Distribuzione test:\n{y_test.value_counts()}")

    # XGBoost richiede label numeriche — encoding D→0, L→1, W→2
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # fit solo su train
    y_test_enc  = le.transform(y_test)

    pipeline = build_model_pipeline()
    grid = train_model(X_train, y_train_enc, pipeline)

    # decodifica predizioni numeriche → W/D/L
    y_pred_enc = grid.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc.astype(int))

    y_proba = grid.predict_proba(X_test)

    eval_results = evaluate_model_decoded(y_test, y_pred, y_proba)

    prediction_table = build_prediction_table_decoded(X_test, y_test, y_pred)
    probability_table = build_probability_table_decoded(X_test, y_test, y_proba, le)

    return {
        "model": grid,
        "label_encoder": le,
        "metrics": eval_results["metrics"],
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_test": y_test,
        "X_test": X_test,
        "prediction_table": prediction_table,
        "probability_table": probability_table,
    }


# ─────────────────────────────────────────────
# OUTPUT TABLES
# ─────────────────────────────────────────────

def build_prediction_table_decoded(X_test, y_test, y_pred):
    df_out = X_test.copy()
    df_out["actual"] = y_test.values
    df_out["predicted"] = y_pred
    df_out["correct"] = df_out["actual"] == df_out["predicted"]
    return df_out


def build_probability_table_decoded(X_test, y_test, y_proba, le: LabelEncoder):
    df_out = X_test.copy()
    df_out["actual"] = y_test.values
    for i, cls in enumerate(le.classes_):      # D, L, W nell'ordine del LabelEncoder
        df_out[f"P_{cls}"] = y_proba[:, i]
    df_out["confidence"] = y_proba.max(axis=1)
    return df_out


def build_prediction_table(model, X_test, y_test):
    """Backwards compatibility — non usata nel nuovo pipeline."""
    y_pred = model.predict(X_test)
    return build_prediction_table_decoded(X_test, y_test, y_pred)


def build_probability_table(model, X_test, y_test):
    """Backwards compatibility — non usata nel nuovo pipeline."""
    probs = model.predict_proba(X_test)
    classes = model.classes_
    df_out = X_test.copy()
    df_out["actual"] = y_test.values
    for i, cls in enumerate(classes):
        df_out[f"P_{cls}"] = probs[:, i]
    df_out["confidence"] = probs.max(axis=1)
    return df_out


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=["L", "D", "W"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["L", "D", "W"],
        yticklabels=["L", "D", "W"],
    )
    plt.title("Confusion Matrix — Match Outcome Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()