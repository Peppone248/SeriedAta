from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_pipeline():
    """
    Sklearn ML pipeline.
    Even if we only use LinearRegression now,
    this structure is future-proof.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def build_regression_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features + target, drop rows with missing values,
    and return X, y ready for sklearn.
    """
    required_cols = list(feature_cols) + [target_col]
    model_df = df[required_cols].dropna().copy()

    X = model_df[list(feature_cols)]
    y = model_df[target_col]

    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def cross_validate_model(pipeline, X, y, cv=5):
    """
    Evaluate model stability across folds.
    """
    mae_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=cv
    )

    rmse_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv
    )

    r2_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="r2",
        cv=cv
    )

    return {
        "cv_mae_mean": float(-mae_scores.mean()),
        "cv_rmse_mean": float(-rmse_scores.mean()),
        "cv_r2_mean": float(r2_scores.mean()),
    }


def split_regression_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split for regression tasks.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """
    Fit a baseline linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Compute standard regression metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return metrics

def evaluate_baseline_mean(y_train, y_test):
    """
    Baseline model: always predicts the mean of y_train.
    """
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)

    metrics = {
        "mae": mean_absolute_error(y_test, baseline_pred),
        "rmse": mean_squared_error(y_test, baseline_pred),
        "r2": r2_score(y_test, baseline_pred),
    }

    return metrics, baseline_pred

def build_coefficients_table(
    model: LinearRegression,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Return model coefficients in a readable table.
    """
    coef_df = pd.DataFrame({
        "feature": list(feature_cols),
        "coefficient": model.coef_,
        "abs_coefficient": abs(model.coef_),
    }).sort_values("abs_coefficient", ascending=False)

    return coef_df


def build_predictions_table(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compare actual vs predicted values.
    """
    y_pred = model.predict(X_test)

    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    pred_df["predicted"] = y_pred
    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]

    return pred_df


def save_regression_outputs(
    metrics: dict[str, float],
    coefficients: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str = "reports/metrics",
    prefix: str = "linear_regression",
) -> None:
    """
    Save regression results to disk.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_path / f"{prefix}_metrics.csv", index=False)
    coefficients.to_csv(out_path / f"{prefix}_coefficients.csv", index=False)
    predictions.to_csv(out_path / f"{prefix}_predictions.csv", index=False)


def run_linear_regression_baseline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = False,
    output_dir: str = "reports/metrics",
    prefix: str = "linear_regression",
) -> dict[str, object]:
    """
    End-to-end baseline regression workflow.
    """
    X, y = build_regression_dataset(df, feature_cols, target_col)

    X_train, X_test, y_train, y_test = split_regression_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_regression(model, X_test, y_test)
    coefficients = build_coefficients_table(model, feature_cols)
    predictions = build_predictions_table(model, X_test, y_test)

    outputs = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
    }

    if save:
        save_regression_outputs(
            metrics=metrics,
            coefficients=coefficients,
            predictions=predictions,
            output_dir=output_dir,
            prefix=prefix,
        )

    return outputs

def run_regression_pipeline(
    df,
    feature_cols,
    target_col,
    test_size=0.2,
    random_state=42,
):
    """
    Full ML workflow using sklearn Pipeline.
    """

    # 1. dataset
    model_df = df[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col]

    # 2. split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # 3. pipeline
    pipeline = build_pipeline()

    # 4. train
    model = train_model(pipeline, X_train, y_train)

    # 5. evaluation
    test_metrics = evaluate_model(model, X_test, y_test)

    # 6. cross-validation (on full dataset)
    cv_metrics = cross_validate_model(pipeline, X, y)

    # 7. predictions + residuals
    y_pred = model.predict(X_test)

    predictions = X_test.copy()
    predictions["actual"] = y_test.values
    predictions["predicted"] = y_pred
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    return {
        "model": model,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "predictions": predictions,
    }

def split_errors(preds: pd.DataFrame):
    preds = preds.copy()

    preds["abs_error"] = abs(preds["residual"])

    easy = preds[preds["abs_error"] <= 0.5]
    medium = preds[(preds["abs_error"] > 0.5) & (preds["abs_error"] <= 1.5)]
    hard = preds[preds["abs_error"] > 1.5]

    print("Easy predictions:", len(easy))
    print("Medium predictions:", len(medium))
    print("Hard predictions:", len(hard))

    return easy, medium, hard