from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


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
    test_size: float = 0.3,
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