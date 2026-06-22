"""
models/xgboost_quantile.py — XGBoost quantile regressor.

Fits one XGBoost model per quantile (default: 0.1, 0.5, 0.9), giving
prediction intervals instead of point estimates. The implementation
matches the existing model interface (fit / predict / coefficients) so
the backtest code works unchanged for the median prediction, with an
additional predict_quantiles() method for the full interval output.

The pinball loss (XGBoost's "reg:quantileerror") is asymmetric: at
quantile tau, it weights underprediction by tau and overprediction by
(1 - tau). Fitting at tau = 0.1, 0.5, 0.9 simultaneously gives a
median + 80%-interval prediction.

Calibration check: ~80% of held-out actuals should fall inside the
[q10, q90] interval. Coverage far from 80% means the intervals are
miscalibrated (too narrow if coverage < 80%, too wide if > 80%).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class XGBoostQuantileRegressor:
    """
    Three XGBoost models fitted at quantiles (low, median, high).

    Args:
        quantiles: tuple of three taus; defaults to (0.1, 0.5, 0.9) for
                   an 80% interval. (0.05, 0.5, 0.95) gives a 90% interval.
        Other hyperparameters mirror the point-prediction wrapper.
    """

    def __init__(
            self,
            quantiles: tuple = (0.1, 0.5, 0.9),
            n_estimators: int = 500,
            max_depth: int = 3,
            learning_rate: float = 0.03,
            subsample: float = 0.6,
            colsample_bytree: float = 0.6,
            min_child_weight: int = 10,
            reg_lambda: float = 2.0,
            early_stopping_rounds: int = 30,
            random_state: int = 42,
            n_jobs: int = -1,
    ) -> None:
        if len(quantiles) != 3 or quantiles[1] != 0.5:
            raise ValueError("Expected three quantiles (low, 0.5, high)")
        self.quantiles = quantiles
        self.early_stopping_rounds = early_stopping_rounds
        self.base_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        self.models_: dict[float, XGBRegressor] = {}
        self.feature_names_: list[str] | None = None

    # ── fitting ───────────────────────────────────────────────────────────

    def fit(
            self,
            X: np.ndarray | pd.DataFrame,
            y: np.ndarray,
            feature_names: list[str] | None = None,
    ) -> "XGBoostQuantileRegressor":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_arr = X.to_numpy(float)
        else:
            self.feature_names_ = (
                feature_names
                if feature_names is not None
                else [f"f{i}" for i in range(X.shape[1])]
            )
            X_arr = X

        # hold out last 15% chronologically for early stopping
        n_val = max(50, int(0.15 * len(X_arr)))
        X_tr, X_val = X_arr[:-n_val], X_arr[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        for tau in self.quantiles:
            params = dict(self.base_params)
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = tau
            params["early_stopping_rounds"] = self.early_stopping_rounds

            m = XGBRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            self.models_[tau] = m
            logger.info("Quantile %.2f fitted: %d trees used",
                        tau, m.best_iteration + 1)

        return self

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Default prediction = median (quantile 0.5)."""
        return self.models_[0.5].predict(X)

    def predict_quantiles(
            self,
            X: np.ndarray | pd.DataFrame,
    ) -> dict[float, np.ndarray]:
        """Return predictions at every fitted quantile."""
        return {tau: m.predict(X) for tau, m in self.models_.items()}

    # ── interpretability ──────────────────────────────────────────────────

    def coefficients(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Importances from the median model — analogous to the point regressor."""
        importances = self.models_[0.5].feature_importances_
        pairs = list(zip(self.feature_names_, importances))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs


# ─── metrics ───────────────────────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Quantile (pinball) loss at level tau.

    For each sample:
      err = y_true - y_pred
      loss = max(tau * err, (tau - 1) * err)
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(tau * err, (tau - 1) * err)))


def coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Fraction of actuals inside [lower, upper]. Should match the nominal interval."""
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


def sharpness(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean width of the interval. Narrower is better, but only if coverage is correct."""
    return float(np.mean(y_upper - y_lower))


def quantile_diagnostics(
        y_true: np.ndarray,
        quantile_pred: dict[float, np.ndarray],
) -> dict:
    """Standard battery of quantile-regression diagnostics."""
    taus = sorted(quantile_pred.keys())
    if len(taus) != 3:
        raise ValueError("Expected exactly 3 quantiles")
    low_tau, mid_tau, hi_tau = taus
    low = quantile_pred[low_tau]
    mid = quantile_pred[mid_tau]
    hi = quantile_pred[hi_tau]

    nominal_coverage = hi_tau - low_tau
    return {
        "nominal_coverage": nominal_coverage,
        "actual_coverage": coverage(y_true, low, hi),
        "sharpness": sharpness(low, hi),
        "pinball_low": pinball_loss(y_true, low, low_tau),
        "pinball_median": pinball_loss(y_true, mid, mid_tau),
        "pinball_high": pinball_loss(y_true, hi, hi_tau),
    }
