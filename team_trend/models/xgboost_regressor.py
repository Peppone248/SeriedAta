"""
models/xgboost_regressor.py — XGBoost regressor wrapper.

Why a wrapper:
  - train_model.py calls .fit() / .predict() / .coefficients() on the model.
    Wrapping XGBoost in the same interface means the backtest code doesn't
    change to compare linear vs trees — only the model class swaps.
  - Trees do not need standardisation, but our pipeline standardises by
    default; the wrapper accepts standardised inputs and ignores the scaling
    (it's a no-op for trees, doesn't hurt).
  - coefficients() is replaced by feature_importances_ (gain-based), which
    plays the analogous diagnostic role.

Why XGBoost and not sklearn's GradientBoostingRegressor:
  - Faster (parallel histogram-based splits)
  - Better default regularisation
  - Native handling of missing values without pre-imputation
  - Industry standard for tabular regression at this dataset size
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class XGBoostRegressorWrapper:
    """
    Thin wrapper around xgboost.XGBRegressor that matches the interface
    of LinearRegressionScratch (fit / predict / coefficients).

    Default hyperparameters are tuned for a small tabular dataset (~3k rows):
      - shallow trees (max_depth=4) to avoid overfit
      - moderate learning rate with enough estimators
      - subsample/colsample for built-in regularisation
      - early stopping uses a validation split passed via fit(..., eval_set=...)
    """

    def __init__(
            self,
            n_estimators: int = 500,  # cap; early stop will use fewer
            max_depth: int = 3,  # was 4
            learning_rate: float = 0.03,  # was 0.05
            subsample: float = 0.6,  # was 0.8
            colsample_bytree: float = 0.6,  # was 0.8
            min_child_weight: int = 10,  # was 3
            reg_lambda: float = 2.0,  # was 1.0
            reg_alpha: float = 0.5,  # was 0.0
            early_stopping_rounds: int = 30,
            random_state: int = 42,
            n_jobs: int = -1,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        self.model_: XGBRegressor | None = None
        self.feature_names_: list[str] | None = None
        self.early_stopping_rounds = early_stopping_rounds

    # ── fitting ───────────────────────────────────────────────────────────

    def fit(
            self,
            X: np.ndarray | pd.DataFrame,
            y: np.ndarray,
            feature_names: list[str] | None = None,
    ) -> "XGBoostRegressorWrapper":
        """
        Fit the XGBoost regressor.

        Args:
            X:             (n, d) features. DataFrame or ndarray.
            y:             (n,) target.
            feature_names: if X is a numpy array, pass these for importance
                           reporting. If X is a DataFrame, the columns are used.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = (
                feature_names
                if feature_names is not None
                else [f"f{i}" for i in range(X.shape[1])]
            )

        n_val = max(50, int(0.15 * len(X)))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self.model_ = XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model_.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost fitted: %d trees used (cap=%d)",
                    self.model_.best_iteration + 1, self.params["n_estimators"])
        return self

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.predict(X)

    # ── interpretability ──────────────────────────────────────────────────

    def coefficients(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """
        Returns gain-based feature importances as (name, importance) pairs,
        sorted by importance descending.

        Replaces the role of linear-regression coefficients in the backtest
        output. Note: NOT directly comparable to coefficients in magnitude,
        but plays the same diagnostic role ('which features drive predictions').
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        importances = self.model_.feature_importances_
        pairs = list(zip(self.feature_names_, importances))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    # ── for compatibility with the linear model's FitResult slot ──────────

    @property
    def fit_result_(self):
        class _Stub:
            solver = "xgboost"
            n_iterations = None
            final_loss = None
            loss_history: list[float] = []
            converged = True

        return _Stub()
