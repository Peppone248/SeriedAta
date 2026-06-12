"""
models/linear_regression.py — linear regression implemented from scratch.

Two solvers:

  1. Normal equations (closed form):
        w = (X^T X + lambda*I)^-1 X^T y
     Exact, one shot, O(d^3) in the number of features d. With our ~30
     features this is trivial. lambda > 0 gives Ridge regularisation and
     also guarantees the matrix is invertible (collinear features would
     otherwise make X^T X singular).

  2. Batch gradient descent:
        w <- w - lr * dL/dw,   L = (1/2n)||Xw - y||^2 + (lambda/2n)||w||^2
     Iterative. Included for learning purposes: it is what scales when
     d or n become too large for the closed form, and it makes the
     mechanics of "fitting" visible (loss curve, convergence).

Both share the same preprocessing, implemented here as well:
  - Standardisation (z-score) of features, fitted on TRAIN ONLY and applied
    to test — fitting the scaler on all data would leak test statistics.
  - Intercept handled by appending a column of ones (not standardised).

The model implements fit/predict and stores training history so the
caller can plot the loss curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ─── preprocessing ─────────────────────────────────────────────────────────────

class Standardizer:
    """
    Z-score standardisation: x' = (x - mean) / std.

    fit() learns mean/std from the training split only.
    transform() applies them to any split.
    Columns with zero variance get std=1 to avoid division by zero
    (they become constant 0 after centering — harmless to the model).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Append a column of ones so the bias is learned as a normal weight."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


# ─── metrics ───────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination: 1 - SS_res / SS_tot.
    Compares the model to the 'always predict the mean' baseline:
    r2 = 0 means no better than the mean; r2 < 0 means worse.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ─── model ─────────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """Training diagnostics returned by fit()."""
    solver: str
    n_iterations: int = 0
    final_loss: float = float("nan")
    loss_history: list[float] = field(default_factory=list)
    converged: bool = True


class LinearRegressionScratch:
    """
    Linear regression with optional L2 (Ridge) regularisation.

    Args:
        solver:     'normal' (closed form) or 'gd' (gradient descent)
        l2:         regularisation strength lambda (0 = ordinary least squares)
        lr:         learning rate (gd only)
        max_iter:   iteration cap (gd only)
        tol:        early-stop threshold on relative loss improvement (gd only)

    Notes:
        - The intercept column is excluded from regularisation: penalising
          the bias would pull predictions toward 0 for no good reason.
        - Features should be standardised before fitting, especially for gd
          (unscaled features give ill-conditioned loss surfaces) and for
          ridge (penalty must act on comparable scales).
    """

    def __init__(
            self,
            solver: str = "normal",
            l2: float = 0.0,
            lr: float = 0.01,
            max_iter: int = 5000,
            tol: float = 1e-8,
    ) -> None:
        if solver not in ("normal", "gd"):
            raise ValueError("solver must be 'normal' or 'gd'")
        self.solver = solver
        self.l2 = l2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        self.weights_: np.ndarray | None = None
        self.fit_result_: FitResult | None = None

    # ── fitting ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """
        X: (n, d) feature matrix WITHOUT intercept column (added internally).
        y: (n,) target vector.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xb = add_intercept(X)  # (n, d+1)

        if self.solver == "normal":
            self._fit_normal(Xb, y)
        else:
            self._fit_gd(Xb, y)
        return self

    def _regularization_matrix(self, d_plus_1: int) -> np.ndarray:
        """lambda * I, with 0 in the intercept position (last column)."""
        reg = self.l2 * np.eye(d_plus_1)
        reg[-1, -1] = 0.0
        return reg

    def _fit_normal(self, Xb: np.ndarray, y: np.ndarray) -> None:
        """
        Closed form: w = (X^T X + reg)^-1 X^T y
        np.linalg.solve is used instead of explicit inverse: it is more
        numerically stable and ~3x faster (solves the linear system directly).
        """
        d1 = Xb.shape[1]
        A = Xb.T @ Xb + self._regularization_matrix(d1)
        b = Xb.T @ y
        self.weights_ = np.linalg.solve(A, b)

        loss = self._loss(Xb, y, self.weights_)
        self.fit_result_ = FitResult(solver="normal", final_loss=loss)
        logger.info("normal equations: loss=%.5f", loss)

    def _fit_gd(self, Xb: np.ndarray, y: np.ndarray) -> None:
        """
        Batch gradient descent on the ridge MSE loss.

        Gradient: dL/dw = (1/n) X^T (Xw - y) + (lambda/n) w   [bias unpenalised]
        Early stop when relative loss improvement < tol.
        """
        n, d1 = Xb.shape
        w = np.zeros(d1)
        history: list[float] = []
        prev_loss = np.inf
        converged = False

        reg_mask = np.ones(d1)
        reg_mask[-1] = 0.0  # don't regularise the intercept

        for it in range(self.max_iter):
            residual = Xb @ w - y  # (n,)
            grad = (Xb.T @ residual) / n  # (d+1,)
            grad += (self.l2 / n) * (w * reg_mask)
            w -= self.lr * grad

            loss = self._loss(Xb, y, w)
            history.append(loss)

            if prev_loss - loss < self.tol * max(prev_loss, 1e-12):
                converged = True
                break
            prev_loss = loss

        self.weights_ = w
        self.fit_result_ = FitResult(
            solver="gd", n_iterations=it + 1, final_loss=history[-1],
            loss_history=history, converged=converged,
        )
        logger.info("gd: %d iters, loss=%.5f, converged=%s",
                    it + 1, history[-1], converged)

    def _loss(self, Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Ridge MSE: (1/2n)||Xw - y||^2 + (lambda/2n)||w_no_bias||^2"""
        n = Xb.shape[0]
        residual = Xb @ w - y
        data_term = (residual @ residual) / (2 * n)
        reg_term = self.l2 * (w[:-1] @ w[:-1]) / (2 * n)
        return float(data_term + reg_term)

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model not fitted.")
        return add_intercept(np.asarray(X, dtype=float)) @ self.weights_

    def coefficients(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """(name, weight) pairs sorted by |weight|, intercept last."""
        if self.weights_ is None:
            raise RuntimeError("Model not fitted.")
        pairs = list(zip(feature_names, self.weights_[:-1]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        pairs.append(("(intercept)", float(self.weights_[-1])))
        return pairs
