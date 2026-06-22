"""
models/permutation_audit.py — model-agnostic feature importance via permutation.

The idea: take a trained model, measure baseline test MAE, then for each feature
shuffle its values in the test set (breaking its relationship to the target)
and measure how much MAE worsens. Features that, when shuffled, don't hurt
performance are not contributing to predictions.

Why this over built-in importances:
  - Linear coefficients are unreliable under multicollinearity (we saw this).
  - XGBoost gain importances measure split usage, not predictive contribution
    on held-out data — a feature can be used a lot and still be useless.
  - Permutation importance measures EFFECT ON GENERALISATION directly.
    A near-zero delta means: removing this feature would lose nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from team_trend.models.linear_regression import mae as _mae

logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    feature: str
    mean_delta: float
    std_delta: float


def permutation_importance(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        metric_fn=_mae,
        n_repeats: int = 5,
        seed: int = 42,
) -> pd.DataFrame:
    """
    For each feature, shuffle its column in X_test n_repeats times and
    record the change in metric.

    Returns a DataFrame sorted by mean_delta DESCENDING:
      - Top rows: shuffling hurts a lot => important features
      - Bottom rows: shuffling doesn't matter => candidates to drop
      - Negative deltas: shuffling actually IMPROVES the metric => the feature
        is actively hurting (rare — usually means correlated with noise)
    """
    baseline = metric_fn(y_test, model.predict(X_test))
    rng = np.random.default_rng(seed)

    rows = []
    for j, feat in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_shuf = X_test.copy()
            X_shuf[:, j] = rng.permutation(X_shuf[:, j])
            shuf_metric = metric_fn(y_test, model.predict(X_shuf))
            deltas.append(shuf_metric - baseline)
        rows.append(PermutationResult(
            feature=feat,
            mean_delta=float(np.mean(deltas)),
            std_delta=float(np.std(deltas)),
        ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.sort_values("mean_delta", ascending=False).reset_index(drop=True)


def report_audit(
        audit: pd.DataFrame,
        drop_thresh: float = 0.005,
        label: str = "",
) -> list[str]:
    """
    Print a tidy report and return the list of features whose mean_delta
    is below `drop_thresh` — i.e. candidates to remove from the feature set.
    """
    print(f"\n{'=' * 78}\n  PERMUTATION IMPORTANCE — {label}\n{'=' * 78}")
    fmt = audit.copy()
    fmt["mean_delta"] = fmt["mean_delta"].round(4)
    fmt["std_delta"] = fmt["std_delta"].round(4)
    print(fmt.to_string(index=False))

    candidates = audit[audit["mean_delta"] < drop_thresh]["feature"].tolist()
    print(f"\n  Drop candidates (mean_delta < {drop_thresh}): {len(candidates)}")
    for f in candidates:
        print(f"    - {f}")
    return candidates
