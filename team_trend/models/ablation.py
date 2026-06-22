"""
models/ablation.py — group-level feature ablation.

For each feature group, retrain the production model WITHOUT that group
and measure the change in test MAE vs the baseline (all features).

Interpretation of the delta column:
  positive  -> removing the group HURT the model -> the group carries signal
  negative  -> removing the group HELPED the model -> the group was noise/harmful
  ~0        -> the group is redundant (its information is captured by others)

Group ablation is stronger than per-feature permutation importance because
it catches the case where individual features look unimportant but the
group as a whole carries signal through interactions. A single squad
quality feature may be noise, while the group of squad quality features
together may meaningfully add information.

This is the test that tells us which CONCEPTUAL investments (player data,
opponent data, volatility) paid off — and which we can drop entirely.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _evaluate_walk_forward(
    model_factory,
    gold:     pd.DataFrame,
    target:   str,
    features: list[str],
    seasons:  list[str],
) -> tuple[float, float]:
    """Walk-forward MAE for a given feature set. Returns (mean, std)."""
    folds = [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]
    maes = []
    for train_seasons, test_season in folds:
        train_df = gold[gold["season"].isin(train_seasons)].dropna(subset=features + [target])
        test_df  = gold[gold["season"] == test_season].dropna(subset=features + [target])
        X_tr, y_tr = train_df[features].to_numpy(float), train_df[target].to_numpy(float)
        X_te, y_te = test_df [features].to_numpy(float), test_df [target].to_numpy(float)
        m = model_factory()
        if "feature_names" in m.fit.__code__.co_varnames:
            m.fit(X_tr, y_tr, feature_names=features)
        else:
            m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        maes.append(float(np.mean(np.abs(y_te - pred))))
    return float(np.mean(maes)), float(np.std(maes))


def run_group_ablation(
    gold:           pd.DataFrame,
    target:         str,
    feature_groups: dict[str, list[str]],
    model_factory,
    seasons:        list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Run the ablation: baseline + one run per group with that group removed.
    Returns (results_dataframe, baseline_mae).
    """
    seasons = seasons or sorted(gold["season"].unique())

    # union of all features in the groups, restricted to those actually in gold
    all_features = sorted({f for group in feature_groups.values() for f in group})
    all_features = [f for f in all_features if f in gold.columns]

    if not all_features:
        raise ValueError("No usable features after filtering against gold columns.")

    logger.info("Running ablation: %d groups, %d total features",
                len(feature_groups), len(all_features))

    baseline_mae, baseline_std = _evaluate_walk_forward(
        model_factory, gold, target, all_features, seasons,
    )
    logger.info("Baseline (all features): MAE = %.3f ± %.3f",
                baseline_mae, baseline_std)

    results = []
    for group_name, group_features in feature_groups.items():
        group_present = [f for f in group_features if f in gold.columns]
        if not group_present:
            logger.info("Skipping %s (no features present in gold)", group_name)
            continue

        reduced = [f for f in all_features if f not in group_present]
        if not reduced:
            logger.info("Skipping %s (would leave zero features)", group_name)
            continue

        mae, std = _evaluate_walk_forward(
            model_factory, gold, target, reduced, seasons,
        )
        delta = mae - baseline_mae
        results.append({
            "group_removed": group_name,
            "n_removed":     len(group_present),
            "n_kept":        len(reduced),
            "test_mae":      round(mae, 3),
            "test_mae_std":  round(std, 3),
            "delta":         round(delta, 3),
        })
        logger.info("  -%s (%d feats): MAE=%.3f  delta=%+.3f",
                    group_name, len(group_present), mae, delta)

    df = pd.DataFrame(results).sort_values("delta", ascending=False).reset_index(drop=True)
    return df, baseline_mae


def print_ablation_report(audit_df: pd.DataFrame, baseline_mae: float) -> None:
    print("\n" + "=" * 78)
    print(f"{'FEATURE GROUP ABLATION  (baseline MAE=' + f'{baseline_mae:.3f}' + ')':^78}")
    print("=" * 78)
    print(audit_df.to_string(index=False))
    print()

    useful  = audit_df[audit_df["delta"] >  0.005]
    neutral = audit_df[audit_df["delta"].abs() <= 0.005]
    harmful = audit_df[audit_df["delta"] < -0.005]

    if not useful.empty:
        print(f"  USEFUL groups (removing hurts): {len(useful)}")
        for _, r in useful.iterrows():
            print(f"    {r['group_removed']:18s}  delta = +{r['delta']:.3f}")
    if not neutral.empty:
        print(f"  NEUTRAL groups (no difference): {len(neutral)}")
        for _, r in neutral.iterrows():
            print(f"    {r['group_removed']}")
    if not harmful.empty:
        print(f"  HARMFUL groups (removing helps): {len(harmful)}")
        for _, r in harmful.iterrows():
            print(f"    {r['group_removed']:18s}  delta = {r['delta']:+.3f}")