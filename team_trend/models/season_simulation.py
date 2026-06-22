"""
models/season_simulation.py — Monte Carlo end-of-season standings simulation.

For each team in a season:
  1. Identify the latest pivot row (max matchweek with valid quantile prediction).
     In a complete Serie A season this is MW33: the row whose target column
     IS the sum of points in MW34-38, the actual last 5 matches.
  2. Sample N times from the quantile-derived predictive distribution.
  3. Add starting_points (cum_points before pivot + actual pivot-match points)
     to each sample -> distribution of FINAL season points per team.
  4. Within each Monte Carlo iteration, rank all teams by final points.
  5. Aggregate the ranks -> probability distribution over final positions.

Why we approximate the predictive distribution as Normal(q_median, sigma):
  - We have only three quantile point estimates per row, not a full posterior
  - Normal is the maximum-entropy distribution given mean + variance
  - sigma chosen so the q_high-q_low interval matches the Normal 80% interval
  - In practice this is the standard approach when quantile estimates are
    the only available uncertainty information

Validation built in: the actual target value at the pivot row IS the true
sum of the next-5 points, so we compare predicted final standings to actual.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# z-distance for an 80% interval of a standard normal:
# norm.ppf(0.9) - norm.ppf(0.1) ≈ 1.2816 - (-1.2816) ≈ 2.5631
NORMAL_80_Z = 2.5631

# bounds for 5-match-window points: in football 5 wins = 15, 5 losses = 0
POINTS_MIN, POINTS_MAX = 0, 15


# ─── sampling helpers ──────────────────────────────────────────────────────────

def _quantiles_to_normal(
    q_low:  np.ndarray,
    q_med:  np.ndarray,
    q_high: np.ndarray,
    z:      float = NORMAL_80_Z,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Normal(mu, sigma) per team from three quantile estimates."""
    mu    = q_med
    sigma = np.maximum((q_high - q_low) / z, 0.1)   # guard zero-variance edge
    return mu, sigma


def _ranks_from_points(final_points: np.ndarray) -> np.ndarray:
    """
    For each row (one Monte Carlo iteration), compute rank of each team
    (1 = best). Higher final_points -> lower rank number.
    """
    sort_idx = np.argsort(-final_points, axis=1)         # desc per row
    ranks    = np.empty_like(sort_idx, dtype=int)
    np.put_along_axis(
        ranks, sort_idx,
        np.tile(np.arange(1, final_points.shape[1] + 1),
                (final_points.shape[0], 1)),
        axis=1,
    )
    return ranks


# ─── main simulator ────────────────────────────────────────────────────────────

def simulate_end_of_season(
    gold:                  pd.DataFrame,
    quantile_predictions:  pd.DataFrame,
    season:                str,
    target_col:            str = "next_5_matchweek_points",
    n_simulations:         int = 10_000,
    seed:                  int = 42,
) -> dict:
    """
    Args:
        gold:                 full Gold DataFrame
        quantile_predictions: DataFrame with [team, matchweek, q_low, q_median, q_high]
                              produced by the quantile model on this season
        season:               which season to simulate
        target_col:           gold's actual-next-5 target (used for validation)
        n_simulations:        Monte Carlo iterations
        seed:                 reproducibility

    Returns dict with keys:
        team_stats:        per-team forecast + actual summary table
        final_points:      (n_sim, n_teams) sampled final points
        ranks:             (n_sim, n_teams) corresponding ranks
        teams:             list of team names (column order)
        position_probs:    (n_teams, n_teams) P(team finishes at position)
        actual_finals:     (n_teams,) actual end-of-season points
        actual_ranks:      (n_teams,) actual end-of-season positions
        season:            the simulated season label
    """
    rng = np.random.default_rng(seed)

    s = gold[gold["season"] == season].copy()
    teams = sorted(s["team"].unique())

    # latest pivot per team: maximum matchweek for which we have BOTH a gold row
    # AND a quantile prediction (predictions may be sparser than gold if features
    # were NaN in some rows and got dropped during model evaluation)
    pred_by_team = quantile_predictions.groupby("team")["matchweek"].max().to_dict()

    pivot_rows = {}
    teams_with_predictions = []
    for t in teams:
        if t not in pred_by_team:
            logger.warning("Team %s has no predictions — skipping in simulation", t)
            continue
        max_mw = pred_by_team[t]
        match = s[(s["team"] == t) & (s["matchweek"] == max_mw)]
        if match.empty:
            logger.warning("Team %s: pivot row at MW%d not found in gold", t, max_mw)
            continue
        pivot_rows[t] = match.iloc[0]
        teams_with_predictions.append(t)

    teams = teams_with_predictions
    if not teams:
        raise ValueError("No teams with valid predictions for simulation.")

    # starting points: cum_points entering pivot + actual pivot-match result
    starting_points = np.array([
        float(pivot_rows[t]["cum_points"]) + float(pivot_rows[t]["points"])
        for t in teams
    ])

    # join quantile predictions to the chosen pivot rows
    pivot_keys = pd.DataFrame({
        "team":      teams,
        "matchweek": [pivot_rows[t]["matchweek"] for t in teams],
    })
    qp = pivot_keys.merge(
        quantile_predictions, on=["team", "matchweek"], how="left",
    )

    if qp[["q_low", "q_median", "q_high"]].isna().any().any():
        missing = qp[qp["q_low"].isna()]["team"].tolist()
        raise ValueError(f"Missing quantile predictions for teams: {missing}")

    q_low   = qp["q_low"].to_numpy(float)
    q_med   = qp["q_median"].to_numpy(float)
    q_high  = qp["q_high"].to_numpy(float)
    mu, sigma = _quantiles_to_normal(q_low, q_med, q_high)

    # Monte Carlo sampling: N draws per team
    samples = rng.normal(
        loc   = mu[np.newaxis, :],
        scale = sigma[np.newaxis, :],
        size  = (n_simulations, len(teams)),
    )
    samples = np.clip(samples, POINTS_MIN, POINTS_MAX)
    final_points = starting_points[np.newaxis, :] + samples     # (n_sim, n_teams)

    # ranks within each iteration
    ranks = _ranks_from_points(final_points)

    # ground truth: actual next-5 IS the actual MW34-38 sum stored in the target
    actual_remaining = np.array([
        float(pivot_rows[t][target_col])
        if pd.notna(pivot_rows[t][target_col]) else np.nan
        for t in teams
    ])
    actual_finals = starting_points + actual_remaining

    if not np.any(np.isnan(actual_finals)):
        actual_sort_idx = np.argsort(-actual_finals)
        actual_ranks    = np.empty_like(actual_sort_idx)
        actual_ranks[actual_sort_idx] = np.arange(1, len(teams) + 1)
    else:
        actual_ranks = np.full(len(teams), np.nan)

    # position probability matrix: P(team t finishes at position p)
    n_teams = len(teams)
    position_probs = np.zeros((n_teams, n_teams))
    for t_idx in range(n_teams):
        for p in range(1, n_teams + 1):
            position_probs[t_idx, p - 1] = (ranks[:, t_idx] == p).mean()

    # per-team summary table
    team_stats = pd.DataFrame({
        "team":             teams,
        "pivot_matchweek":  [pivot_rows[t]["matchweek"] for t in teams],
        "starting_points":  starting_points,
        "pred_q10":         q_low,
        "pred_q50":         q_med,
        "pred_q90":         q_high,
        "final_mean":       final_points.mean(axis=0),
        "final_p10":        np.percentile(final_points, 10, axis=0),
        "final_p50":        np.percentile(final_points, 50, axis=0),
        "final_p90":        np.percentile(final_points, 90, axis=0),
        "actual_final":     actual_finals,
        "position_mean":    ranks.mean(axis=0),
        "actual_position":  actual_ranks,
        "p_top_4":          (ranks <= 4).mean(axis=0),
        "p_top_6":          (ranks <= 6).mean(axis=0),
        "p_relegated":      (ranks >= n_teams - 2).mean(axis=0),  # bottom 3
    }).sort_values("final_mean", ascending=False).reset_index(drop=True)

    return {
        "season":           season,
        "team_stats":       team_stats,
        "final_points":     final_points,
        "ranks":            ranks,
        "teams":            teams,
        "position_probs":   position_probs,
        "actual_finals":    actual_finals,
        "actual_ranks":     actual_ranks,
    }


# ─── reporting ─────────────────────────────────────────────────────────────────

def print_standings_forecast(sim_result: dict, top_n: int | None = None) -> None:
    print("\n" + "=" * 92)
    print(f"{'END-OF-SEASON STANDINGS FORECAST  —  season ' + str(sim_result['season']):^92}")
    print("=" * 92)

    ts = sim_result["team_stats"].copy()
    if top_n is not None:
        ts = ts.head(top_n)
    cols = [
        "team", "starting_points", "pred_q50",
        "final_p10", "final_mean", "final_p90",
        "actual_final", "position_mean", "actual_position",
        "p_top_4", "p_relegated",
    ]
    for c in ["pred_q50", "final_p10", "final_mean", "final_p90",
              "position_mean"]:
        ts[c] = ts[c].round(1)
    for c in ["p_top_4", "p_relegated"]:
        ts[c] = ts[c].round(3)
    print(ts[cols].to_string(index=False))


def validate_simulation(sim_result: dict) -> dict:
    """Compute and print validation metrics."""
    ts      = sim_result["team_stats"]
    actual  = ts["actual_final"].to_numpy(float)
    pred    = ts["final_mean"].to_numpy(float)
    p10     = ts["final_p10"].to_numpy(float)
    p90     = ts["final_p90"].to_numpy(float)

    mae_points = float(np.mean(np.abs(actual - pred)))
    coverage   = float(np.mean((actual >= p10) & (actual <= p90)))

    actual_pos = ts["actual_position"].to_numpy()
    pred_pos   = ts["position_mean"].round().astype(int).to_numpy()

    pos_match   = float(np.mean(actual_pos == pred_pos)) if not np.any(np.isnan(actual_pos)) else None
    pos_within2 = float(np.mean(np.abs(actual_pos - pred_pos) <= 2)) if not np.any(np.isnan(actual_pos)) else None

    print("\n" + "=" * 92)
    print(f"{'SIMULATION VALIDATION':^92}")
    print("=" * 92)
    print(f"  MAE on final season points:        {mae_points:.2f}")
    print(f"  80% interval covers actual:        {coverage:.1%}  (target 80%)")
    if pos_match is not None:
        print(f"  Exact position match:              {pos_match:.1%}")
        print(f"  Position match within ±2:          {pos_within2:.1%}")

    return {
        "mae_points":   mae_points,
        "coverage":     coverage,
        "pos_match":    pos_match,
        "pos_within2": pos_within2,
    }