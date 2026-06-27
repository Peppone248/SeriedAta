"""DefR model: zone assignment, event classification, score computation.

Conceptual model (from Hudl/StatsBomb's article on Defensive Responsibility):
    For every opposition attacking action, somebody on the defending team
    is "expected" to respond. The expected response rate varies by zone
    (it's near 1.0 right in front of your own goal, near 0.1 in midfield).
    A team's DefR is the gap between its actual defensive actions and
    those expected from the volume of opposition attacking actions
    it faced, zone by zone.

Our adaptation, given we have aggregate event data without player coordinates:
    - Discretize the pitch into a 6×4 grid of zones (24 zones total)
    - Estimate the league-wide baseline defensive rate per zone:
          rate_z = (Σ defensive_actions in zone z) / (Σ attacking_actions in zone z)
      taken across all 380 Serie A 2017/18 matches.
    - For each team in each match:
          expected_def = Σ_z (opp_attacking_actions_in_z × rate_z)
          actual_def   = Σ_z (own_defensive_actions_in_z)
          defr_score   = actual_def − expected_def

Sign convention:
    + Positive DefR → defended more than expected (aggressive/proactive)
    − Negative DefR → defended less than expected (passive or dominant via possession)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ─── Zone assignment ──────────────────────────────────────────────────
def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each event to a zone on the 6×4 grid.

    Wyscout pitch coordinates are 0-100 on both axes, with (0,0) at the
    attacking team's own goal line. Events with missing coordinates are
    placed at the pitch centre (50,50) — a safe default that does not
    bias the rate estimates because missing-coordinate events are rare
    and uniformly distributed.
    """
    df = df.copy()
    x = df["x_origin"].clip(0, config.PITCH_X_MAX).fillna(config.PITCH_X_MAX / 2)
    y = df["y_origin"].clip(0, config.PITCH_Y_MAX).fillna(config.PITCH_Y_MAX / 2)

    df["zone_col"] = np.minimum(
        (x / (config.PITCH_X_MAX / config.N_ZONE_COLS)).astype(int),
        config.N_ZONE_COLS - 1,
    )
    df["zone_row"] = np.minimum(
        (y / (config.PITCH_Y_MAX / config.N_ZONE_ROWS)).astype(int),
        config.N_ZONE_ROWS - 1,
    )
    df["zone_id"] = df["zone_row"] * config.N_ZONE_COLS + df["zone_col"]
    return df


# ─── Event classification ─────────────────────────────────────────────
def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each event as defensive, attacking, or neither.

    The exclusion of clearances/throw-ins/goal-kicks from "attacking" is
    important: these are defensive recovery actions, not pressure on the
    opponent. Including them would inflate expected defensive demand.
    """
    df = df.copy()

    def_set = config.DEFENSIVE_EVENTS

    df["is_defensive"] = df.apply(
        lambda r: (r["event_name"], r["sub_event"]) in def_set, axis=1
    )
    df["is_attacking"] = df["event_name"].isin(config.ATTACKING_EVENT_NAMES)
    df.loc[df["sub_event"].isin(config.NON_ATTACKING_SUBEVENTS), "is_attacking"] = False
    return df


# ─── Opponent lookup ──────────────────────────────────────────────────
def add_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Add an opponent_id column to each event row."""
    df = df.copy()
    team_pairs = (
        df.groupby("match_id")["team_id"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    def get_opponent(row):
        pair = team_pairs.get(row["match_id"], [])
        if len(pair) != 2:
            return None
        return pair[1] if row["team_id"] == pair[0] else pair[0]

    df["opponent_id"] = df.apply(get_opponent, axis=1)
    return df


# ─── Zone baselines ───────────────────────────────────────────────────
def compute_zone_baselines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-zone league-baseline defensive rates.

    Returns:
        zone_rates: DataFrame indexed by zone with total_atk, total_def, baseline_rate
        attacking:  per-(match,attacking_team,zone) attacking action counts
        defensive:  per-(match,defending_team,zone) defensive action counts
    """
    team_pairs = (
        df.groupby("match_id")["team_id"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    # Attacking pressure per zone per match per team
    attacking = (
        df[df["is_attacking"]]
        .groupby(["match_id", "team_id", "zone_id"])
        .size()
        .reset_index(name="atk_actions")
        .rename(columns={"team_id": "attacking_team_id"})
    )
    attacking["defending_team_id"] = attacking.apply(
        lambda r: (
            [t for t in team_pairs[r["match_id"]] if t != r["attacking_team_id"]][0]
            if len(team_pairs.get(r["match_id"], [])) == 2
            else None
        ),
        axis=1,
    )

    # Defensive actions per zone per match per team
    defensive = (
        df[df["is_defensive"]]
        .groupby(["match_id", "team_id", "zone_id"])
        .size()
        .reset_index(name="def_actions")
    )

    # League-wide baseline: total defensive actions / total attacking
    # demand per zone. This is a maximum-likelihood estimate of the
    # rate at which defenders engage with each unit of opposition pressure.
    zone_atk = attacking.groupby("zone_id")["atk_actions"].sum()
    zone_def = defensive.groupby("zone_id")["def_actions"].sum()

    zone_rates = pd.DataFrame({"zone_id": range(config.N_ZONES)})
    zone_rates["total_atk"] = zone_rates["zone_id"].map(zone_atk).fillna(0).astype(int)
    zone_rates["total_def"] = zone_rates["zone_id"].map(zone_def).fillna(0).astype(int)
    zone_rates["baseline_rate"] = np.where(
        zone_rates["total_atk"] > 0,
        zone_rates["total_def"] / zone_rates["total_atk"],
        0.0,
    )
    zone_rates["zone_col"] = zone_rates["zone_id"] % config.N_ZONE_COLS
    zone_rates["zone_row"] = zone_rates["zone_id"] // config.N_ZONE_COLS

    return zone_rates, attacking, defensive


# ─── DefR scoring ─────────────────────────────────────────────────────
def compute_team_defr(
    attacking: pd.DataFrame,
    defensive: pd.DataFrame,
    zone_rates: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-team-match DefR scores.

    For each team in each match:
        expected_def(team, match) = Σ_z opp_attacking[z] × baseline_rate[z]
        actual_def(team, match)   = Σ_z own_defensive[z]
        defr_score                = actual_def − expected_def
        defr_ratio                = actual_def / expected_def
    """
    rate_lookup = zone_rates.set_index("zone_id")["baseline_rate"].to_dict()

    attacking = attacking.copy()
    attacking["expected_def"] = (
        attacking["zone_id"].map(rate_lookup) * attacking["atk_actions"]
    )

    expected = (
        attacking.groupby(["match_id", "defending_team_id"])["expected_def"]
        .sum()
        .reset_index()
        .rename(columns={"defending_team_id": "team_id"})
    )
    actual = (
        defensive.groupby(["match_id", "team_id"])["def_actions"]
        .sum()
        .reset_index()
        .rename(columns={"def_actions": "actual_def"})
    )

    defr = expected.merge(actual, on=["match_id", "team_id"], how="outer")
    defr["expected_def"] = defr["expected_def"].fillna(0)
    defr["actual_def"] = defr["actual_def"].fillna(0)
    defr["defr_score"] = defr["actual_def"] - defr["expected_def"]
    defr["defr_ratio"] = np.where(
        defr["expected_def"] > 0,
        defr["actual_def"] / defr["expected_def"],
        1.0,
    )

    # Attach team names
    name_lookup = (
        events[["team_id", "team_name"]]
        .drop_duplicates()
        .set_index("team_id")["team_name"]
        .to_dict()
    )
    defr["team_name"] = defr["team_id"].map(name_lookup)
    return defr


def aggregate_season(defr_match: pd.DataFrame) -> pd.DataFrame:
    """Season-level DefR aggregates per team."""
    season = (
        defr_match.groupby(["team_id", "team_name"])
        .agg(
            matches=("match_id", "count"),
            avg_defr=("defr_score", "mean"),
            std_defr=("defr_score", "std"),
            total_actual=("actual_def", "sum"),
            total_expected=("expected_def", "sum"),
            avg_ratio=("defr_ratio", "mean"),
        )
        .reset_index()
        .sort_values("avg_defr", ascending=False)
        .reset_index(drop=True)
    )
    season["rank"] = season.index + 1
    return season
