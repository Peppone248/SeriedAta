"""
test_scraper.py — validate FBref access via soccerdata.

Run from the team_trend/ directory:
    python test_scraper.py

Steps:
  1. Read the season schedule (gives match list + game_ids)
  2. Read team match stats (schedule stat_type)
  3. Read player match stats for ONE match (summary)
  4. Read all player-match stat types for that match

soccerdata downloads on first run (slow, respects rate limits) and
caches under ~/soccerdata/data/FBref. Re-runs are fast.

Delete this file once validated.
"""

from __future__ import annotations

import logging

from scrapers.fbref_source import FBrefSource

src = FBrefSource(seasons="2024-2025")

sched = src.read_team_match_stats(stat_type="schedule")
shoot = src.read_team_match_stats(stat_type="shooting")

print("SCHEDULE:", [c for c in sched.columns])
print()
print("SHOOTING:", [c for c in shoot.columns])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

SEASON = "2024-2025"


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {n} — {title}\n{'=' * 70}")


def main() -> None:
    source = FBrefSource(seasons=SEASON)

    # STEP 1: schedule
    step(1, "Read season schedule")
    schedule = source.read_schedule()
    print(f"\n  Schedule shape: {schedule.shape}")
    print(f"  Index names: {schedule.index.names}")
    print(f"  Columns: {schedule.columns.tolist()}")
    print(f"\n  First 3 matches:")
    print(schedule.head(3).to_string())

    # extract a game_id for later steps — use the game_id COLUMN, not the index
    game_ids = schedule["game_id"].dropna().tolist()
    sample_game = game_ids[0] if game_ids else None
    print(f"\n  Sample game id: {sample_game}")

    # STEP 2: team match stats
    step(2, "Read team match stats (schedule)")
    team_stats = source.read_team_match_stats(stat_type="schedule")
    print(f"\n  Team match stats shape: {team_stats.shape}")
    print(f"  Columns: {team_stats.columns.tolist()[:15]} ...")

    # STEP 3: player match stats for one match
    step(3, "Read player match stats (summary) for one match")
    if sample_game is not None:
        summary = source.read_player_match_stats(
            stat_type="summary", match_id=sample_game
        )
        print(f"\n  Player summary shape: {summary.shape}")
        print(f"  Columns: {summary.columns.tolist()[:15]} ...")
        print(f"\n  First 5 players:")
        print(summary.head(5).to_string())
    else:
        print("  SKIP: no game id available")

    # STEP 4: all player-match stat types for that match
    step(4, "Read all player-match stat types for one match")
    if sample_game is not None:
        all_stats = source.read_all_player_match_stats(match_id=sample_game)
        print(f"\n  Stat types fetched: {len(all_stats)}")
        for name, df in all_stats.items():
            print(f"    {name:<14}  {df.shape}")
    else:
        print("  SKIP: no game id available")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
