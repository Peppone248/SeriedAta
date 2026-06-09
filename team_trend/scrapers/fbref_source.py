"""
scrapers/fbref_source.py — FBref data access via the soccerdata library.

Why soccerdata instead of a custom scraper:
  - FBref uses Cloudflare bot protection that blocks raw requests/Selenium
  - soccerdata is actively maintained to stay compatible with FBref
  - It handles rate limiting, caching, and parsing, returning clean DataFrames
  - It lets us focus on the ETL and modelling, not anti-bot workarounds

This module wraps soccerdata.FBref and exposes the data our Bronze layer
needs, keyed and named consistently with the rest of the project.

soccerdata caches everything under ~/soccerdata/data/FBref by default.
"""

from __future__ import annotations

import logging

import pandas as pd

import soccerdata as sd

logger = logging.getLogger(__name__)

# soccerdata league identifier for Serie A
LEAGUE = "ITA-Serie A"

# stat types available per granularity
PLAYER_MATCH_STAT_TYPES = [
    "summary",       # goals, assists, xG, shots, SoT, cards, fouls, TklW, Int
    "keepers",       # saves, goals against, PSxG, save%
]


class FBrefSource:
    """
    Thin wrapper around soccerdata.FBref.

    Args:
        seasons:  Season string(s). soccerdata accepts '2024-2025', '24-25', 2024.
        no_cache: Force re-download (default False — use cache).
        no_store: Don't write to cache (default False).
    """

    def __init__(
        self,
        seasons:  str | int | list,
        no_cache: bool = False,
        no_store: bool = False,
    ) -> None:
        self.seasons = seasons
        self._fbref  = sd.FBref(
            leagues  = LEAGUE,
            seasons  = seasons,
            no_cache = no_cache,
            no_store = no_store,
        )
        logger.info("FBrefSource initialised: league=%s seasons=%s", LEAGUE, seasons)

    # ── schedule ──────────────────────────────────────────────────────────

    def read_schedule(self) -> pd.DataFrame:
        """
        Match schedule with results, dates, and FBref game_id index.
        The game_id is needed to fetch per-match player stats.
        """
        logger.info("Reading schedule...")
        df = self._fbref.read_schedule()
        logger.info("Schedule: %d matches", len(df))
        return df

    # ── team-level ────────────────────────────────────────────────────────

    def read_team_match_stats(self, stat_type: str = "schedule") -> pd.DataFrame:
        """
        Team-level stats per match (one row per team per match).
        stat_type options include: schedule, shooting, keeper, passing,
        defense, possession, misc.
        """
        logger.info("Reading team match stats: %s", stat_type)
        return self._fbref.read_team_match_stats(stat_type=stat_type)

    def read_team_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """Team-level aggregated season stats."""
        logger.info("Reading team season stats: %s", stat_type)
        return self._fbref.read_team_season_stats(stat_type=stat_type)

    # ── player-level ──────────────────────────────────────────────────────

    def read_player_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """Player-level aggregated season stats."""
        logger.info("Reading player season stats: %s", stat_type)
        return self._fbref.read_player_season_stats(stat_type=stat_type)

    def read_player_match_stats(
        self,
        stat_type: str = "summary",
        match_id:  str | list | None = None,
    ) -> pd.DataFrame:
        """
        Player-level stats for a specific match (or all matches if match_id None).

        Args:
            stat_type: one of PLAYER_MATCH_STAT_TYPES
            match_id:  FBref game_id from read_schedule().index, or None for all
        """
        logger.info("Reading player match stats: %s (match_id=%s)", stat_type, match_id)
        return self._fbref.read_player_match_stats(
            stat_type=stat_type, match_id=match_id
        )

    def read_all_player_match_stats(
        self,
        match_id: str | list | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch every player-match stat type in one call.
        Returns dict keyed by stat_type → DataFrame.
        This is the main entry point for the Bronze layer.
        """
        result = {}
        for stat_type in PLAYER_MATCH_STAT_TYPES:
            try:
                result[stat_type] = self.read_player_match_stats(
                    stat_type=stat_type, match_id=match_id
                )
            except Exception as exc:
                logger.warning("Could not read %s: %s", stat_type, exc)
                result[stat_type] = pd.DataFrame()
        return result
