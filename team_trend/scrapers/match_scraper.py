"""
scrapers/match_scraper.py — scrapes all tables from a FBref match report.

Each match report contains multiple stat tables:
    summary      → scorers, cards, VAR events
    keeper_stats → goalkeeper performance (PSxG, saves, etc.)
    shots        → individual shot events (minute, xG, distance, body part)
    passing      → per-player passing stats
    pass_types   → breakdown of pass types
    defense      → tackles, interceptions, pressures, blocks
    possession   → dribbles, carries, progressive carries
    misc         → fouls, aerial duels, offsides
    lineup       → starting XI with shirt numbers and positions

Output: dict[table_name → DataFrame], one entry per available table.
All tables include match_url and date as join keys.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .base_scraper import FBrefScraper

logger = logging.getLogger(__name__)

# FBref table IDs in a match report — prefix pattern
# home team tables: stats_{team_id}_summary, stats_{team_id}_passing, etc.
# We collect both teams and stack them.
STAT_SUFFIXES = [
    "summary",
    "keeper_stats",
    "shots_all",
    "passing",
    "passing_types",
    "defense",
    "possession",
    "misc",
]


@dataclass
class MatchReport:
    """
    Container for all scraped tables from one match report.

    Attributes:
        match_url:   Source URL
        date:        Match date (parsed from URL or page)
        home_team:   Home team name
        away_team:   Away team name
        home_score:  Goals scored by home team
        away_score:  Goals scored by away team
        tables:      Dict mapping table name to DataFrame
    """
    match_url:  str
    date:       Optional[pd.Timestamp]   = None
    home_team:  str                      = ""
    away_team:  str                      = ""
    home_score: Optional[int]            = None
    away_score: Optional[int]            = None
    tables:     dict[str, pd.DataFrame]  = field(default_factory=dict)

    @property
    def result(self) -> Optional[str]:
        """Shorthand score string, e.g. '2-1'."""
        if self.home_score is not None and self.away_score is not None:
            return f"{self.home_score}-{self.away_score}"
        return None


class MatchScraper(FBrefScraper):

    def scrape_match(self, match_url: str) -> MatchReport:
        """
        Scrape all available stat tables from one match report URL.

        Args:
            match_url: e.g. https://fbref.com/en/matches/2fc37926/...

        Returns:
            MatchReport with populated tables dict.
        """
        logger.info("Scraping match: %s", match_url)
        all_tables = self.read_all_tables(match_url)

        report = MatchReport(match_url=match_url)
        report.date = self._parse_date_from_url(match_url)

        # ── identify team IDs from table names ────────────────────────────
        team_ids = self._extract_team_ids(all_tables)

        if len(team_ids) >= 2:
            report.home_team = team_ids[0]
            report.away_team = team_ids[1]

        # ── collect and stack per-team tables ─────────────────────────────
        for suffix in STAT_SUFFIXES:
            frames = []
            for i, tid in enumerate(team_ids):
                table_key = f"stats_{tid}_{suffix}"
                if table_key in all_tables:
                    df = all_tables[table_key].copy()
                    df["team"]     = tid
                    df["is_home"]  = int(i == 0)
                    df["match_url"]= match_url
                    frames.append(df)

            if frames:
                report.tables[suffix] = pd.concat(frames, ignore_index=True)

        # ── shots table (single table, both teams) ────────────────────────
        shots_key = "shots_all"
        if shots_key in all_tables:
            report.tables["shots"] = all_tables[shots_key].copy()
            report.tables["shots"]["match_url"] = match_url

        # ── parse scoreline from summary ──────────────────────────────────
        report.home_score, report.away_score = self._parse_score(all_tables)

        logger.info(
            "Match %s vs %s (%s): %d tables scraped",
            report.home_team, report.away_team,
            report.result or "N/A", len(report.tables)
        )
        return report

    def scrape_matches(self, match_urls: list[str]) -> list[MatchReport]:
        """Scrape a list of match URLs sequentially."""
        reports = []
        for i, url in enumerate(match_urls):
            logger.info("Progress: %d / %d", i + 1, len(match_urls))
            try:
                report = self.scrape_match(url)
                reports.append(report)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", url, exc)
        return reports

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_team_ids(tables: dict[str, pd.DataFrame]) -> list[str]:
        """
        Infer team IDs from table names.
        FBref names them: stats_{team_id}_{suffix}
        Returns two IDs: [home_id, away_id]
        """
        ids = []
        pattern = re.compile(r"^stats_([a-f0-9]+)_summary$")
        for key in tables:
            match = pattern.match(key)
            if match:
                ids.append(match.group(1))
        return ids[:2]

    @staticmethod
    def _parse_date_from_url(url: str) -> Optional[pd.Timestamp]:
        """
        Extract date from a URL like:
            .../Fiorentina-Monza-September-1-2024-Serie-A
        """
        pattern = re.search(
            r"-(\w+)-(\d{1,2})-(\d{4})-Serie-A",
            url
        )
        if pattern:
            try:
                date_str = f"{pattern.group(1)} {pattern.group(2)} {pattern.group(3)}"
                return pd.to_datetime(date_str, format="%B %d %Y")
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_score(
        tables: dict[str, pd.DataFrame]
    ) -> tuple[Optional[int], Optional[int]]:
        """Attempt to extract home/away score from available tables."""
        # placeholder — implement based on actual FBref HTML structure
        return None, None
