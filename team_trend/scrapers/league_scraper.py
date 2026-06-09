"""
scrapers/league_scraper.py — extract team URLs and match URLs for a season.

FBref structure for Serie A:
    schedule page:  /en/comps/11/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures
    team page:      /en/squads/{team_id}/{season}/{Team-Name}-Stats
    match report:   /en/matches/{match_id}/{teams}-{date}-Serie-A

Strategy:
    1. Scrape schedule page → extract all match URLs (one per played match)
    2. Derive team URLs from the schedule's team names
    3. For each team, scrape matchlogs_for table for the rich per-match data

Note: FBref wraps many tables in HTML comments (lazy rendering via JS).
We parse both normal DOM and comment contents to find links.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from bs4 import BeautifulSoup, Comment

from .base_scraper import FBrefScraper

logger = logging.getLogger(__name__)

SERIE_A_COMP_ID = "11"


class LeagueScraper(FBrefScraper):

    # URL builders
    def schedule_url(self, season: str) -> str:
        return (
            f"{self.BASE_URL}/en/comps/{SERIE_A_COMP_ID}/"
            f"{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"
        )

    # Phase 1: extract all match URLs from the schedule page
    def get_match_urls(self, season: str) -> list[str]:
        """
        Extract every match report URL from the season schedule.
        Returns absolute URLs for played matches only (unplayed have no link).
        """
        url  = self.schedule_url(season)
        html = self.get(url)
        soup = BeautifulSoup(html, "lxml")

        urls: list[str] = []
        seen: set[str] = set()

        # search in normal HTML and inside HTML comments
        self._collect_match_urls(soup, urls, seen)

        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "/en/matches/" in c:
                inner = BeautifulSoup(c, "lxml")
                self._collect_match_urls(inner, urls, seen)

        if not urls:
            debug = self.CACHE_DIR / f"debug_schedule_{season}.html"
            debug.write_text(html, encoding="utf-8")
            logger.error(
                "No match URLs found on %s - saved HTML to %s", url, debug,
            )

        logger.info("Season %s: %d match URLs found", season, len(urls))
        return urls

    # Phase 2: derive team URLs from the schedule
    def get_team_urls(self, season: str) -> list[tuple[str, str]]:
        """
        Extract team URLs from the schedule page.

        Each row of the schedule contains links to the two teams playing.
        We collect unique /squads/.../{season}/Team-Stats links.
        """
        url  = self.schedule_url(season)
        html = self.get(url)
        soup = BeautifulSoup(html, "lxml")

        teams: list[tuple[str, str]] = []
        seen:  set[str] = set()

        self._collect_team_links(soup, season, teams, seen)

        # also search inside HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "/squads/" not in c:
                continue
            inner = BeautifulSoup(c, "lxml")
            self._collect_team_links(inner, season, teams, seen)

        if not teams:
            debug = self.CACHE_DIR / f"debug_schedule_{season}.html"
            debug.write_text(html, encoding="utf-8")
            logger.error(
                "No team URLs found on %s - saved HTML to %s", url, debug,
            )
            return []

        logger.info("Found %d teams for season %s", len(teams), season)
        return teams

    # Phase 3: matchlogs_for table per team
    def get_team_matches(
        self,
        team_url: str,
        team_name: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Scrape the matchlogs_for table for a single team.
        Returns a DataFrame with one row per match.
        """
        logger.info("Scraping matches: %s (%s)", team_name, season)

        html = self.get(team_url)
        df   = self._parse_table(html, "matchlogs_for")

        # try inside comments if not found
        if df.empty:
            soup = BeautifulSoup(html, "lxml")
            for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
                if 'id="matchlogs_for"' not in c:
                    continue
                inner_html = c.strip()
                df = self._parse_table(inner_html, "matchlogs_for")
                if not df.empty:
                    break

        # Selenium last resort
        if df.empty and self.use_selenium_fallback:
            html = self._fetch_with_selenium(team_url, wait_for="matchlogs_for")
            df   = self._parse_table(html, "matchlogs_for")

        if df.empty:
            logger.warning("No matches found for %s", team_name)
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, c))) for c in df.columns]

        df["team"]   = team_name
        df["season"] = season

        if "Comp" in df.columns:
            df = df[df["Comp"] == "Serie A"]

        logger.info("  %s: %d Serie A matches", team_name, len(df))
        return df.reset_index(drop=True)

    # Full season scrape
    def scrape_season(self, season: str) -> pd.DataFrame:
        """League schedule -> team URLs -> all team matchlogs."""
        teams = self.get_team_urls(season)
        if not teams:
            return pd.DataFrame()

        frames = []
        for team_name, team_url in teams:
            df = self.get_team_matches(team_url, team_name, season)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            "Season %s: %d rows from %d teams", season, len(result), len(frames),
        )
        return result

    # Helpers
    @staticmethod
    def _team_name_from_url(href: str) -> str:
        last = href.rstrip("/").split("/")[-1]
        return last.replace("-Stats", "").replace("-", " ")

    def _collect_team_links(
        self,
        soup:   BeautifulSoup,
        season: str,
        teams:  list[tuple[str, str]],
        seen:   set[str],
    ) -> None:
        """
        Squad links matching pattern: /en/squads/{id}/{season}/{Name}-Stats
        """
        pattern = re.compile(
            rf"^/en/squads/[a-f0-9]+/{re.escape(season)}/[^/]+-Stats$"
        )
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not pattern.match(href):
                continue
            name = self._team_name_from_url(href)
            if not name or name in seen:
                continue
            seen.add(name)
            teams.append((name, f"{self.BASE_URL}{href}"))

    @staticmethod
    def _collect_match_urls(
        soup: BeautifulSoup,
        urls: list[str],
        seen: set[str],
    ) -> None:
        pattern = re.compile(r"^/en/matches/[a-f0-9]+/")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not pattern.match(href):
                continue
            full = f"https://fbref.com{href}"
            if full in seen:
                continue
            seen.add(full)
            urls.append(full)