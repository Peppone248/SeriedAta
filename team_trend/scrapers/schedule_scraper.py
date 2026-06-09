"""
scrapers/schedule_scraper.py — scrapes the Serie A season schedule from FBref.

Produces a DataFrame where each row is a match with:
    date, time, home_team, away_team, score, match_url, season

Used downstream by match_scraper.py to iterate over all match report URLs.

FBref schedule URL pattern:
    https://fbref.com/en/comps/11/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from base_scraper import FBrefScraper

logger = logging.getLogger(__name__)

SERIE_A_COMP_ID = "11"
SCHEDULE_TABLE_ID_PATTERN = "sched_{season}_11_1"


class ScheduleScraper(FBrefScraper):

    def schedule_url(self, season: str) -> str:
        """
        Args:
            season: e.g. "2024-2025"
        """
        return (
            f"{self.BASE_URL}/en/comps/{SERIE_A_COMP_ID}/"
            f"{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"
        )

    def scrape_season(self, season: str) -> pd.DataFrame:
        """
        Scrape the full schedule for one season.

        Returns:
            DataFrame with columns:
                date, time, home_team, away_team, home_score, away_score,
                attendance, venue, referee, match_url, season
        """
        url      = self.schedule_url(season)
        table_id = SCHEDULE_TABLE_ID_PATTERN.format(season=season)

        logger.info("Scraping schedule: season=%s", season)
        raw = self.read_table(url, table_id)

        if raw.empty:
            logger.warning("Empty schedule for season %s", season)
            return pd.DataFrame()

        df = self._parse_schedule(raw, season)
        logger.info("Season %s: %d matches found", season, len(df))
        return df

    def scrape_seasons(self, seasons: list[str]) -> pd.DataFrame:
        """Scrape multiple seasons and concatenate."""
        frames = []
        for season in seasons:
            df = self.scrape_season(season)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _parse_schedule(self, raw: pd.DataFrame, season: str) -> pd.DataFrame:
        """
        Clean and standardise raw schedule table from FBref.

        FBref multi-level columns are flattened; irrelevant rows (e.g.
        repeated header rows mid-table) are dropped.
        """
        df = raw.copy()

        # flatten multi-level column headers if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, c))).strip()
                          for c in df.columns]

        # drop repeated header rows (FBref inserts them every ~10 rows)
        df = df[df.get("Date", pd.Series()) != "Date"].copy()

        # rename to standard names — adjust if FBref column names change
        rename_map = {
            "Date":       "date",
            "Time":       "time",
            "Home":       "home_team",
            "Away":       "away_team",
            "Score":      "score",
            "Attendance": "attendance",
            "Venue":      "venue",
            "Referee":    "referee",
            "Match Report": "match_report_text",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # parse score into home_score / away_score
        if "score" in df.columns:
            score_split = df["score"].str.extract(r"(\d+)–(\d+)")
            df["home_score"] = pd.to_numeric(score_split[0], errors="coerce")
            df["away_score"] = pd.to_numeric(score_split[1], errors="coerce")

        # extract match URL from the HTML — needs raw HTML pass for full URL
        # placeholder: match_url populated by ScheduleScraper._extract_match_urls
        df["season"]    = season
        df["match_url"] = pd.NA

        # parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # drop rows without date (unplayed future matches)
        df = df.dropna(subset=["date"])

        return df.reset_index(drop=True)

    def scrape_match_urls(self, season: str) -> list[str]:
        """
        Extract all match report URLs for a season.
        These are the /en/matches/... links in the schedule page.
        """
        from bs4 import BeautifulSoup

        url  = self.schedule_url(season)
        html = self.get(url)
        soup = BeautifulSoup(html, "lxml")

        urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.match(r"/en/matches/[a-f0-9]+/", href):
                full_url = self.BASE_URL + href
                if full_url not in urls:
                    urls.append(full_url)

        logger.info("Season %s: %d match URLs found", season, len(urls))
        return urls
