"""
scrapers/base_scraper.py — base HTTP client for FBref.

Handles:
  - Browser-like headers to avoid bot detection
  - Rate limiting (FBref requires > 3s between requests)
  - Local HTML cache to avoid re-scraping during development
  - Automatic retry on transient errors (5xx, timeout)

All scrapers inherit from FBrefScraper and call self.get() / self.read_table().
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class FBrefScraper:
    BASE_URL  = "https://fbref.com"
    CACHE_DIR = Path("data/raw/cache")

    # Mimics a real browser to avoid 403 / bot blocks
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://fbref.com/",
        "Connection":      "keep-alive",
    }

    def __init__(
        self,
        delay_range: tuple[float, float] = (4.0, 7.0),
        use_cache:   bool                = True,
        max_retries: int                 = 3,
    ) -> None:
        """
        Args:
            delay_range: (min, max) seconds between HTTP requests.
                         FBref enforces > 3s — values below cause 429.
            use_cache:   Cache raw HTML locally. Set False to force re-fetch.
            max_retries: Retry on 5xx or connection errors.
        """
        self.delay_range = delay_range
        self.use_cache   = use_cache
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        retry_strategy = Retry(
            total         = max_retries,
            backoff_factor= 2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.headers.update(self.HEADERS)

    # ── cache ─────────────────────────────────────────────────────────────

    def _cache_path(self, url: str) -> Path:
        key = hashlib.md5(url.encode()).hexdigest()
        return self.CACHE_DIR / f"{key}.html"

    def _is_cached(self, url: str) -> bool:
        return self.use_cache and self._cache_path(url).exists()

    def _read_cache(self, url: str) -> str:
        return self._cache_path(url).read_text(encoding="utf-8")

    def _write_cache(self, url: str, content: str) -> None:
        self._cache_path(url).write_text(content, encoding="utf-8")

    # ── HTTP ──────────────────────────────────────────────────────────────

    def get(self, url: str) -> str:
        """
        Fetch a URL, using local cache if available.
        Applies random delay between requests to respect FBref rate limits.
        """
        if self._is_cached(url):
            logger.debug("Cache hit: %s", url)
            return self._read_cache(url)

        delay = random.uniform(*self.delay_range)
        logger.info("Fetching %s (delay=%.1fs)", url, delay)
        time.sleep(delay)

        response = self._session.get(url, timeout=30)
        response.raise_for_status()

        if self.use_cache:
            self._write_cache(url, response.text)

        return response.text

    # ── table parsing ─────────────────────────────────────────────────────

    def read_table(self, url: str, table_id: str) -> pd.DataFrame:
        """
        Parse a specific HTML table by its id attribute.

        Args:
            url:      Full FBref URL.
            table_id: Value of the HTML id attribute (e.g. 'sched_2024-2025_11_1').

        Returns:
            Parsed DataFrame, or empty DataFrame if table not found.
        """
        html   = self.get(url)
        tables = pd.read_html(html, attrs={"id": table_id}, flavor="lxml")
        if not tables:
            logger.warning("Table '%s' not found at %s", table_id, url)
            return pd.DataFrame()
        return tables[0]

    def read_all_tables(self, url: str) -> dict[str, pd.DataFrame]:
        """
        Parse all HTML tables from a page, keyed by their id attribute.
        Useful for match reports that contain multiple tables.
        """
        from bs4 import BeautifulSoup

        html   = self.get(url)
        soup   = BeautifulSoup(html, "lxml")
        result = {}

        for table in soup.find_all("table"):
            tid = table.get("id", "")
            if tid:
                try:
                    df = pd.read_html(str(table))[0]
                    result[tid] = df
                except Exception as exc:
                    logger.debug("Could not parse table '%s': %s", tid, exc)

        logger.info("Parsed %d tables from %s", len(result), url)
        return result
