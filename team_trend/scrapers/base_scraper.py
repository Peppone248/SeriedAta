"""
scrapers/base_scraper.py — base HTTP client for FBref.

Strategy (inspired by https://medium.com/@henrik.schjoth/...):
  - Try requests + BeautifulSoup first (fast, no browser overhead)
  - Fall back to Selenium headless Chrome only when the table is not
    found in static HTML (some FBref tables are JS-rendered)

Features:
  - Minimal User-Agent only (more headers = bigger bot fingerprint)
  - Fixed 6-second delay between requests (FBref tolerates >= 5s)
  - Local HTML cache to avoid re-scraping during development
  - Lazy Selenium initialisation (Chrome opens only if needed)
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FBrefScraper:
    BASE_URL  = "https://fbref.com"
    CACHE_DIR = Path("data/raw/cache")

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(
        self,
        delay:                 float = 6.0,
        use_cache:             bool  = True,
        use_selenium_fallback: bool  = True,
        selenium_timeout:      int   = 15,
    ) -> None:
        """
        Args:
            delay:                 Seconds to wait between non-cached requests.
            use_cache:             Cache HTML files locally to skip repeat fetches.
            use_selenium_fallback: If True, retry with headless Chrome when
                                   a table is not found via requests.
            selenium_timeout:      Max seconds to wait for a JS-rendered table.
        """
        self.delay                 = delay
        self.use_cache             = use_cache
        self.use_selenium_fallback = use_selenium_fallback
        self.selenium_timeout      = selenium_timeout

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._driver = None   # lazy: only initialised if Selenium needed

    # cache helpers
    def _cache_path(self, url: str) -> Path:
        key = hashlib.md5(url.encode()).hexdigest()
        return self.CACHE_DIR / f"{key}.html"

    def _is_cached(self, url: str) -> bool:
        return self.use_cache and self._cache_path(url).exists()

    def _read_cache(self, url: str) -> str:
        return self._cache_path(url).read_text(encoding="utf-8")

    def _write_cache(self, url: str, html: str) -> None:
        self._cache_path(url).write_text(html, encoding="utf-8")

    # HTTP via requests
    def _fetch_with_requests(self, url: str) -> str:
        """Standard HTTP GET with delay. Raises on HTTP errors."""
        logger.info("requests: GET %s (delay=%.1fs)", url, self.delay)
        time.sleep(self.delay)

        response = requests.get(url, headers=self.HEADERS, timeout=30)
        response.raise_for_status()
        return response.text

    # HTTP via Selenium (lazy init)
    def _init_selenium(self) -> None:
        """Initialise headless Chrome on first use."""
        if self._driver is not None:
            return

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument(f"--user-agent={self.HEADERS['User-Agent']}")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=opts)

        # rimuove il flag navigator.webdriver che rivela Selenium
        self._driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": (
                    "Object.defineProperty(navigator, 'webdriver', "
                    "{get: () => undefined})"
                )
            },
        )

        logger.info("Selenium initialised (headless Chrome, anti-detection)")

    def _fetch_with_selenium(self, url: str, wait_for: Optional[str] = None) -> str:
        """
        Fetch via headless Chrome.
        Optionally waits for a specific element (by id) before returning HTML.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        self._init_selenium()
        logger.info("selenium: GET %s", url)

        self._driver.get(url)

        if wait_for:
            try:
                WebDriverWait(self._driver, self.selenium_timeout).until(
                    EC.presence_of_element_located((By.ID, wait_for))
                )
            except Exception as exc:
                logger.warning("Selenium wait for '%s' failed: %s", wait_for, exc)

        return self._driver.page_source

    def close(self) -> None:
        """Release the Selenium driver if it was initialised."""
        if self._driver is not None:
            self._driver.quit()
            self._driver = None
            logger.info("Selenium driver closed")

    # public API
    def get(self, url: str, force_selenium: bool = False) -> str:
        """
        Fetch a page. Uses cache if available.
        Falls back to Selenium automatically on requests failures (403, etc.)
        when use_selenium_fallback is enabled.
        """
        if self._is_cached(url):
            logger.debug("cache hit: %s", url)
            return self._read_cache(url)

        if force_selenium:
            html = self._fetch_with_selenium(url)
        else:
            try:
                html = self._fetch_with_requests(url)
            except requests.HTTPError as exc:
                if self.use_selenium_fallback:
                    logger.warning(
                        "requests failed (%s) — falling back to Selenium",
                        exc.response.status_code,
                    )
                    html = self._fetch_with_selenium(url)
                else:
                    raise
            except requests.RequestException as exc:
                if self.use_selenium_fallback:
                    logger.warning(
                        "requests failed (%s) — falling back to Selenium",
                        type(exc).__name__,
                    )
                    html = self._fetch_with_selenium(url)
                else:
                    raise

        if self.use_cache:
            self._write_cache(url, html)

        return html

    def read_table(self, url: str, table_id: str) -> pd.DataFrame:
        """
        Parse a single HTML table by id, with Selenium fallback.

        Flow:
            1. Try requests -> BeautifulSoup
            2. If table not found and fallback enabled, retry with Selenium
            3. Return empty DataFrame if still not found
        """
        html = self.get(url)
        df   = self._parse_table(html, table_id)

        if df.empty and self.use_selenium_fallback:
            logger.info("Table '%s' missing in static HTML - trying Selenium", table_id)
            html = self._fetch_with_selenium(url, wait_for=table_id)
            df   = self._parse_table(html, table_id)

            if not df.empty and self.use_cache:
                # overwrite cache with the richer Selenium version
                self._write_cache(url, html)

        if df.empty:
            logger.warning("Table '%s' not found at %s", table_id, url)

        return df

    def read_all_tables(self, url: str) -> dict[str, pd.DataFrame]:
        """Parse every HTML table on a page, keyed by its id attribute."""
        html  = self.get(url)
        soup  = BeautifulSoup(html, "lxml")
        result: dict[str, pd.DataFrame] = {}

        for table in soup.find_all("table"):
            tid = table.get("id", "")
            if not tid:
                continue
            try:
                df = pd.read_html(str(table))[0]
                result[tid] = df
            except Exception as exc:
                logger.debug("Could not parse table '%s': %s", tid, exc)

        logger.info("Parsed %d tables from %s", len(result), url)
        return result

    # parser helper
    @staticmethod
    def _parse_table(html: str, table_id: str) -> pd.DataFrame:
        """Find a table by id using BeautifulSoup, parse with pandas."""
        soup  = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": table_id})
        if table is None:
            return pd.DataFrame()
        try:
            return pd.read_html(str(table))[0]
        except Exception as exc:
            logger.warning("read_html failed for table '%s': %s", table_id, exc)
            return pd.DataFrame()