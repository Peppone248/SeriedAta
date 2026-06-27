"""Wyscout data download and parsing.

Source: Pappalardo et al. (2019), "A public data set of spatio-temporal
match events in soccer competitions", Scientific Data 6:236.
Mirror:  https://github.com/koenvo/wyscout-soccer-match-event-dataset

This module:
    1. Discovers Serie A 2017/18 match IDs from the dataset index
    2. Downloads each match's events JSON (idempotent — skips existing)
    3. Parses all events into a flat pandas DataFrame
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def fetch_serie_a_match_ids() -> list[int]:
    """Parse the dataset index README to extract Serie A match IDs."""
    print(f"  Fetching match index from {config.WYSCOUT_INDEX_URL}")
    with urllib.request.urlopen(config.WYSCOUT_INDEX_URL) as resp:
        index_text = resp.read().decode("utf-8")

    # Match index rows look like:
    # |[2575959](files/2575959.json)|Atalanta - Roma, 0 - 1|... |matches_Italy.json|
    # Match lines like: |[2575959](files/2575959.json)|Atalanta - Roma...|...|matches_Italy.json|
    # Strategy: find all lines containing "matches_Italy.json", then extract
    # the numeric ID from each. This is more robust than a single regex that
    # tries to match the full pipe-separated structure (which varies by
    # whitespace and encoding across OS/download methods).
    match_ids = []
    for line in index_text.splitlines():
        if "matches_Italy.json" not in line:
            continue
        id_match = re.search(r"\[(\d+)\]\(files/", line)
        if id_match:
            match_ids.append(int(id_match.group(1)))
    print(f"  Found {len(match_ids)} Serie A match IDs in index")
    return match_ids


def download_match(match_id: int, dest_dir: Path) -> tuple[int, bool]:
    """Download a single match JSON if not already present."""
    dest = dest_dir / f"{match_id}.json"
    if dest.exists() and dest.stat().st_size > 1000:
        return match_id, True
    url = f"{config.WYSCOUT_BASE_URL}/{match_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return match_id, True
    except Exception as exc:
        print(f"    failed: {match_id} ({exc})")
        return match_id, False


def download_all_matches(match_ids: list[int], dest_dir: Path, max_workers: int = 20) -> int:
    """Download all matches in parallel."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_match, mid, dest_dir): mid for mid in match_ids}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            _, ok = future.result()
            successes += int(ok)
            if i % 50 == 0:
                print(f"  Downloaded {i}/{len(match_ids)}")
    return successes


def ensure_data(force_download: bool = False) -> Path:
    """Ensure Wyscout data is present locally. Returns the data directory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.DATA_DIR.glob("*.json"))
    if len(existing) >= 380 and not force_download:
        print(f"  Data already present: {len(existing)} files at {config.DATA_DIR}")
        return config.DATA_DIR

    print("  Downloading Wyscout Serie A 2017/18 event data...")
    match_ids = fetch_serie_a_match_ids()
    if not match_ids:
        raise RuntimeError(
            "Failed to find any Serie A match IDs in the Wyscout index. "
            "This usually means the GitHub download returned unexpected "
            "content. Check your internet connection and try again."
        )
    n_ok = download_all_matches(match_ids, config.DATA_DIR)
    if n_ok == 0:
        raise RuntimeError(
            f"Downloaded 0/{len(match_ids)} match files. "
            "Check your internet connection and firewall settings."
        )
    print(f"  Downloaded {n_ok}/{len(match_ids)} matches successfully")
    return config.DATA_DIR


def parse_match_file(fpath: Path) -> tuple[list[dict], dict]:
    """Parse a single match JSON into (events_list, metadata_dict)."""
    with open(fpath) as f:
        data = json.load(f)

    match_id = int(fpath.stem)
    teams = data.get("teams", {})
    players_data = data.get("players", {})

    # Build player → role lookup (defensive against malformed entries)
    player_roles: dict[int, str] = {}
    for _, roster in players_data.items():
        for entry in roster:
            if not entry or not entry.get("player"):
                continue
            pid = entry.get("playerId")
            if pid is None:
                continue
            role_info = entry["player"].get("role", {}) or {}
            player_roles[pid] = role_info.get("code2", "")

    team_names = {int(tid): info["name"] for tid, info in teams.items()}
    team_ids = list(team_names.keys())

    events = []
    for ev in data["events"]:
        tags = {t["id"] for t in ev.get("tags", [])}
        positions = ev.get("positions", []) or []
        origin = positions[0] if positions else {}
        dest = positions[1] if len(positions) > 1 else {}

        events.append({
            "match_id": match_id,
            "event_sec": ev.get("eventSec", 0),
            "period": ev.get("matchPeriod", ""),
            "team_id": ev["teamId"],
            "team_name": team_names.get(ev["teamId"], ""),
            "player_id": ev.get("playerId"),
            "player_role": player_roles.get(ev.get("playerId"), ""),
            "event_name": ev["eventName"],
            "sub_event": ev.get("subEventName", ""),
            "x_origin": origin.get("x"),
            "y_origin": origin.get("y"),
            "x_dest": dest.get("x"),
            "y_dest": dest.get("y"),
            "accurate": config.TAG_ACCURATE in tags,
            "is_goal": config.TAG_GOAL in tags,
        })

    meta = {"match_id": match_id, "team_ids": team_ids, "team_names": list(team_names.values())}
    return events, meta


def parse_all_matches() -> pd.DataFrame:
    """Parse all match JSONs in DATA_DIR into a single events DataFrame."""
    match_files = sorted(config.DATA_DIR.glob("*.json"))
    print(f"  Parsing {len(match_files)} match files...")
    all_events = []
    for fp in match_files:
        events, _ = parse_match_file(fp)
        all_events.extend(events)
    df = pd.DataFrame(all_events)
    print(f"  Parsed {len(df):,} events across {df['match_id'].nunique()} matches "
          f"and {df['team_name'].nunique()} teams")
    return df
