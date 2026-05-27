"""
Historical NBA injury data from prosports-transactions.com.

Key design: fetch all transactions for a season ONCE, then replay in memory
for each game date. This avoids re-fetching from season start on every date,
which would be O(n_dates × n_chunks) HTTP requests instead of O(n_chunks).
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.prosports-transactions.com/basketball/search.php"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _season_bounds(d: date) -> tuple[date, date]:
    """Return (season_start, season_end) for the NBA season containing date d."""
    year = d.year if d.month >= 10 else d.year - 1
    return date(year, 10, 1), date(year + 1, 6, 30)


def _fetch_chunk(start: date, end: date) -> list[dict]:
    params = {
        "Sport": "bbl",
        "startDate": start.strftime("%m/%d/%Y"),
        "endDate": end.strftime("%m/%d/%Y"),
        "isOnlyInjuries": "yes",
        "submit": "yes",
    }
    resp = requests.get(_BASE_URL, params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for tr in soup.select("table tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue
        rows.append({
            "date": cells[0],
            "team": cells[1],
            "relinquished": cells[2],
            "acquired": cells[3],
        })
    return rows


def fetch_season_transactions(season_start: date, season_end: date) -> list[dict]:
    """
    Fetch all injury transactions for a date range in monthly chunks.
    Returns rows sorted by date with parsed date objects attached.
    Call this once per season, then use snapshot_at_date() for each game date.
    """
    all_rows = []
    cursor = season_start
    while cursor <= season_end:
        chunk_end = min(cursor + timedelta(days=29), season_end)
        try:
            rows = _fetch_chunk(cursor, chunk_end)
            all_rows.extend(rows)
            logger.debug(f"Fetched {len(rows)} transactions {cursor} → {chunk_end}")
        except Exception as e:
            logger.warning(f"Failed chunk {cursor}–{chunk_end}: {e}")
        cursor = chunk_end + timedelta(days=1)
        time.sleep(1)

    # Parse and attach date objects for fast comparison during replay
    for row in all_rows:
        try:
            row["_date"] = pd.to_datetime(row["date"]).date()
        except Exception:
            row["_date"] = None

    all_rows = [r for r in all_rows if r["_date"] is not None]
    all_rows.sort(key=lambda r: r["_date"])
    logger.info(f"Season {season_start.year}: {len(all_rows)} total transactions fetched")
    return all_rows


def snapshot_at_date(transactions: list[dict], target_date: date) -> dict[str, list[dict]]:
    """
    Replay a pre-fetched transaction list up to target_date to get the IL snapshot.
    Transactions must be sorted by date (fetch_season_transactions guarantees this).

    Tracks IL entry date per player so days_out can be used downstream for decay.
    Returns {team_abbreviation: [{"player_name", "status", "reason", "days_out"}]}.
    """
    on_il: dict[str, dict[str, date]] = {}  # team → {player_name → il_entry_date}
    for row in transactions:
        if row["_date"] > target_date:
            break
        team = row["team"]
        if row["relinquished"]:
            on_il.setdefault(team, {})[row["relinquished"]] = row["_date"]
        if row["acquired"]:
            on_il.get(team, {}).pop(row["acquired"], None)

    return {
        team: [
            {
                "player_name": name,
                "status": "Out",
                "reason": "",
                "days_out": (target_date - entry_date).days,
            }
            for name, entry_date in players.items()
        ]
        for team, players in on_il.items()
        if players
    }
