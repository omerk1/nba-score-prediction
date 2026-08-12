"""
Gamma API access: NBA game discovery + market classification.

Schema verified empirically against https://gamma-api.polymarket.com on
2026-07-08. Key findings vs. the design doc's assumptions:

- Each market object carries a `sportsMarketType` field directly (e.g.
  "moneyline", "spreads", "totals", "first_half_moneyline", "points",
  "rebounds", ...). This is far more reliable than regexing the `question`
  string, so classification is based on this field.
- Polymarket creates ONE spread market and ONE totals market PER POSSIBLE
  LINE (e.g. "Spread: Spurs (-5.5)", "Spread: Spurs (-6.5)", ... a dozen+
  markets per game). Only one or two of these ever get real volume. We pick
  the highest-volume market of each type per game, per the design doc.
- The event's `teams` list has an `ordering` field ("home"/"away") which is
  a more robust source of home/away than parsing the slug, though the slug
  format `nba-{away}-{home}-{date}` (verified) agrees with it.
- `gameStartTime` is present on individual markets (format like
  "2026-06-14 00:30:00+00", parseable by pandas) for real game markets. It
  was seen as None on some older / non-standard markets (e.g. play-in
  specials) - callers must be prepared to fall back to `startDate`.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .http_utils import cached_get_json

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# sportsMarketType values that represent the FULL-GAME market of interest.
# (as opposed to first_half_*, player props, firsts, etc. which we drop)
MONEYLINE_TYPE = "moneyline"
SPREAD_TYPE = "spreads"
TOTALS_TYPE = "totals"
FULL_GAME_TYPES = {MONEYLINE_TYPE, SPREAD_TYPE, TOTALS_TYPE}


@dataclass
class MarketRecord:
    """A single Gamma market, with the fields we care about extracted."""

    condition_id: str
    slug: str
    question: str
    sports_market_type: Optional[str]
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    volume: float
    game_start_time: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def winner_index(self) -> Optional[int]:
        """Index into outcomes/clob_token_ids of the resolved winner, if resolved."""
        if not self.outcome_prices:
            return None
        try:
            return max(range(len(self.outcome_prices)), key=lambda i: self.outcome_prices[i])
        except (ValueError, TypeError):
            return None


@dataclass
class GameMarkets:
    """All markets found for one NBA game, classified."""

    event_slug: str
    event_id: Optional[str]
    away_team: Optional[str]
    home_team: Optional[str]
    game_date: Optional[str]  # derived from slug (YYYY-MM-DD)
    moneyline: Optional[MarketRecord]
    spread: Optional[MarketRecord]
    totals: Optional[MarketRecord]
    dropped: list[MarketRecord]  # markets classified as "other" (exotics, props, first-half, etc.)
    flags: list[str]


def _parse_market(m: dict[str, Any]) -> Optional[MarketRecord]:
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        outcome_prices_raw = json.loads(m.get("outcomePrices") or "[]")
        outcome_prices = [float(p) for p in outcome_prices_raw]
        clob_token_ids = json.loads(m.get("clobTokenIds") or "[]")
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug(f"Failed to parse market {m.get('slug')}: {e}")
        return None

    return MarketRecord(
        condition_id=m.get("conditionId", ""),
        slug=m.get("slug", ""),
        question=m.get("question", ""),
        sports_market_type=m.get("sportsMarketType"),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        clob_token_ids=clob_token_ids,
        volume=float(m.get("volume") or 0.0),
        game_start_time=m.get("gameStartTime"),
        start_date=m.get("startDate"),
        end_date=m.get("endDate"),
        raw=m,
    )


SLUG_RE = re.compile(r"^nba-([a-z]+)-([a-z]+)-(\d{4}-\d{2}-\d{2})$")


def fetch_event_by_slug(slug: str, raw_dir: str, force_refresh: bool = False) -> Optional[dict[str, Any]]:
    """Fetch (and cache) the full Gamma event payload for a game slug."""
    cache_path = os.path.join(raw_dir, "events", f"{slug}.json.gz")
    try:
        data = cached_get_json(
            cache_path, f"{GAMMA_BASE}/events", params={"slug": slug}, force_refresh=force_refresh
        )
    except Exception as e:
        logger.warning(f"Failed to fetch event {slug}: {e}")
        return None

    if not data:
        logger.warning(f"No event found for slug={slug}")
        return None
    return data[0]


def classify_event_markets(event: dict[str, Any]) -> GameMarkets:
    """
    Classify an event's markets into moneyline / spread / totals (highest-volume
    line each) and everything else (dropped).
    """
    slug = event.get("slug", "")
    flags: list[str] = []

    m = SLUG_RE.match(slug)
    away_from_slug, home_from_slug, date_from_slug = (m.groups() if m else (None, None, None))

    # Prefer the event's explicit teams[] (has 'ordering': home/away) over slug parsing.
    away_team = home_team = None
    for t in event.get("teams", []) or []:
        if t.get("ordering") == "home":
            home_team = t.get("name") or t.get("abbreviation")
        elif t.get("ordering") == "away":
            away_team = t.get("name") or t.get("abbreviation")
    if not (away_team and home_team):
        flags.append("teams_from_slug_fallback")
        away_team = away_team or away_from_slug
        home_team = home_team or home_from_slug

    markets = [_parse_market(m) for m in event.get("markets", [])]
    markets = [m for m in markets if m is not None]

    moneyline_candidates = [m for m in markets if m.sports_market_type == MONEYLINE_TYPE]
    spread_candidates = [m for m in markets if m.sports_market_type == SPREAD_TYPE]
    totals_candidates = [m for m in markets if m.sports_market_type == TOTALS_TYPE]
    dropped = [m for m in markets if m.sports_market_type not in FULL_GAME_TYPES]

    moneyline = max(moneyline_candidates, key=lambda m: m.volume, default=None)
    if len(moneyline_candidates) > 1:
        flags.append(f"multiple_moneyline_markets({len(moneyline_candidates)})")

    spread = max(spread_candidates, key=lambda m: m.volume, default=None)
    if len(spread_candidates) > 1:
        flags.append(f"multiple_spread_lines({len(spread_candidates)})_picked_highest_volume")
    elif not spread_candidates:
        flags.append("missing_spread_market")

    totals = max(totals_candidates, key=lambda m: m.volume, default=None)
    if len(totals_candidates) > 1:
        flags.append(f"multiple_totals_lines({len(totals_candidates)})_picked_highest_volume")
    elif not totals_candidates:
        flags.append("missing_totals_market")

    if moneyline is None:
        flags.append("MISSING_MONEYLINE_MARKET")

    return GameMarkets(
        event_slug=slug,
        event_id=event.get("id"),
        away_team=away_team,
        home_team=home_team,
        game_date=date_from_slug,
        moneyline=moneyline,
        spread=spread,
        totals=totals,
        dropped=dropped,
        flags=flags,
    )


def parse_spread_line(question: str) -> Optional[dict[str, Any]]:
    """
    Parse a spread market question like "Spread: Spurs (-5.5)" into
    {"team": "Spurs", "line": -5.5}.
    """
    m = re.search(r"Spread:\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)", question)
    if not m:
        return None
    return {"team": m.group(1).strip(), "line": float(m.group(2))}


def parse_totals_line(question: str) -> Optional[float]:
    """Parse a totals market question like "Knicks vs. Spurs: O/U 217.5" -> 217.5."""
    m = re.search(r"O/U\s*(\d+(?:\.\d+)?)", question)
    if not m:
        return None
    return float(m.group(1))


def parse_game_start_time(record: MarketRecord) -> Optional[pd.Timestamp]:
    """
    Parse gameStartTime (preferred) falling back to startDate. Returns a
    tz-aware UTC Timestamp, or None if both are unparseable/missing.
    """
    for raw in (record.game_start_time, record.start_date):
        if not raw:
            continue
        try:
            ts = pd.Timestamp(raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        except (ValueError, TypeError):
            continue
    return None
