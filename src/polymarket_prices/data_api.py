"""
Data API access: historical trades per market.

Schema verified empirically on 2026-07-08 against
https://data-api.polymarket.com/trades?market=<conditionId>&limit=500&offset=0

Confirmed fields per trade: proxyWallet, side, asset, conditionId, size,
price, timestamp (Unix seconds), title, slug, eventSlug, outcome,
outcomeIndex, transactionHash (plus profile display metadata: name,
pseudonym, bio, profileImage, profileImageOptimized, icon - all wallet/UI
metadata, not needed for price-series reconstruction).

Confirmed behavior (round 1, 5-game test):
- Trades are returned NEWEST-FIRST (descending timestamp).
- The endpoint enforces a hard offset cap: requests with offset > 3000
  return HTTP 400 {"error":"max historical activity offset of 3000
  exceeded"}.

Confirmed behavior (round 2, Step 1 undocumented-capability probe - see
docs/polymarket_trades_api_step1_findings.md for the full writeup):
- No ascending-order option exists (`sort`/`order`/`orderBy`/`sortDirection`/
  `ascending`/`sortOrder` are all silently ignored - identical results with
  or without).
- No time-window filter exists (`from`/`to`/`before`/`after`/`startTs`/
  `endTs`/`timestamp_gt`/`timestamp_gte`/`minTimestamp`/`maxTimestamp` all
  silently ignored).
- `asset` (single outcome token), `filterType` (a real enum: CASH/TOKENS -
  invalid values error, but valid ones are no-ops), and `takerOnly` (a real
  bool param - invalid values error, but true/false are no-ops) do NOT
  partition trades or grant separate offset budgets.
- `side` (BUY/SELL) DOES genuinely filter, and **each side gets its own
  independent ~3000-3500-offset budget**. The two sides are disjoint (no
  overlap). In every NBA moneyline market tested (including the highest-
  volume capped game in the 50-game batch, $6.85M), one side (typically
  SELL - the minority side, since bettors overwhelmingly buy the outcome
  they favor rather than sell it) exhausts naturally under its own cap and
  reaches back through all of pre-game history, even when BUY alone is
  still capped mid-game. Fetching both sides and merging therefore restores
  full pre-game + in-game coverage without needing the Goldsky subgraph
  fallback. This is now the default (only) fetch strategy below.

Storage: raw trade dumps are cached gzip-compressed, trimmed to only the
fields this pipeline actually uses (asset/price/size/side/timestamp/outcome/
outcomeIndex). `conditionId`, `proxyWallet`, `transactionHash`, and all
profile/display/title/slug metadata are dropped before caching - the
conditionId is already the cache key (recoverable from the filename) and the
rest is wallet-display metadata never used by the analysis. This matters
because this cache is meant to scale to a full season (~1,230 games) without
becoming unreasonably large.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from .http_utils import get_json, read_json_gz, write_json_gz

logger = logging.getLogger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"
PAGE_LIMIT = 500
OFFSET_CAP = 3000  # empirically confirmed hard cap on this endpoint, PER SIDE
SIDES = ("BUY", "SELL")
# Bump this when the cache payload shape changes, so stale pre-side-split
# cache entries (single unfiltered pagination, no per-side cap info) are
# transparently treated as missing and refetched rather than silently
# returned in the old, potentially-capped shape.
CACHE_SCHEMA_VERSION = 2

# Fields we keep from each raw trade record; everything else (proxyWallet,
# conditionId, transactionHash, title, slug, eventSlug, icon, name,
# pseudonym, bio, profileImage, profileImageOptimized, ...) is dropped
# before caching to keep the raw cache lean at full-season scale.
_KEEP_FIELDS = ("asset", "price", "size", "side", "timestamp", "outcome", "outcomeIndex")


def _trim_trade(t: Dict[str, Any]) -> Dict[str, Any]:
    return {k: t[k] for k in _KEEP_FIELDS if k in t}


def _trades_cache_path(raw_dir: str, condition_id: str) -> str:
    safe_id = condition_id.replace("0x", "")
    return os.path.join(raw_dir, "trades", f"{safe_id}.json.gz")


MAX_SAFETY_OFFSET = 20000  # sanity guard only; real per-side cap is ~3000-3500


def _fetch_side_all(condition_id: str, side: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Paginate a single `side` (BUY or SELL) to exhaustion or its own offset
    cap. Each side gets an independent ~3000-3500-offset budget (see
    docs/polymarket_trades_api_step1_findings.md), so splitting the query
    this way roughly doubles usable coverage vs. a single unfiltered
    pagination - and in every NBA moneyline market tested, the minority
    side (typically SELL) exhausted naturally, recovering full pre-game
    history even when the majority side alone was still capped.
    """
    trades: List[Dict[str, Any]] = []
    offset = 0
    capped = False

    while True:
        try:
            page = get_json(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": PAGE_LIMIT, "offset": offset, "side": side},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                logger.warning(
                    f"Offset cap hit for {condition_id} side={side} at offset={offset} "
                    f"({len(trades)} trades fetched so far)"
                )
                capped = True
                break
            raise

        if not page:
            break

        trades.extend(_trim_trade(t) for t in page)
        offset += PAGE_LIMIT

        if len(page) < PAGE_LIMIT:
            break  # exhausted naturally

        if offset > MAX_SAFETY_OFFSET:
            logger.error(
                f"Safety offset limit ({MAX_SAFETY_OFFSET}) hit for {condition_id} "
                f"side={side}; treating as capped (this should never happen given "
                f"the observed ~3000-3500 cap - investigate if it does)."
            )
            capped = True
            break

    return trades, capped


def _cache_meta_fields(cached: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "capped": cached["capped"],
        "buy_capped": cached.get("buy_capped"),
        "sell_capped": cached.get("sell_capped"),
        "earliest_trade_ts": cached.get("earliest_trade_ts"),
        "buy_count": cached.get("buy_count"),
        "sell_count": cached.get("sell_count"),
    }


def fetch_all_trades(
    condition_id: str, raw_dir: str, force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch the full trade history for a market by paginating BUY and SELL
    separately (see _fetch_side_all) and merging, caching the merged,
    trimmed, time-sorted result to disk gzip-compressed.

    Returns (trades, meta) where `trades` is sorted ascending by timestamp
    and `meta` is a dict with:
      - capped: True if EITHER side hit its own offset cap (informational;
        does not by itself mean coverage is incomplete - see was_capped vs.
        pregame_coverage in pipeline.py)
      - buy_capped / sell_capped: per-side cap flags
      - earliest_trade_ts: timestamp of the earliest trade across both sides
      - buy_count / sell_count: trades fetched per side
    """
    cache_path = _trades_cache_path(raw_dir, condition_id)
    if not force_refresh and os.path.exists(cache_path):
        cached = read_json_gz(cache_path)
        if cached.get("schema") == CACHE_SCHEMA_VERSION:
            return cached["trades"], _cache_meta_fields(cached)
        logger.info(f"Stale cache schema for {condition_id} (pre-side-split fetch); refetching")

    buy_trades, buy_capped = _fetch_side_all(condition_id, "BUY")
    sell_trades, sell_capped = _fetch_side_all(condition_id, "SELL")

    trades = buy_trades + sell_trades
    trades.sort(key=lambda t: t["timestamp"])

    earliest_ts = trades[0]["timestamp"] if trades else None
    meta = {
        "capped": buy_capped or sell_capped,
        "buy_capped": buy_capped,
        "sell_capped": sell_capped,
        "earliest_trade_ts": earliest_ts,
        "buy_count": len(buy_trades),
        "sell_count": len(sell_trades),
    }

    write_json_gz(cache_path, {"schema": CACHE_SCHEMA_VERSION, "trades": trades, **meta})

    logger.info(
        f"Fetched {len(trades)} trades for market {condition_id} "
        f"(buy={len(buy_trades)} capped={buy_capped}, sell={len(sell_trades)} capped={sell_capped})"
    )
    return trades, meta


def _fetch_side_until_before(
    condition_id: str, side: str, before_ts: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """Same early-stop-once-we-cross-before_ts logic as before, per side."""
    trades: List[Dict[str, Any]] = []
    offset = 0
    capped = False

    while True:
        try:
            page = get_json(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": PAGE_LIMIT, "offset": offset, "side": side},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                capped = True
                break
            raise

        if not page:
            break

        trades.extend(_trim_trade(t) for t in page)

        if any(t["timestamp"] < before_ts for t in page):
            break  # we've crossed tip-off; we have enough to find the pre-game close

        offset += PAGE_LIMIT
        if len(page) < PAGE_LIMIT:
            break
        if offset > MAX_SAFETY_OFFSET:
            capped = True
            break

    return trades, capped


def fetch_trades_until_before(
    condition_id: str, before_ts: int, raw_dir: str, force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Bounded fetch for pre-game-snapshot-only use (spread/totals markets):
    page BUY and SELL separately from the most recent trade backward in
    time, each stopping as soon as it crosses before_ts. Does not fetch the
    full series.

    Returns (trades_fetched_so_far, capped) where capped is True only if
    NO pre-game trade was found on either side AND at least one side hit
    its own offset cap while looking (i.e. genuinely couldn't determine the
    pre-game price, as opposed to the market simply having no pre-game
    trades at all).
    """
    cache_path = _trades_cache_path(raw_dir, condition_id)
    if not force_refresh and os.path.exists(cache_path):
        cached = read_json_gz(cache_path)
        if cached.get("schema") == CACHE_SCHEMA_VERSION:
            return cached["trades"], cached["capped"]

    buy_trades, buy_capped = _fetch_side_until_before(condition_id, "BUY", before_ts)
    sell_trades, sell_capped = _fetch_side_until_before(condition_id, "SELL", before_ts)

    trades = buy_trades + sell_trades
    trades.sort(key=lambda t: t["timestamp"])

    found_pregame = any(t["timestamp"] < before_ts for t in trades)
    capped = (not found_pregame) and (buy_capped or sell_capped)

    write_json_gz(
        cache_path,
        {
            "schema": CACHE_SCHEMA_VERSION,
            "trades": trades,
            "capped": capped,
            "buy_capped": buy_capped,
            "sell_capped": sell_capped,
        },
    )

    return trades, capped
