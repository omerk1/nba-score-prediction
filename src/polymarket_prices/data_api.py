"""
Data API access: historical trades per market.

Schema verified empirically on 2026-07-08 against
https://data-api.polymarket.com/trades?market=<conditionId>&limit=500&offset=0

Confirmed fields per trade: proxyWallet, side, asset, conditionId, size,
price, timestamp (Unix seconds), title, slug, eventSlug, outcome,
outcomeIndex, transactionHash (plus profile display metadata: name,
pseudonym, bio, profileImage, profileImageOptimized, icon - all wallet/UI
metadata, not needed for price-series reconstruction).

Confirmed behavior:
- Trades are returned NEWEST-FIRST (descending timestamp).
- The endpoint enforces a hard offset cap: requests with offset > 3000
  return HTTP 400 {"error":"max historical activity offset of 3000
  exceeded"}. This was hit on a high-volume NBA Finals market
  ($18.7M volume); several lower-volume regular-season markets (~$450K-770K
  volume, 1600-2600 total trades) paginated to exhaustion well under the cap
  with no issue, but one ~$715K-volume game also hit the cap (trade *count*,
  not just $ volume, drives this - many small fills). Per the design doc,
  when the cap is hit we record how many trades were fetched vs. expected
  and set a `trades_capped` flag rather than silently truncating; we do NOT
  fall back to the Goldsky subgraph in this pass (design doc: only go there
  "if the Data API proves insufficient" - not needed for the 5-game
  small-scale test, which was deliberately chosen to avoid the cap).
- `after`/`before` timestamp query params were tried and are NOT respected
  by this endpoint (identical results with/without them) - so time-windowed
  pagination is not currently possible; only offset-based.
- An `asset` (single outcome token) filter param was tried and does NOT
  filter as hoped either (returned unrelated trades) - `market` (conditionId)
  is the only working filter.

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
from typing import Any, Dict, List, Tuple

import requests

from .http_utils import get_json, read_json_gz, write_json_gz

logger = logging.getLogger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"
PAGE_LIMIT = 500
OFFSET_CAP = 3000  # empirically confirmed hard cap on this endpoint

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


def fetch_all_trades(
    condition_id: str, raw_dir: str, force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch the full trade history for a market (paginating until exhausted or
    the offset cap is hit), caching the merged, trimmed result to disk
    gzip-compressed.

    Returns (trades, capped) where `trades` is newest-first (matching API
    order) and `capped` is True if the 3000-offset cap was hit before
    pagination naturally exhausted (i.e. older trades may be missing).
    """
    cache_path = _trades_cache_path(raw_dir, condition_id)
    if not force_refresh and os.path.exists(cache_path):
        cached = read_json_gz(cache_path)
        return cached["trades"], cached["capped"]

    trades: List[Dict[str, Any]] = []
    offset = 0
    capped = False

    while True:
        try:
            page = get_json(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": PAGE_LIMIT, "offset": offset},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                logger.warning(
                    f"Offset cap hit for {condition_id} at offset={offset} "
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

        if offset >= OFFSET_CAP:
            # Next request would exceed the cap; stop here to avoid the
            # guaranteed 400 (we already know the cap from testing).
            logger.warning(
                f"Reached offset cap ({OFFSET_CAP}) for {condition_id}; "
                f"{len(trades)} trades fetched, older trades may be missing."
            )
            capped = True
            break

    write_json_gz(cache_path, {"trades": trades, "capped": capped})

    logger.info(f"Fetched {len(trades)} trades for market {condition_id} (capped={capped})")
    return trades, capped


def fetch_trades_until_before(
    condition_id: str, before_ts: int, raw_dir: str, force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Bounded fetch for pre-game-snapshot-only use (spread/totals markets):
    page from the most recent trade backward in time, stopping as soon as
    we've collected at least one trade with timestamp < before_ts (i.e. we
    have what we need to find the last pre-game trade). Does not fetch the
    full series.

    Returns (trades_fetched_so_far, capped).
    """
    cache_path = _trades_cache_path(raw_dir, condition_id)
    if not force_refresh and os.path.exists(cache_path):
        cached = read_json_gz(cache_path)
        return cached["trades"], cached["capped"]

    trades: List[Dict[str, Any]] = []
    offset = 0
    capped = False

    while True:
        try:
            page = get_json(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": PAGE_LIMIT, "offset": offset},
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
        if offset >= OFFSET_CAP:
            capped = True
            break

    write_json_gz(cache_path, {"trades": trades, "capped": capped})

    return trades, capped
