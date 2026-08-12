"""
Shared HTTP helpers for the Polymarket price-history pipeline.

Provides:

- A polite, rate-limited GET with exponential backoff on 429/5xx.
- Simple on-disk JSON response caching so re-runs don't re-download data
  that's already been fetched (resumability requirement).

No authentication is required for any of the Polymarket public APIs used here
(Gamma, Data API, CLOB).
"""

import gzip
import json
import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Be polite: cap request rate across all callers in this process.
MIN_REQUEST_INTERVAL_SECONDS = 0.15  # ~6-7 req/s max
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
REQUEST_TIMEOUT = 20

_last_request_time = 0.0


def _throttle() -> None:
    global _last_request_time
    now = time.monotonic()
    wait = MIN_REQUEST_INTERVAL_SECONDS - (now - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def get_json(url: str, params: Optional[dict[str, Any]] = None) -> Any:
    """
    Rate-limited GET with retry + exponential backoff on 429/5xx.

    Raises requests.HTTPError for non-retryable failures (e.g. 400) so
    callers can decide how to handle them (Polymarket uses 400 for things
    like "offset cap exceeded", which is an expected, handled condition).
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(f"Request error ({e}); retrying in {sleep_s:.1f}s ({url})")
            time.sleep(sleep_s)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429 or resp.status_code >= 500:
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                f"HTTP {resp.status_code} from {url}; retrying in {sleep_s:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(sleep_s)
            continue

        # Non-retryable (4xx other than 429). Let the caller decide -
        # raise_for_status gives them the status code + body via the exception.
        resp.raise_for_status()

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Exhausted retries for {url}")


def cached_get_json(
    cache_path: str,
    url: str,
    params: Optional[dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Any:
    """
    Fetch JSON from `url`, caching the raw response gzip-compressed at
    `cache_path` (expected to end in .json.gz - raw JSON compresses very
    well, and this matters once caching is done across a full season).

    If the cache file already exists (and force_refresh is False), it is
    loaded from disk instead of hitting the network at all.
    """
    if not force_refresh and os.path.exists(cache_path):
        return read_json_gz(cache_path)

    data = get_json(url, params=params)
    write_json_gz(cache_path, data)
    return data


def read_json_gz(path: str) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
