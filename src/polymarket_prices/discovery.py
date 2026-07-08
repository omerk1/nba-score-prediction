"""
Bulk NBA game discovery via the Gamma API's /events listing endpoint.

Companion to gamma.py's single-slug fetch_event_by_slug (which requires
already knowing the slug this module answers "which slugs exist?".

Schema verified empirically against https://gamma-api.polymarket.com/events
on 2026-07-08:

- `tag_slug=nba&closed=true` returns NBA-tagged events that are closed. For
  actual completed games this reliably means fully RESOLVED (outcomePrices
  is ["1","0"] / ["0","1"] on the moneyline market -- confirmed against
  nba-orl-chi-2026-04-10, one of the 5-game checkpoint's games), not merely
  "no longer trading". There is no separate "resolved" boolean on the event;
  `closed` is the field to use.
- The `tag_slug=nba` result set also includes a lot of non-game NBA content
  (free-agency / "who will X sign with" prop markets, award futures, etc.)
  whose slugs don't match the `nba-{away}-{home}-{date}` game pattern --
  these are filtered out here via the same slug regex gamma.py uses
  (gamma.SLUG_RE), so callers only ever get real game slugs back.
- `start_date_min` / `start_date_max` (ISO8601, e.g. "2026-03-01T00:00:00Z")
  ARE respected server-side (verified: narrowing the window changes the
  result set, not just what's echoed back in the params) -- despite
  data_api.py's finding that the /trades endpoint's after/before params are
  silently ignored, this is a *different* endpoint/service and behaves
  differently. This is what makes it possible to target a specific/recent
  date range instead of paginating from the oldest available event.
- IMPORTANT: an event's `startDate` is roughly when the market was
  created/listed (empirically ~1-9 days before the actual game), NOT the
  game date itself. The game date must be read from the slug
  (`nba-{away}-{home}-{YYYY-MM-DD}`), same as gm.game_date elsewhere in this
  package. Don't assume `start_date_min`/`start_date_max` line up exactly
  with game dates -- widen the window a bit beyond your target game-date
  range, then filter/sort on the slug-embedded date.
- `order=startDate&ascending=true|false` works and gives stable,
  chronologically-ordered pages (asc = oldest-listed first).
- Standard `limit`/`offset` pagination works as expected; a page shorter
  than `limit` means the result set is exhausted. No hard offset cap was
  observed on this endpoint (unlike the Data API's /trades 3000 cap).
"""

import logging
from typing import List, Optional

from .gamma import SLUG_RE
from .http_utils import get_json

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
PAGE_LIMIT = 100


def discover_nba_game_slugs(
    start_date_min: Optional[str] = None,
    start_date_max: Optional[str] = None,
    closed: bool = True,
    ascending: bool = True,
    max_pages: int = 200,
) -> List[str]:
    """
    Paginate the Gamma /events listing to enumerate NBA game slugs.

    start_date_min/start_date_max: ISO8601 strings filtering on event
    `startDate` (market listing time, NOT game date -- see module
    docstring). Since listing typically happens ~1-9 days before the actual
    game, widen the window a bit beyond your target game-date range and
    filter/sort on the slug-embedded date afterward.

    closed=True (default) restricts to fully resolved games, which is what
    this pipeline needs (unresolved games have no winner to normalize prices
    against).

    Returns game slugs (format nba-{away}-{home}-{date}) in the order
    returned by the API (startDate order, ascending or descending per
    `ascending`), deduplicated. Non-game NBA events (futures, props, etc.)
    are dropped via slug-pattern match against gamma.SLUG_RE.
    """
    slugs: List[str] = []
    seen = set()
    offset = 0

    for _ in range(max_pages):
        params = {
            "tag_slug": "nba",
            "closed": str(closed).lower(),
            "limit": PAGE_LIMIT,
            "offset": offset,
            "order": "startDate",
            "ascending": str(ascending).lower(),
        }
        if start_date_min:
            params["start_date_min"] = start_date_min
        if start_date_max:
            params["start_date_max"] = start_date_max

        page = get_json(f"{GAMMA_BASE}/events", params=params)
        if not page:
            break

        new_count = 0
        for e in page:
            slug = e.get("slug", "")
            if SLUG_RE.match(slug) and slug not in seen:
                seen.add(slug)
                slugs.append(slug)
                new_count += 1

        logger.info(
            f"discover_nba_game_slugs: offset={offset} -> {len(page)} events, "
            f"{new_count} new game slugs ({len(slugs)} total so far)"
        )

        if len(page) < PAGE_LIMIT:
            break  # exhausted naturally
        offset += PAGE_LIMIT
    else:
        logger.warning(f"discover_nba_game_slugs: hit max_pages={max_pages} without exhausting results")

    return slugs
