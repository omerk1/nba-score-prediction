"""
Daily data refresh orchestrator: keeps the three stores the live serving
path reads current through yesterday's games — the `game` table
(fetch_data), the style-fingerprint cache (precompute_scores), and the
injury features DB (nightly ESPN update). Scope and store-by-store
reasoning: docs/features/serving/daily_data_refresh_scope.md.

Each underlying refresh is already incremental and idempotent, so this
module only sequences them, adds retries (3 attempts, linear backoff —
the backfill scripts' established pattern), and ends with a freshness
gate on the one store that goes stale silently: a failed game fetch
leaves no error behind (fetch_data.main logs and continues), it just
quietly shifts every rolling window and Elo rating. Rerunning the whole
thing is the recovery mechanism for any partial failure — no state to
clean up.

The downstream daily recommendation job should treat a non-zero exit
from this job as "don't trust today's features": skip or caveat the
send rather than assume the stores are current.
"""

import datetime
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5  # linear backoff: 5s, 10s
SCOREBOARD_SLEEP_SECONDS = 0.7  # same stats.nba.com courtesy sleep as fetch_data
# How many days back the freshness gate will query the scoreboard for
# scheduled games before giving up and assuming an off-period. Caps the
# offseason cost (every checked day is one ScoreboardV2 call); a gap
# older than this that still contains missed games goes undetected, but
# any in-season fetch failure surfaces within a day.
FRESHNESS_LOOKBACK_DAYS = 7

# NBA game dates are US/Eastern calendar days; "yesterday" must be
# computed there, not in the machine's local zone — an evening run in
# Israel is still the same Eastern date, but a small-hours local run
# would otherwise demand games that haven't finished yet.
NBA_TZ = ZoneInfo("US/Eastern")


def _today_nba() -> datetime.date:
    return datetime.datetime.now(NBA_TZ).date()


def _season_for(d: datetime.date) -> str:
    """2026-10-04 → '2026-27'; dates before October belong to the previous season."""
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def _retry(step_name: str, fn: Callable[[], object]) -> object:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_WAIT_SECONDS * attempt
            logger.warning(f"{step_name}: attempt {attempt} failed ({e}); retry in {wait}s")
            time.sleep(wait)


# Step imports are deferred to call time: the injury chain reads
# GOOGLE_API_KEY at module import (see scripts/build_injury_features.py's
# load_dotenv-before-import note), so the CLI wrapper must get a chance
# to load .env first — and tests monkeypatch these module attributes
# without pulling the heavy import chains at all.


def refresh_games() -> None:
    """Incremental fetch into the `game` table (2 LeagueGameLog calls in-season)."""
    from src.data_processing import fetch_data

    fetch_data.main()


def refresh_style_cache() -> None:
    """Rebuild the style-fingerprint cache; its docstring already prescribes
    running right after each fetch_data refresh."""
    from src.matchups.precompute_scores import precompute_and_cache

    precompute_and_cache()


def refresh_injuries() -> None:
    """Today's ESPN injury report → (game_date, team_id) impact rows."""
    from src.news_scraping.pipeline import run_nightly

    # Explicit Eastern date: run_nightly defaults to the machine-local
    # calendar day, and the serving path's injury join is exact-date — a
    # small-hours local run would otherwise file tonight's report one day
    # ahead of the Eastern game night and zero out the live injury signal.
    run_nightly(_today_nba())


def refresh_player_importance() -> None:
    """Current-season player_importance snapshot (weekly cadence is enough —
    scoring asof-joins the latest snapshot, so staleness degrades gently).
    backfill_season skips already-stored dates, so re-running is cheap."""
    from src.news_scraping.player_importance import backfill_season

    today_nba = _today_nba()
    backfill_season(_season_for(today_nba), end=today_nba)


def should_run_importance_snapshot(today: Optional[datetime.date] = None) -> bool:
    """Weekly, pinned to Monday, so the daily job needs no state file to
    remember when it last ran."""
    today = today or _today_nba()
    return today.weekday() == 0


@dataclass
class FreshnessResult:
    fresh: bool
    reason: str


def check_game_table_freshness(
    db_path: Optional[Path] = None,
    today: Optional[datetime.date] = None,
    lookback_days: int = FRESHNESS_LOOKBACK_DAYS,
) -> FreshnessResult:
    """Fail only when games were actually scheduled after MAX(game_date):
    walks back from yesterday (Eastern) querying the scoreboard until it
    finds a scheduled date (→ stale), reaches the last stored date, or
    exhausts the lookback cap (→ assume off-period). The common in-season
    cases cost 0 extra API calls (table current) or 1 (stale, games
    yesterday)."""
    db_path = db_path or Path(load_config().data_paths.raw_db)
    today = today or _today_nba()
    yesterday = today - datetime.timedelta(days=1)

    if not db_path.exists():
        return FreshnessResult(False, f"game DB missing at {db_path}")
    # Read-only URI: a bare connect() would create an empty DB file as a
    # side effect of a check that must never mutate state.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(game_date) FROM game").fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return FreshnessResult(False, "game table is empty")

    last_stored = datetime.date.fromisoformat(row[0][:10])
    if last_stored >= yesterday:
        return FreshnessResult(True, f"game table current through {last_stored}")

    from src.data_processing.fetch_data import fetch_upcoming_games

    earliest = max(
        last_stored + datetime.timedelta(days=1),
        yesterday - datetime.timedelta(days=lookback_days - 1),
    )
    d = yesterday
    while d >= earliest:
        # Same retry budget as the refresh steps — a single transient
        # scoreboard error must not fail an otherwise-successful run.
        scheduled = _retry(
            f"scoreboard check {d}", lambda d=d: fetch_upcoming_games(d.isoformat())
        )
        # Only count game types the table can actually contain: fetch_data
        # pulls Regular Season + Playoffs, but the scoreboard also lists
        # preseason (GAME_ID prefix 001) and All-Star (003) — demanding
        # those would fail the gate every preseason day on games that
        # LeagueGameLog will never return (observed live 2026-10-06).
        # Prefixes: 001 preseason, 002 regular season, 003 All-Star,
        # 004 playoffs, 005 play-in.
        if not scheduled.empty:
            scheduled = scheduled[
                scheduled["game_id"].astype(str).str.startswith(("002", "004"))
            ]
        if not scheduled.empty:
            return FreshnessResult(
                False,
                f"game table ends {last_stored} but {len(scheduled)} game(s) "
                f"were scheduled on {d}",
            )
        d -= datetime.timedelta(days=1)
        if d >= earliest:
            time.sleep(SCOREBOARD_SLEEP_SECONDS)
    return FreshnessResult(
        True,
        f"no scheduled games found between {earliest} and {yesterday} "
        f"(last stored {last_stored}; lookback capped at {lookback_days}d)",
    )


def run_daily_update(include_importance_snapshot: bool = False) -> int:
    """Run every refresh step (later steps still run if an earlier one
    fails — each degrades gracefully and independently), then the
    freshness gate. Returns a process exit code: 0 only if every step
    succeeded AND the game table is verifiably current."""
    steps: list[tuple[str, Callable[[], None]]] = [
        ("game table", refresh_games),
        ("style fingerprint cache", refresh_style_cache),
        ("injury nightly update", refresh_injuries),
    ]
    if include_importance_snapshot:
        steps.append(("player importance snapshot", refresh_player_importance))

    failures: list[str] = []
    for name, fn in steps:
        try:
            _retry(name, fn)
            logger.info(f"{name}: ok")
        except Exception as e:
            logger.error(f"{name}: failed after {MAX_RETRIES} attempts: {e}")
            failures.append(name)

    try:
        freshness = check_game_table_freshness()
    except Exception as e:
        # Can't verify = don't claim fresh; the downstream job must not
        # send recommendations on an unverifiable game table.
        freshness = FreshnessResult(False, f"freshness check failed: {e}")

    if failures:
        logger.error(f"daily update: {len(failures)} step(s) failed: {', '.join(failures)}")
    logger.info(f"freshness gate: {'PASS' if freshness.fresh else 'FAIL'} — {freshness.reason}")
    return 0 if not failures and freshness.fresh else 1
