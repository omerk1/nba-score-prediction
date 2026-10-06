"""
The daily end-to-end job launchd invokes at 16:00 Israel time: data
refresh -> Winner capture -> vision extraction -> recommendations ->
Telegram. Chains the already-built pieces; every design decision here
comes from the serving scope docs (docs/features/serving/
daily_scheduling_scope.md and siblings).

Operational contract:
- Telegram is the dead-man's switch: the job always tries to say
  SOMETHING on the configured channel — recommendations, "no NBA games
  today", or a failure notice. Silence means the machine didn't wake or
  the job never ran; check logs/ and `launchctl print`.
- A failed data refresh (or freshness gate) aborts BEFORE capture:
  recommendations built on silently stale rolling windows are worse
  than no recommendations.
- A single game failing to recommend does not kill the slate — it's
  reported alongside whatever succeeded.
- A lockfile guards against a wake-coalesced run overlapping a manual
  one; the second runner exits 0 quietly (the work is being done).

Step imports are deferred to call time, same pattern (and same .env
ordering reason) as daily_update.
"""

import datetime
import fcntl
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_PATH = Path("outputs/daily_job.lock")
LOG_DIR = Path("logs")
LOG_RETENTION_DAYS = 30
_LOG_NAME = re.compile(r"^daily_job_(\d{4}-\d{2}-\d{2})\.log$")


def acquire_lock(lock_path: Path = LOCK_PATH):
    """Non-blocking flock; returns the open handle (keep it alive for the
    run) or None when another run holds it."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close()
        return None


def prune_old_logs(
    log_dir: Path = LOG_DIR, retention_days: int = LOG_RETENTION_DAYS, today=None
) -> int:
    """The launchd plist writes one dated log per run day; nothing else
    rotates them (the scheduling scope's no-sudo alternative to
    newsyslog), so the job prunes its own."""
    today = today or datetime.date.today()
    cutoff = today - datetime.timedelta(days=retention_days)
    removed = 0
    if not log_dir.exists():
        return 0
    for f in log_dir.iterdir():
        m = _LOG_NAME.match(f.name)
        if m and datetime.date.fromisoformat(m.group(1)) < cutoff:
            f.unlink()
            removed += 1
    return removed


# --- steps, as module attributes so tests monkeypatch the seams ---


def refresh_data() -> int:
    from src.data_processing.daily_update import run_daily_update, should_run_importance_snapshot

    return run_daily_update(include_importance_snapshot=should_run_importance_snapshot())


def capture() -> Path:
    from src.serving.capture_winner import capture_nba_page

    return capture_nba_page()


def extract(screenshot_path: Path) -> list[dict]:
    from src.serving.extract_picks import extract_picks_from_screenshot

    return extract_picks_from_screenshot(str(screenshot_path))


def recommend_all(picks: list[dict]) -> tuple[list[dict], list[str]]:
    """Per-pick isolation: early-season games without enough history (or
    any one bad extraction) must not take down the rest of the slate."""
    from src.serving.recommend import load_resources, recommend_pick

    resources = load_resources()
    recs, errors = [], []
    for pick in picks:
        try:
            recs.append(recommend_pick(pick, resources))
        except Exception as e:
            label = f"{pick.get('home_team_id', '?')} vs {pick.get('away_team_id', '?')}"
            logger.error(f"recommendation failed for {label}: {e}")
            errors.append(f"{label}: {e}")
    return recs, errors


def notify(recs: list[dict]):
    from src.serving.notify_telegram import send_recommendations

    return send_recommendations(recs)


def notice(text: str):
    from src.serving.notify_telegram import send_notice

    logger.info(f"notice: {text}")
    return send_notice(text)


def run_daily_job() -> int:
    # Module globals resolved at call time (not frozen defaults) so tests
    # can repoint LOCK_PATH/LOG_DIR.
    lock = acquire_lock(LOCK_PATH)
    if lock is None:
        logger.info("another daily job run holds the lock — exiting quietly")
        return 0
    try:
        prune_old_logs(LOG_DIR)

        if refresh_data() != 0:
            notice(
                "Daily job: data refresh or freshness gate FAILED — no "
                "recommendations today (stale features). Check logs/."
            )
            return 1

        try:
            screenshot = capture()
        except Exception as e:
            notice(f"Daily job: Winner capture FAILED: {e}")
            return 1

        try:
            picks = extract(screenshot)
        except Exception as e:
            notice(f"Daily job: extraction FAILED on {screenshot}: {e}")
            return 1

        if not picks:
            notice("Daily job: no NBA games on Winner today.")
            return 0

        recs, errors = recommend_all(picks)
        if not recs:
            notice(f"Daily job: all {len(picks)} game(s) FAILED to recommend. Check logs/.")
            return 1

        from src.serving.notify_telegram import SendStatus

        status = notify(recs)
        if errors:
            notice(f"Daily job note: {len(errors)} game(s) skipped — {'; '.join(errors)}")
        if status is SendStatus.FAILED:
            logger.error("Telegram send failed — recommendations are in the log above")
            return 1
        return 0
    finally:
        lock.close()
