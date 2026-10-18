"""
The daily end-to-end job launchd invokes at 16:00 Israel time: data
refresh -> Winner lines capture -> parse + schedule gate -> recommendations ->
prediction log -> Telegram. Chains the already-built pieces; every
design decision here comes from the serving scope docs (docs/features/serving/
daily_scheduling_scope.md and siblings).

Operational contract:
- Telegram is the dead-man's switch: the job always tries to say
  SOMETHING on the configured channel — recommendations, "no NBA games
  today", a failure notice naming the failed stage, or a crash notice
  from the top-level guard. Silence means the machine didn't wake or
  the job never ran; check logs/ and `launchctl print`. On a day whose
  only message is a notice, a failed send also fails the exit code, so
  `launchctl print`'s last-exit can't show success while the channel
  is silent.
- A failed data refresh (or freshness gate) aborts BEFORE capture:
  recommendations built on silently stale rolling windows are worse
  than no recommendations.
- A single game failing to recommend does not kill the slate — it's
  reported alongside whatever succeeded.
- A lockfile guards against a wake-coalesced run overlapping a manual
  one (second runner exits 0 quietly), and a dated success marker
  guards against the sequential version of the same event — launchd's
  coalesced 16:00 firing AFTER a manual run already finished — which
  would otherwise re-run the pipeline and duplicate the day's slate.
  `force=True` (CLI --force) bypasses the marker for deliberate
  re-runs. The marker records the date the run STARTED: a run that
  straddles midnight must not mark the next day done and skip its real
  16:00 run. A run started before 08:00 counts toward the previous day's
  slate (whose games run until early morning), so a manual recovery run
  after midnight doesn't mark the coming day done either.
- A scheduled run (CLI --scheduled, passed only by the launchd agent)
  that starts outside its 16:00-22:00 window — the lid-closed Mac slept
  through the 16:00 event and launchd fired it at some later wake — or
  only reaches the refresh or recommendation checkpoints after 22:00,
  sends a "missed" notice (unless that day already succeeded) instead of
  delivering a slate hours late (or a day late, under the wrong date).
  Manual runs are never gated.

Step imports are deferred to call time, same pattern (and same .env
ordering reason) as daily_update.
"""

import datetime
import fcntl
import logging
import re
from pathlib import Path

from src.serving.notify_telegram import SendStatus

logger = logging.getLogger(__name__)

LOCK_PATH = Path("outputs/daily_job.lock")
SUCCESS_MARKER = Path("outputs/daily_job_last_success")
LOG_DIR = Path("logs")
LOG_RETENTION_DAYS = 30
# The launchd StandardErrorPath fallback (pre-redirect failures) is the
# one log the dated-prune below can't rotate; cap it by size instead.
ERR_LOG_NAME = "launchd_daily_job.err.log"
ERR_LOG_MAX_BYTES = 5_000_000
_LOG_NAME = re.compile(r"^daily_job_(\d{4}-\d{2}-\d{2})\.log$")
# Scheduled-run window (local time). SCHEDULED_START must match the
# plist's StartCalendarInterval in scripts/install_launchd.sh. 22:00
# still leaves hours before the earliest regular tip-offs (~02:00 Israel
# time) while catching a run launchd fired at a late wake.
SCHEDULED_START = datetime.time(16, 0)
SCHEDULED_DEADLINE = datetime.time(22, 0)
# A slate's games run until ~07:00 Israel time the next morning, so a run
# started before this belongs to the previous day's slate (marker and
# prediction-log date), not the calendar day it started on.
SLATE_ROLLOVER = datetime.time(8, 0)


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


def prune_old_logs(log_dir: Path = LOG_DIR, retention_days: int = LOG_RETENTION_DAYS, today=None) -> int:
    """The launchd plist writes one dated log per run day; nothing else
    rotates them (the scheduling scope's no-sudo alternative to
    newsyslog), so the job prunes its own. A name that matches the
    pattern but isn't a real date (a stray test artifact) is skipped,
    not fatal — housekeeping must never take the job down. The err-log
    size cap lives here too, same rationale."""
    today = today or datetime.date.today()
    cutoff = today - datetime.timedelta(days=retention_days)
    removed = 0
    if not log_dir.exists():
        return 0
    for f in log_dir.iterdir():
        m = _LOG_NAME.match(f.name)
        if not m:
            continue
        try:
            file_date = datetime.date.fromisoformat(m.group(1))
        except ValueError:
            logger.warning(f"skipping unparseable log name during prune: {f.name}")
            continue
        if file_date < cutoff:
            f.unlink()
            removed += 1
    err_log = log_dir / ERR_LOG_NAME
    if err_log.exists() and err_log.stat().st_size > ERR_LOG_MAX_BYTES:
        logger.warning(f"truncating oversized {err_log} ({err_log.stat().st_size} bytes)")
        err_log.write_text("")
    return removed


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _already_succeeded(run_date: datetime.date) -> bool:
    try:
        return SUCCESS_MARKER.read_text().strip() == run_date.isoformat()
    except (OSError, ValueError):
        return False


def _mark_success(run_date: datetime.date) -> None:
    SUCCESS_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SUCCESS_MARKER.write_text(run_date.isoformat())


def _slate_date(started: datetime.datetime) -> datetime.date:
    if started.time() < SLATE_ROLLOVER:
        return started.date() - datetime.timedelta(days=1)
    return started.date()


def _missed_day(started: datetime.datetime) -> datetime.date | None:
    """The day whose scheduled run this start time missed, or None when
    it's inside the window. Before 16:00 means launchd is firing
    yesterday's event at a wake after midnight."""
    if started.time() < SCHEDULED_START:
        return started.date() - datetime.timedelta(days=1)
    if started.time() > SCHEDULED_DEADLINE:
        return started.date()
    return None


def _past_deadline(run_date: datetime.date) -> bool:
    return _now() > datetime.datetime.combine(run_date, SCHEDULED_DEADLINE)


def _missed_mid_run(run_date: datetime.date, stage: str) -> int:
    return _notice_exit(
        f"Daily job: the {run_date.isoformat()} run was MISSED — {stage} only finished at "
        f"{_now():%Y-%m-%d %H:%M} (Mac slept mid-run?). No recommendations.",
        ok=False,
    )


# --- steps, as module attributes so tests monkeypatch the seams ---


def refresh_data() -> int:
    from src.data_processing.daily_update import run_daily_update, should_run_importance_snapshot

    return run_daily_update(include_importance_snapshot=should_run_importance_snapshot())


def capture() -> Path:
    from src.serving.capture_winner import capture_nba_lines

    return capture_nba_lines()


def extract(snapshot_path: Path) -> list[dict]:
    from src.serving.extract_picks import match_to_slate
    from src.serving.schedule_lookup import upcoming_games
    from src.serving.winner_lines import picks_from_snapshot

    # Winner lists games days ahead: only games the NBA schedule has
    # tipping off in the next 24h. The schedule also sets home/away.
    return match_to_slate(picks_from_snapshot(snapshot_path), upcoming_games())


def recommend_all(picks: list[dict]) -> tuple[list[dict], list[str]]:
    """Per-pick isolation: early-season games without enough history (or
    any one bad extraction) must not take down the rest of the slate."""
    from src.serving.recommend import load_resources, recommend_pick

    resources = load_resources()
    recs, errors = [], []
    for pick in picks:
        try:
            recs.append(recommend_pick(pick, resources, game_date=pick.get("game_date")))
        except Exception as e:
            label = f"{pick.get('home_team_id', '?')} vs {pick.get('away_team_id', '?')}"
            logger.error(f"recommendation failed for {label}: {e}")
            errors.append(f"{label}: {e}")
    return recs, errors


def persist(recs: list[dict], run_date: datetime.date) -> None:
    from src.serving.prediction_log import save_predictions

    save_predictions(recs, game_date=run_date)


def notify(recs: list[dict]):
    from src.serving.notify_telegram import send_recommendations

    return send_recommendations(recs)


def notice(text: str):
    from src.serving.notify_telegram import send_notice

    logger.info(f"notice: {text}")
    return send_notice(text)


def _notice_exit(text: str, ok: bool) -> int:
    """A path whose ONLY output for the day is this notice: the send
    status must reach the exit code, or a dead channel (bad chat_id)
    looks like success from both monitoring directions at once."""
    status = notice(text)
    if status is SendStatus.FAILED:
        return 1
    return 0 if ok else 1


def _run(force: bool, scheduled: bool) -> int:
    started = _now()
    run_date = _slate_date(started)
    prune_old_logs(LOG_DIR, today=started.date())

    if scheduled:
        missed = _missed_day(started)
        if missed is not None and _already_succeeded(missed):
            # The coalesced event firing late after a manual run already
            # delivered that day's slate — nothing was missed.
            logger.info(f"{missed.isoformat()} already succeeded (marker) — late firing, skipping")
            return 0
        if missed is not None:
            return _notice_exit(
                f"Daily job: the {missed.isoformat()} run was MISSED — it only started at "
                f"{started:%Y-%m-%d %H:%M} (Mac asleep at 16:00?). No recommendations.",
                ok=False,
            )

    if not force and _already_succeeded(run_date):
        # launchd's coalesced 16:00 event firing after a manual (or
        # earlier) run already delivered today's slate — don't re-run
        # the pipeline and duplicate the message.
        logger.info("today's run already succeeded (marker) — skipping; use --force to re-run")
        return 0

    if refresh_data() != 0:
        return _notice_exit(
            "Daily job: data refresh or freshness gate FAILED — no "
            "recommendations today (stale features). Check logs/.",
            ok=False,
        )

    if scheduled and _past_deadline(run_date):
        return _missed_mid_run(run_date, "data refresh")

    try:
        screenshot = capture()
    except Exception as e:
        return _notice_exit(f"Daily job: Winner capture FAILED: {e}", ok=False)

    try:
        picks = extract(screenshot)
    except Exception as e:
        return _notice_exit(f"Daily job: extraction FAILED on {screenshot}: {e}", ok=False)

    if not picks:
        rc = _notice_exit("Daily job: no NBA games in the next 24h that are on Winner.", ok=True)
        if rc == 0:
            _mark_success(run_date)
        return rc

    recs, errors = recommend_all(picks)
    if not recs:
        return _notice_exit(
            f"Daily job: all {len(picks)} game(s) FAILED to recommend. Check logs/.", ok=False
        )

    # Second gate: capture and extraction can stall too (a re-sleep, a
    # hung screenshot), and a slate sent after tip-off is worse than none.
    if scheduled and _past_deadline(run_date):
        return _missed_mid_run(run_date, "recommendations")

    # Before notify, so a crash in the send can't lose the record — but a
    # storage failure must never cost the day's message either.
    persist_error = None
    try:
        persist(recs, run_date)
    except Exception as e:
        logger.error(f"saving predictions failed: {e}")
        persist_error = e

    status = notify(recs)
    if errors:
        notice(f"Daily job note: {len(errors)} game(s) skipped — {'; '.join(errors)}")
    if persist_error is not None:
        notice(f"Daily job note: today's predictions were NOT saved to the DB: {persist_error}")
    if status is SendStatus.FAILED:
        logger.error("Telegram send failed — recommendations are in the log above")
        return 1
    _mark_success(run_date)
    return 0


def run_daily_job(force: bool = False, scheduled: bool = False) -> int:
    lock = acquire_lock(LOCK_PATH)
    if lock is None:
        logger.info("another daily job run holds the lock — exiting quietly")
        return 0
    try:
        return _run(force, scheduled)
    except Exception as e:
        # The dead-man's-switch contract covers crashes too: anything a
        # step raises that the per-stage handlers didn't anticipate
        # still produces a Telegram notice, not just a traceback in a
        # log nobody is watching.
        logger.exception("daily job crashed")
        notice(f"Daily job CRASHED: {e}")
        return 1
    finally:
        lock.close()
