"""
Evening Winner-vs-Polymarket snapshot (scripts/evening_snapshot.py, 22:30
local via launchd): a second capture of Winner's lines after the NBA's
afternoon injury report, priced against Polymarket at the same moment and
appended to the line-snapshot log as "evening" rows. Research data only:
no Telegram message, no slate.

Question it answers (docs/features/winner_vs_polymarket/PLAN.md, step 3):
when injury news breaks, does Winner's line fall behind Polymarket's long
enough to bet? The afternoon capture can't see that; both markets have had
all day to agree by then.
"""

import datetime
import logging
from pathlib import Path

from src.serving.capture_winner import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# launchd fires a missed calendar event at the next wake; a capture after
# most games have tipped (02:00-05:30 local) would measure nothing useful.
WINDOW_START = datetime.time(21, 0)
WINDOW_END = datetime.time(1, 30)


def in_window(now: datetime.datetime) -> bool:
    t = now.time()
    return t >= WINDOW_START or t < WINDOW_END


def evening_capture_path(now: datetime.datetime, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    # Outside capture_winner's retention pattern on purpose: these are the
    # raw evidence for the evening comparison, kept for the whole test.
    return output_dir / f"winner_lines_evening_{now:%Y-%m-%d_%H%M}.json"


def run(now: datetime.datetime = None, scheduled: bool = False) -> int:
    from src.serving import daily_job
    from src.serving.capture_winner import capture_nba_lines
    from src.serving.line_snapshots import append_snapshot

    now = now or datetime.datetime.now()
    if scheduled and not in_window(now):
        logger.info(f"started at {now:%H:%M}, outside the evening window — skipping")
        return 0
    path = capture_nba_lines(output_path=evening_capture_path(now))
    picks = daily_job.extract(path)
    if not picks:
        logger.info("no NBA games in the next 24h on Winner")
        return 0
    for pick in picks:
        pick.update(daily_job.polymarket_fields(pick))
    priced = sum(any(k.endswith("_pm_ev") for k in p) for p in picks)
    append_snapshot(picks, "evening")
    logger.info(f"evening snapshot: {len(picks)} games, {priced} priced on Polymarket")
    return 0
