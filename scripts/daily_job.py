"""
Daily recommendations job CLI — the single entry point the launchd
agent invokes (data refresh -> Winner capture -> extraction ->
recommendations -> Telegram). Thin wrapper over
src/serving/daily_job.py; install the schedule with
scripts/install_launchd.sh.

Usage:
    venv/bin/python3 scripts/daily_job.py [--force]

--force re-runs even if today's run already succeeded (the success
marker otherwise stops a wake-coalesced launchd event from duplicating
a slate that a manual run already delivered).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))
# Everything downstream is repo-relative (DB paths, outputs/, logs/,
# configs/, .env); launchd's default cwd is "/".
os.chdir(_REPO_ROOT)

load_dotenv()  # GOOGLE_API_KEY + TELEGRAM_* before any deferred step import

from src.serving.daily_job import run_daily_job  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily recommendations job.")
    parser.add_argument("--force", action="store_true", help="re-run despite today's success marker")
    args = parser.parse_args()
    sys.exit(run_daily_job(force=args.force))
