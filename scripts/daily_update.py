"""
Daily data refresh CLI — keeps the serving-path data stores current
through yesterday's games (game table, style-fingerprint cache, injury
features). Thin wrapper over src/data_processing/daily_update.py; scope:
docs/features/serving/daily_data_refresh_scope.md.

Usage:
    venv/bin/python3 scripts/daily_update.py
    venv/bin/python3 scripts/daily_update.py --importance-snapshot always

Exits non-zero if any refresh step failed or the game-table freshness
gate fails — the daily recommendation job should check this before
sending anything.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()  # must run before the injury-chain imports inside the refresh steps

from src.data_processing.daily_update import (  # noqa: E402
    run_daily_update,
    should_run_importance_snapshot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the serving-path data stores.")
    parser.add_argument(
        "--importance-snapshot",
        choices=["auto", "always", "never"],
        default="auto",
        help="Current-season player_importance snapshot: auto = Mondays only (default).",
    )
    args = parser.parse_args()

    include = {
        "auto": should_run_importance_snapshot(),
        "always": True,
        "never": False,
    }[args.importance_snapshot]
    return run_daily_update(include_importance_snapshot=include)


if __name__ == "__main__":
    sys.exit(main())
