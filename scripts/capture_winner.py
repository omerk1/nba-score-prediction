"""
Capture winner.co.il's basketball lines page to a dated PNG — the first
step of the daily recommendation chain (capture -> extract -> recommend
-> Telegram). Thin wrapper over src/serving/capture_winner.py; scope:
docs/features/serving/winner_acquisition_scope.md.

Usage:
    venv/bin/python3 scripts/capture_winner.py
    venv/bin/python3 scripts/capture_winner.py --url <override> --output out.png

Exits non-zero when the capture fails (after one retry); a best-effort
.failed.png of whatever rendered is saved for diagnosis.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
# A relative --output must mean "relative to where I ran this", so grab
# the invoking cwd before pinning the process to the repo root (which
# default output paths and config loading need; launchd runs from "/").
_INVOKED_FROM = Path.cwd()
sys.path.append(str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.serving.capture_winner import WINNER_BASKETBALL_URL, capture_nba_page  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Winner's basketball lines page.")
    parser.add_argument("--url", default=WINNER_BASKETBALL_URL)
    parser.add_argument("--output", default=None, help="PNG path (default: dated file)")
    args = parser.parse_args()

    try:
        path = capture_nba_page(
            output_path=(_INVOKED_FROM / args.output) if args.output else None, url=args.url
        )
    except RuntimeError as e:
        logging.getLogger(__name__).error(str(e))
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
