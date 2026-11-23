"""
Evening Winner-vs-Polymarket line snapshot (src/serving/evening_snapshot.py).
Installed as a 22:30 LaunchAgent by scripts/install_launchd.sh.

Usage:
    venv/bin/python3 scripts/evening_snapshot.py [--scheduled]

--scheduled is passed only by the launchd agent: a run starting outside
21:00-01:30 (a late firing after the Mac slept) is skipped.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))
os.chdir(_REPO_ROOT)  # launchd's default cwd is "/"; everything downstream is repo-relative

from src.serving.evening_snapshot import run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    try:
        return run(scheduled=args.scheduled)
    except Exception:
        logging.exception("evening snapshot failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
