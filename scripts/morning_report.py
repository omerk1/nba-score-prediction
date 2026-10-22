"""Morning performance report: refresh final scores, grade the predictions
log, print the report and send it to Telegram.

    venv/bin/python3 scripts/morning_report.py [--no-send] [--no-refresh]

Meant for a morning LaunchAgent (scripts/install_launchd.sh). --no-refresh
skips the game-table fetch (uses scores already in the DB).
"""

import argparse
import html
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.serving import morning_report  # noqa: E402
from src.serving.notify_telegram import SendStatus, send_html_report, send_notice  # noqa: E402

logger = logging.getLogger("morning_report")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-send", action="store_true", help="print only")
    parser.add_argument("--no-refresh", action="store_true", help="skip fetching final scores")
    args = parser.parse_args()

    stale_note = ""
    if not args.no_refresh:
        try:
            from src.data_processing.daily_update import refresh_games

            refresh_games()
        except Exception as e:
            logger.error(f"score refresh failed: {e}")
            stale_note = f"\n⚠ score refresh failed ({html.escape(str(e))}); results may be incomplete."

    try:
        result = morning_report.run()
    except Exception as e:
        logger.exception("morning report failed")
        if not args.no_send:
            send_notice(f"Morning report FAILED: {e}")
        return 1

    if isinstance(result, str):  # nothing to report
        print(result)
        if args.no_send:
            return 0
        return 1 if send_notice(result + stale_note) is SendStatus.FAILED else 0

    caption, path = result
    print(f"{caption}\n\nwritten: {path}")
    if args.no_send:
        return 0
    return 1 if send_html_report(path, caption + stale_note) is SendStatus.FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
