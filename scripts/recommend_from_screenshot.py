"""
End-to-end CLI: a bookmaker screenshot in, recommendations for every NBA
game it contains out. Ties src/serving/extract_picks.py (vision extraction)
to src/serving/recommend.py (prediction + heatmap).

Usage:
    venv/bin/python3 scripts/recommend_from_screenshot.py path/to/screenshot.png
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()  # GOOGLE_API_KEY for extract_picks_from_screenshot's Gemini call

from src.serving.extract_picks import extract_picks_from_screenshot  # noqa: E402
from src.serving.recommend import format_recommendation, load_resources, recommend_game  # noqa: E402

# recommend_game's own accepted kwargs -- a pick may carry extra fields
# (e.g. push_odds) that extract_picks_from_screenshot returns but
# recommend_game doesn't consume yet; filtered out here rather than passed
# through and erroring.
_RECOMMEND_KWARGS = {
    "home_team_id",
    "away_team_id",
    "home_spread",
    "home_spread_odds",
    "away_spread_odds",
    "home_moneyline_odds",
    "away_moneyline_odds",
    "total_line",
    "over_odds",
    "under_odds",
}


def main():
    parser = argparse.ArgumentParser(description="Recommend bets from a screenshot of NBA odds")
    parser.add_argument("image_path", type=str, help="Path to the screenshot")
    parser.add_argument("--date", type=str, default=None, help="Game date (YYYY-MM-DD), default: today")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model for extraction")
    parser.add_argument(
        "--heatmap-top", type=int, default=15, help="How many top-probability margins to print"
    )
    args = parser.parse_args()

    picks = extract_picks_from_screenshot(args.image_path, model=args.model, game_date=args.date)
    if not picks:
        print("No NBA games recognized in this screenshot.")
        return

    print(f"Extracted {len(picks)} NBA game(s).")
    resources = load_resources()
    for pick in picks:
        dropped = {k: v for k, v in pick.items() if k not in _RECOMMEND_KWARGS}
        if dropped:
            print(f"(ignoring fields recommend_game doesn't use yet: {dropped})")
        kwargs = {k: v for k, v in pick.items() if k in _RECOMMEND_KWARGS}
        rec = recommend_game(game_date=args.date, resources=resources, **kwargs)
        print(f"\n{'=' * 60}")
        print(format_recommendation(rec, heatmap_top=args.heatmap_top))


if __name__ == "__main__":
    main()
