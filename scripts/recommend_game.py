"""
CLI demo for src/serving/recommend.py -- given two team IDs and optional
market lines, prints the full recommendation (point prediction, win/cover/
over-under probabilities, edge vs. given odds, margin heatmap). Takes
already-structured picks; the screenshot-to-picks extraction step isn't
built yet (docs/features/serving/scope.md).

Usage:
    venv/bin/python3 scripts/recommend_game.py --home 1610612747 --away 1610612744 \
        --home-spread -9 --home-spread-odds 1.80 --away-spread-odds 1.80
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.serving.recommend import format_recommendation, load_resources, recommend_game  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Recommend a bet on one NBA game")
    parser.add_argument("--home", type=int, required=True, help="Home team ID")
    parser.add_argument("--away", type=int, required=True, help="Away team ID")
    parser.add_argument("--date", type=str, default=None, help="Game date (YYYY-MM-DD), default: today")
    parser.add_argument(
        "--home-spread", type=float, default=None, help="Home spread, market sign convention (e.g. -9)"
    )
    parser.add_argument("--home-spread-odds", type=float, default=None, help="Decimal odds for home covering")
    parser.add_argument("--away-spread-odds", type=float, default=None, help="Decimal odds for away covering")
    parser.add_argument("--home-ml-odds", type=float, default=None, help="Decimal moneyline odds, home")
    parser.add_argument("--away-ml-odds", type=float, default=None, help="Decimal moneyline odds, away")
    parser.add_argument("--total", type=float, default=None, help="Over/under total line")
    parser.add_argument("--over-odds", type=float, default=None, help="Decimal odds for over")
    parser.add_argument("--under-odds", type=float, default=None, help="Decimal odds for under")
    parser.add_argument(
        "--heatmap-top", type=int, default=15, help="How many top-probability margins to print"
    )
    args = parser.parse_args()

    resources = load_resources()
    rec = recommend_game(
        args.home,
        args.away,
        resources,
        game_date=args.date,
        home_spread=args.home_spread,
        home_spread_odds=args.home_spread_odds,
        away_spread_odds=args.away_spread_odds,
        home_moneyline_odds=args.home_ml_odds,
        away_moneyline_odds=args.away_ml_odds,
        total_line=args.total,
        over_odds=args.over_odds,
        under_odds=args.under_odds,
    )

    print(f"\n{format_recommendation(rec, heatmap_top=args.heatmap_top)}")


if __name__ == "__main__":
    main()
