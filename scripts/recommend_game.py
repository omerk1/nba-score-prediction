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

from src.serving.recommend import load_resources, recommend_game  # noqa: E402


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

    print(f"\n{args.home} (home) vs {args.away} (away){f' on {args.date}' if args.date else ''}")
    print(f"Predicted score: {rec['predicted_home_score']} - {rec['predicted_away_score']}")
    print(f"Predicted diff: {rec['predicted_diff']:+.1f} | Predicted total: {rec['predicted_total']:.1f}")
    print(f"Home win probability: {rec['home_win_probability']:.1%}")

    if "home_moneyline_edge" in rec:
        print(
            f"Home moneyline: market {rec['home_moneyline_market_probability']:.1%} "
            f"| edge {rec['home_moneyline_edge']:+.1%}"
        )
    if "away_moneyline_edge" in rec:
        print(
            f"Away moneyline: model {rec['away_win_probability']:.1%} vs market "
            f"{rec['away_moneyline_market_probability']:.1%} | edge {rec['away_moneyline_edge']:+.1%}"
        )

    if "home_spread" in rec:
        print(f"\nSpread {rec['home_spread']:+.1f} (home):")
        print(f"  Home cover probability: {rec['home_cover_probability']:.1%}", end="")
        if "home_spread_edge" in rec:
            print(
                f" | market {rec['home_spread_market_probability']:.1%} | edge {rec['home_spread_edge']:+.1%}"
            )
        else:
            print()
        print(f"  Away cover probability: {rec['away_cover_probability']:.1%}", end="")
        if "away_spread_edge" in rec:
            print(
                f" | market {rec['away_spread_market_probability']:.1%} | edge {rec['away_spread_edge']:+.1%}"
            )
        else:
            print()

    if "total_line" in rec:
        print(f"\nTotal {rec['total_line']}:")
        print(f"  Over probability: {rec['over_probability']:.1%}", end="")
        if "over_edge" in rec:
            print(f" | market {rec['over_market_probability']:.1%} | edge {rec['over_edge']:+.1%}")
        else:
            print()
        print(f"  Under probability: {rec['under_probability']:.1%}", end="")
        if "under_edge" in rec:
            print(f" | market {rec['under_market_probability']:.1%} | edge {rec['under_edge']:+.1%}")
        else:
            print()

    print(f"\nMargin heatmap (home margin : probability), top {args.heatmap_top}:")
    top = sorted(rec["margin_heatmap"].items(), key=lambda kv: kv[1], reverse=True)[: args.heatmap_top]
    for margin, prob in sorted(top):
        print(f"  {margin:+3d}: {prob:.4f}")


if __name__ == "__main__":
    main()
