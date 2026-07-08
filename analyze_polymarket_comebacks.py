"""
Comeback threshold analysis.

Given data/polymarket_comeback/games.csv (produced by run_polymarket_comeback.py),
print, for each threshold p in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20], the
number and percent of games where the eventual winner's in-game (robust) min
price was <= p, plus a list of the most extreme comebacks with Polymarket links.

Usage:
    python analyze_polymarket_comebacks.py [--games-csv data/polymarket_comeback/games.csv]
"""

import argparse

import pandas as pd

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]


def main():
    parser = argparse.ArgumentParser(description="Comeback threshold analysis over games.csv")
    parser.add_argument("--games-csv", default="data/polymarket_comeback/games.csv")
    parser.add_argument("--top-n", type=int, default=10, help="How many most-extreme comebacks to list")
    args = parser.parse_args()

    df = pd.read_csv(args.games_csv)

    usable = df[df["in_game_min_price_winner_robust"].notna()].copy()
    dropped = len(df) - len(usable)

    print(f"Loaded {len(df)} games from {args.games_csv} ({dropped} excluded: missing in-game min price)")
    if not usable.empty:
        dq_counts = usable["data_quality"].value_counts()
        print("\ndata_quality breakdown (usable games):")
        for k, v in dq_counts.items():
            print(f"  {k}: {v}")

    print(f"\n{'threshold p':>12} | {'# games winner traded <= p':>28} | {'% of games':>10}")
    print("-" * 58)
    n = len(usable)
    for p in THRESHOLDS:
        count = int((usable["in_game_min_price_winner_robust"] <= p).sum())
        pct = 100.0 * count / n if n else 0.0
        print(f"{p:>12.2%} | {count:>28d} | {pct:>9.1f}%")

    print(f"\nTop {min(args.top_n, len(usable))} most extreme comebacks (lowest winner in-game min price):")
    top = usable.sort_values("in_game_min_price_winner_robust").head(args.top_n)
    for _, row in top.iterrows():
        link = f"https://polymarket.com/event/{row['slug']}"
        pregame = row.get("pregame_price_winner")
        pregame_str = f"{pregame:.1%}" if pd.notna(pregame) else "n/a"
        print(
            f"  {row['slug']:<28} winner={row['winner_team']:<10} "
            f"pregame={pregame_str:>6} in_game_min={row['in_game_min_price_winner_robust']:.1%}  {link}"
        )


if __name__ == "__main__":
    main()
