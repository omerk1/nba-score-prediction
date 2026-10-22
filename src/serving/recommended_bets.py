"""
Which priced sides of a recommend_game() result count as a "recommended
bet": a side whose model edge over the break-even price is above
notifications.recommend_min_edge. Shared by the daily message, the HTML
slate and the morning performance report so all three agree on what
"recommended" means.
"""

# Edge at/above which a recommended bet is flagged "strong"; the floor for
# "recommended" at all is notifications.recommend_min_edge (0 = any positive edge).
STRONG_EDGE = 0.05

# (market, side, rec key: model probability, odds, market probability, edge)
SIDES = [
    ("spread", "home", "home_cover_probability", "home_spread_odds", "home_spread_market_probability", "home_spread_edge"),
    ("spread", "away", "away_cover_probability", "away_spread_odds", "away_spread_market_probability", "away_spread_edge"),
    ("total", "over", "over_probability", "over_odds", "over_market_probability", "over_edge"),
    ("total", "under", "under_probability", "under_odds", "under_market_probability", "under_edge"),
    ("moneyline", "home", "home_win_probability", "home_moneyline_odds", "home_moneyline_market_probability", "home_moneyline_edge"),
    ("moneyline", "away", "away_win_probability", "away_moneyline_odds", "away_moneyline_market_probability", "away_moneyline_edge"),
]


def pick_label(market: str, side: str, rec: dict, home: str, away: str) -> str:
    if market == "moneyline":
        return f"{home if side == 'home' else away} to win"
    if market == "spread":
        line = rec["home_spread"]
        return f"{home} {line:+g}" if side == "home" else f"{away} {-line:+g}"
    return f"{'Over' if side == 'over' else 'Under'} {rec['total_line']:g}"


def recommended_bets(rec: dict, min_edge: float, team_names: dict) -> list[dict]:
    """Sides of one game with edge > min_edge, best edge first."""
    home = str(team_names.get(rec["home_team_id"], rec["home_team_id"]))
    away = str(team_names.get(rec["away_team_id"], rec["away_team_id"]))
    bets = []
    for market, side, p_key, odds_key, mkt_key, edge_key in SIDES:
        edge = rec.get(edge_key)
        if edge is None or edge <= min_edge:
            continue
        bets.append(
            {
                "game": f"{away} @ {home}",
                "home": home,
                "away": away,
                "home_team_id": rec["home_team_id"],
                "away_team_id": rec["away_team_id"],
                "market": market,
                "pick": pick_label(market, side, rec, home, away),
                "odds": rec.get(odds_key),
                "model_prob": rec[p_key],
                "market_prob": rec.get(mkt_key),
                "edge": edge,
            }
        )
    return sorted(bets, key=lambda b: b["edge"], reverse=True)
