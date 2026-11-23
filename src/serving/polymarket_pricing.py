"""
Prices Winner's spread/total sides at Polymarket's fair value.

Polymarket lists one market per line (e.g. "Spread: Celtics (-5.5)", "O/U
221.5"). Each line with real volume is priced at its latest price-history
point, and each priced line implies a fair expected home margin / total
under a normal model (same sigmas as the slate's cover probabilities):

    spread: P(home covers h) = Phi((mu + h) / sigma)  ->  mu = sigma * z - h
    total:  P(over L)        = Phi((mu - L) / sigma)  ->  mu = L + sigma * z

The game's fair value is the volume-weighted mean over lines priced inside
PRICE_BAND (extreme prices sit on the flat tail, where z is noisy). Winner's
own line is then priced against it: probability of each side, and expected
value at Winner's odds (p * odds - 1).

This is a price comparison, not a prediction: it says which Winner sides
cost the least against the sharpest available market, not which win.
Evidence: docs/MARKET_EDGE.md, 2026-11-13 entries (Winner's lines track
Polymarket's within ~0.4 pts, so expected values are mostly -5% to -12%).
"""

import json
import logging
import time
from typing import Callable, Optional

from scipy.stats import norm

from src.polymarket_prices.http_utils import get_json

logger = logging.getLogger(__name__)

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
SIGMA = {"spread": 13.5, "total": 19.5}
MIN_VOLUME = 2000.0  # dollars traded; most listed lines are near-empty
PRICE_BAND = (0.15, 0.85)
HISTORY_WINDOW_S = 6 * 3600
# nba_api abbreviation -> Polymarket slug spellings seen (slug: nba-{away}-{home}-{date})
SLUG_ALIASES = {"phx": ["phx", "pho"], "was": ["was", "wsh"], "uta": ["uta", "utah"], "nop": ["nop", "no"]}

# (market, side, Winner odds key, output key prefix)
SIDES = [
    ("spread", "home", "home_spread_odds", "home_spread"),
    ("spread", "away", "away_spread_odds", "away_spread"),
    ("total", "over", "over_odds", "over"),
    ("total", "under", "under_odds", "under"),
]


def _teams() -> dict:
    from nba_api.stats.static import teams

    return {t["id"]: t for t in teams.get_teams()}


def find_event(
    away_id: int, home_id: int, game_date: str, get: Callable = get_json, teams: dict = None
) -> Optional[dict]:
    teams = teams or _teams()
    away, home = teams[away_id]["abbreviation"].lower(), teams[home_id]["abbreviation"].lower()
    for a in SLUG_ALIASES.get(away, [away]):
        for h in SLUG_ALIASES.get(home, [home]):
            events = get(GAMMA_EVENTS, params={"slug": f"nba-{a}-{h}-{game_date}"})
            if events:
                return events[0]
    return None


def parse_spread(question: str, outcomes: list, home_name: str):
    """'Spread: Celtics (-5.5)' -> (home spread, index of the home outcome).
    Team names match exactly: substring matching reads 'Nets' inside 'Hornets'."""
    try:
        value = float(question[question.rindex("(") + 1 : question.rindex(")")])
    except ValueError:
        return None, None
    named = question.split(":", 1)[1].split("(")[0].strip().lower()
    home = home_name.lower()
    home_idx = next((i for i, o in enumerate(outcomes) if o.strip().lower() == home), None)
    if home_idx is None:
        return None, None
    return (value if named == home else -value), home_idx


def parse_total(question: str, outcomes: list):
    """'Celtics vs. Pistons: O/U 221.5' -> (line, index of the Over outcome)."""
    try:
        value = float(question.rsplit(" ", 1)[-1].strip("?"))
    except ValueError:
        return None, None
    over_idx = next((i for i, o in enumerate(outcomes) if o.strip().lower() == "over"), None)
    return value, over_idx


def latest_price(token_id: str, ts: int, get: Callable = get_json) -> Optional[float]:
    data = get(
        CLOB_HISTORY,
        params={"market": token_id, "startTs": ts - HISTORY_WINDOW_S, "endTs": ts, "fidelity": 10},
    )
    points = [h for h in (data or {}).get("history", []) if h["t"] <= ts]
    return float(max(points, key=lambda h: h["t"])["p"]) if points else None


def priced_lines(event: dict, home_name: str, ts: int, get: Callable = get_json) -> list[dict]:
    rows = []
    for m in event.get("markets", []):
        kind = {"spreads": "spread", "totals": "total"}.get(m.get("sportsMarketType"))
        volume = float(m.get("volume") or 0.0)
        if kind is None or volume < MIN_VOLUME:
            continue
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except (TypeError, ValueError):
            continue
        q = m.get("question", "")
        line, idx = parse_spread(q, outcomes, home_name) if kind == "spread" else parse_total(q, outcomes)
        if line is None or idx is None or idx >= len(tokens):
            continue
        p = latest_price(tokens[idx], ts, get)
        if p is not None:
            rows.append({"market": kind, "line": line, "p": p, "volume": volume})
    return rows


def fair_value(lines: list[dict], market: str) -> tuple[Optional[float], int]:
    """(fair home margin or total, number of lines used)."""
    s = SIGMA[market]
    use = [r for r in lines if r["market"] == market and PRICE_BAND[0] <= r["p"] <= PRICE_BAND[1]]
    if not use:
        return None, 0
    total_w = sum(r["volume"] for r in use)
    mus = [
        (s * norm.ppf(r["p"]) - r["line"]) if market == "spread" else (r["line"] + s * norm.ppf(r["p"]))
        for r in use
    ]
    return sum(mu * r["volume"] for mu, r in zip(mus, use)) / total_w, len(use)


def side_probability(market: str, side: str, fair: float, line: float) -> float:
    """Probability of one Winner side under the fair value. `line` is the home
    spread (spread) or the total line (total)."""
    s = SIGMA[market]
    p_a = norm.cdf((fair + line) / s) if market == "spread" else norm.cdf((fair - line) / s)
    return p_a if side in ("home", "over") else 1.0 - p_a


def price_pick(
    pick: dict, game_date: str, ts: int = None, get: Callable = get_json, teams: dict = None
) -> dict:
    """Polymarket fields for one Winner pick (keys `<side>_pm_probability`,
    `<side>_pm_ev`, plus `pm_fair_margin` / `pm_fair_total`). Empty when the
    game isn't listed; a market with no usable lines is left out."""
    ts = ts or int(time.time())
    teams = teams or _teams()
    event = find_event(pick["away_team_id"], pick["home_team_id"], game_date, get, teams)
    if event is None:
        return {}
    lines = priced_lines(event, teams[pick["home_team_id"]]["nickname"], ts, get)
    out = {"pm_priced_at": ts}
    fair = {}
    for market, key, line_key in (
        ("spread", "pm_fair_margin", "home_spread"),
        ("total", "pm_fair_total", "total_line"),
    ):
        mu, n = fair_value(lines, market)
        if mu is not None and pick.get(line_key) is not None:
            fair[market] = mu
            out[key], out[f"pm_{market}_lines"] = float(mu), n
    for market, side, odds_key, prefix in SIDES:
        line = pick.get("home_spread" if market == "spread" else "total_line")
        odds = pick.get(odds_key)
        if market not in fair or line is None or odds is None:
            continue
        p = float(side_probability(market, side, fair[market], line))
        out[f"{prefix}_pm_probability"] = p
        out[f"{prefix}_pm_ev"] = p * odds - 1.0
    return out
