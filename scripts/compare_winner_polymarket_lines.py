"""
Is Winner's spread/total line ever far enough from Polymarket's to be +EV?

For every game in outputs/predictions_log.csv (Winner lines as captured by
the daily job), fetch that game's Polymarket event, take every full-game
spread and totals market (Polymarket lists one market per line), and price
each at the moment Winner's line was captured -- not at tip-off, which
would use news a bettor didn't have yet.

Price at capture time = the last point at or before created_at of the CLOB
price history (clob.polymarket.com/prices-history, 10-minute fidelity,
cached under --raw-dir). The Data API v1 trades endpoint the collector
used was retired (HTTP 410). Polymarket lists a market for every line and
most are near-empty, so only lines with at least MIN_VOLUME dollars traded
are priced.

Each priced Polymarket line gives a fair expected margin/total under a
normal model (same sigmas as the live job: 13.5 spread, 19.5 total):
    spread: P(home covers h) = Phi((mu + h) / sigma)  ->  mu = sigma * z - h
    total:  P(over L)        = Phi((mu - L) / sigma)  ->  mu = L + sigma * z
The game's fair value is the volume-weighted mean over fresh lines with
price in [0.15, 0.85] (extreme prices sit on the flat tail where z is noisy).
Winner's own line is then priced against that fair value; EV = p * odds - 1.

No outcomes are used: this only measures whether usable gaps exist.
Analysis only -- does not touch the model, features or config.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_api.stats.static import teams as nba_teams  # noqa: E402

from src.polymarket_prices.gamma import (  # noqa: E402
    SPREAD_TYPE,
    TOTALS_TYPE,
    _parse_market,
    fetch_event_by_slug,
)
from src.polymarket_prices.http_utils import cached_get_json  # noqa: E402

SIGMA = {"spread": 13.5, "total": 19.5}
STALE_HOURS = 6
MIN_VOLUME = 2000
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
PRICE_BAND = (0.15, 0.85)
SLUG_ALIASES = {"phx": ["phx", "pho"], "was": ["was", "wsh"], "uta": ["uta", "utah"], "nop": ["nop", "no"]}

logger = logging.getLogger(__name__)


def team_abbrevs() -> dict:
    return {t["id"]: t["abbreviation"].lower() for t in nba_teams.get_teams()}


def find_event(away: str, home: str, date: str, raw_dir: str):
    for a in SLUG_ALIASES.get(away, [away]):
        for h in SLUG_ALIASES.get(home, [home]):
            ev = fetch_event_by_slug(f"nba-{a}-{h}-{date}", raw_dir)
            if ev:
                return ev
    return None


def price_at(token_id: str, ts: int, raw_dir: str):
    """Last price-history point at/before ts for one outcome token."""
    cache = os.path.join(raw_dir, "history", f"{token_id[:40]}_{ts}.json.gz")
    data = cached_get_json(
        cache,
        CLOB_HISTORY,
        params={"market": token_id, "startTs": ts - STALE_HOURS * 3600, "endTs": ts, "fidelity": 10},
    )
    pts = [h for h in (data or {}).get("history", []) if h["t"] <= ts]
    if not pts:
        return None, None
    last = max(pts, key=lambda h: h["t"])
    return float(last["p"]), int(last["t"])


def parse_spread_line(market, home_name: str):
    """Return (home_spread, outcome index of the home side) from 'Spread: Team (-X.5)'."""
    q = market.question
    try:
        val = float(q[q.rindex("(") + 1 : q.rindex(")")])
    except ValueError:
        return None, None
    named = q.split(":", 1)[1].split("(")[0].strip()
    home_idx = next(
        (
            i
            for i, o in enumerate(market.outcomes)
            if o.strip().lower() == home_name.lower()
        ),
        None,
    )
    if home_idx is None:
        return None, None
    named_is_home = named.lower() == home_name.lower()
    return (val if named_is_home else -val), home_idx


def parse_total_line(market):
    q = market.question
    try:
        val = float(q.rsplit(" ", 1)[-1].strip("?"))
    except ValueError:
        return None, None
    over_idx = next((i for i, o in enumerate(market.outcomes) if o.lower() == "over"), None)
    return val, over_idx


def poly_lines(event: dict, capture_ts: int, home_name: str, raw_dir: str) -> list:
    rows = []
    for m in event.get("markets", []):
        mk = _parse_market(m)
        if mk is None or mk.sports_market_type not in (SPREAD_TYPE, TOTALS_TYPE) or mk.volume < MIN_VOLUME:
            continue
        if mk.sports_market_type == SPREAD_TYPE:
            market, (line, idx) = "spread", parse_spread_line(mk, home_name)
        else:
            market, (line, idx) = "total", parse_total_line(mk)
        if line is None or idx is None:
            continue
        p, ts = price_at(mk.clob_token_ids[idx], capture_ts, raw_dir)
        rows.append({"market": market, "line": line, "p": p, "trade_ts": ts, "volume": mk.volume})
    return rows


def fair_value(lines: list, market: str, capture_ts: int):
    s = SIGMA[market]
    use = [
        r
        for r in lines
        if r["market"] == market
        and r["p"] is not None
        and PRICE_BAND[0] <= r["p"] <= PRICE_BAND[1]
        and capture_ts - r["trade_ts"] <= STALE_HOURS * 3600
    ]
    if not use:
        return None, 0, 0.0
    z = norm.ppf([r["p"] for r in use])
    lines_ = np.array([r["line"] for r in use])
    mu = s * z - lines_ if market == "spread" else lines_ + s * z
    w = np.array([r["volume"] for r in use])
    return float(np.average(mu, weights=w)), len(use), float(w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/predictions_log.csv")
    ap.add_argument("--raw-dir", default="data/polymarket_prices/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    log = pd.read_csv(args.log)
    if args.limit:
        log = log.head(args.limit)
    abbr = team_abbrevs()
    names = {t["id"]: t["nickname"] for t in nba_teams.get_teams()}

    out = []
    for _, r in log.iterrows():
        capture_ts = int(pd.Timestamp(r.created_at).timestamp())
        home, away = abbr[r.home_team_id], abbr[r.away_team_id]
        ev = find_event(away, home, r.game_date, args.raw_dir)
        base = {"game_date": r.game_date, "away": away, "home": home, "capture_utc": r.created_at}
        if ev is None:
            out.append({**base, "status": "no_event"})
            continue
        lines = poly_lines(ev, capture_ts, names[r.home_team_id], args.raw_dir)
        row = {
            **base,
            "status": "ok",
            "n_lines_priced": len(lines),
        }
        for market, w_line, odds_a, odds_b in (
            ("spread", r.home_spread, r.home_spread_odds, r.away_spread_odds),
            ("total", r.total_line, r.over_odds, r.under_odds),
        ):
            mu, n, vol = fair_value(lines, market, capture_ts)
            row[f"{market}_winner_line"] = w_line
            row[f"{market}_poly_fair"] = mu
            row[f"{market}_poly_n_lines"] = n
            row[f"{market}_poly_volume"] = vol
            if mu is None or pd.isna(w_line):
                continue
            s = SIGMA[market]
            # side A = home cover / over; B = away cover / under
            p_a = norm.cdf((mu + w_line) / s) if market == "spread" else norm.cdf((mu - w_line) / s)
            # gap in points, signed toward side A being the value side
            row[f"{market}_gap_pts"] = (mu + w_line) if market == "spread" else (mu - w_line)
            row[f"{market}_ev_a"] = p_a * odds_a - 1
            row[f"{market}_ev_b"] = (1 - p_a) * odds_b - 1
            row[f"{market}_best_ev"] = max(row[f"{market}_ev_a"], row[f"{market}_ev_b"])
        out.append(row)
        print(
            f"{r.game_date} {away}@{home}: spread gap {row.get('spread_gap_pts')}, total gap {row.get('total_gap_pts')}",
            flush=True,
        )

    pd.DataFrame(out).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
