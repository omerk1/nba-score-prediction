"""
Grades the daily job's logged recommendations (outputs/predictions_log.csv)
against final scores, to answer whether the model's "edge" is real: for
every priced side of every market it records whether the side won, and
summarizes hit rate, calibration and flat-stake return by edge bucket.

Every side the log priced counts as a bet, not only positive-edge ones —
the negative-edge buckets are the control showing the edge column ranks
sides in the right order. A flat 1-unit stake per side keeps the numbers
free of any staking assumption.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.serving.recommended_bets import SIDES as _SIDES

GAMES_DB = Path("data/raw/nba_api.sqlite")
LOG_PATH = Path("outputs/predictions_log.csv")
# The log's game_date is the run's slate day; the NBA game lands that day
# (US time) or, for a run after midnight, the day after.
RESULT_WINDOW_DAYS = 2
EDGE_BUCKETS = [-1.0, -0.10, -0.05, 0.0, 0.03, 0.06, 0.10, 1.0]

def load_results(db_path: Path = GAMES_DB) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT game_date, team_id_home AS home_team_id, team_id_away AS away_team_id, "
            "pts_home, pts_away FROM game WHERE pts_home IS NOT NULL",
            conn,
        )
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _outcome(market: str, side: str, rec: dict, home_pts: float, away_pts: float) -> str:
    """'win' | 'loss' | 'push' for one side of one market."""
    margin = home_pts - away_pts
    if market == "spread":
        value = margin + rec["home_spread"]  # >0: home covers
        value = value if side == "home" else -value
    elif market == "total":
        value = (home_pts + away_pts) - rec["total_line"]
        value = value if side == "over" else -value
    else:
        value = margin if side == "home" else -margin
    return "win" if value > 0 else "loss" if value < 0 else "push"


def grade(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per priced side of each logged game that has been played."""
    results = results.copy()
    rows = []
    for rec_row in log.to_dict("records"):
        rec = json.loads(rec_row["rec_json"])
        day = pd.to_datetime(rec_row["game_date"]).date()
        match = results[
            (results["home_team_id"] == rec_row["home_team_id"])
            & (results["away_team_id"] == rec_row["away_team_id"])
            & (results["game_date"] >= day)
            & (results["game_date"] <= day + pd.Timedelta(days=RESULT_WINDOW_DAYS))
        ].sort_values("game_date")
        if match.empty:
            continue
        final = match.iloc[0]
        for market, side, p_key, odds_key, mkt_key, edge_key in _SIDES:
            if edge_key not in rec:
                continue
            # Rows logged before odds were stored carry only the implied probability.
            odds = rec.get(odds_key) or 1.0 / rec[mkt_key]
            outcome = _outcome(market, side, rec, final["pts_home"], final["pts_away"])
            profit = {"win": odds - 1.0, "loss": -1.0, "push": 0.0}[outcome]
            rows.append(
                {
                    "game_date": day,
                    "home_team_id": rec_row["home_team_id"],
                    "away_team_id": rec_row["away_team_id"],
                    "market": market,
                    "side": side,
                    "model_prob": rec[p_key],
                    "odds": odds,
                    "edge": rec[edge_key],
                    "outcome": outcome,
                    "profit": profit,
                    "pred_diff_error": rec["predicted_diff"] - (final["pts_home"] - final["pts_away"]),
                    "pred_total_error": rec["predicted_total"] - (final["pts_home"] + final["pts_away"]),
                }
            )
    return pd.DataFrame(rows)


def summarize(bets: pd.DataFrame) -> pd.DataFrame:
    """Per edge bucket: bets, hit rate (pushes excluded), the model's mean
    probability (calibration: should match the hit rate), and flat-stake ROI."""
    if bets.empty:
        return pd.DataFrame()
    bets = bets.assign(bucket=pd.cut(bets["edge"], EDGE_BUCKETS, right=False))
    decided = bets[bets["outcome"] != "push"]
    out = pd.DataFrame(
        {
            "bets": bets.groupby("bucket", observed=True).size(),
            "hit_rate": decided.groupby("bucket", observed=True)["outcome"].apply(lambda s: (s == "win").mean()),
            "model_prob": decided.groupby("bucket", observed=True)["model_prob"].mean(),
            "roi": bets.groupby("bucket", observed=True)["profit"].mean(),
        }
    )
    return out
