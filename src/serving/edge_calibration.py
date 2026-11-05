"""
Learns how much of the model's disagreement with the market line actually
shows up in final scores, so cover/over probabilities (and the edges built on
them) only claim what graded results support.

For each market, over graded games: gap = prediction - line, realised =
actual - line. The least-squares slope of realised on gap (through the
origin) is the share of the gap that turned out real: 1 means the model's
disagreement is fully informative, 0 means the line already knew everything
the model did. Pricing then uses line + weight * gap instead of the raw
prediction.

The weight is the slope's lower bound (slope - z * standard error), clipped
to [0, 1], and 0 until `min_games` are graded: edges appear only once the
record shows the disagreement is real, and return on their own if a better
model earns them. Only games priced by the currently loaded model count, so
a retrained model starts from zero rather than inheriting another model's
record.

Held-out evidence for why the raw prediction can't be priced as-is:
docs/MARKET_EDGE.md, 2026-11-04 entry (claimed 67% on strong bets, hit 50%).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.serving.grade_predictions import grade

MARKETS = ("spread", "total")


@dataclass(frozen=True)
class GapWeight:
    weight: float  # what pricing uses: lower bound of slope, clipped to [0, 1]
    slope: float  # point estimate (NaN below min_games)
    se: float
    n: int


def line_games(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per graded game per market: gap (prediction - line) and
    realised (actual - line), both in points."""
    bets = grade(log, results)
    if bets.empty:
        return pd.DataFrame(columns=["market", "gap", "realised"])
    bets = bets[bets["market"].isin(MARKETS)].drop_duplicates(
        ["game_date", "home_team_id", "away_team_id", "market"]
    )
    is_spread = bets["market"] == "spread"
    pred = np.where(is_spread, bets["pred_diff"], bets["pred_total"])
    actual = np.where(is_spread, bets["actual_diff"], bets["actual_total"])
    return pd.DataFrame(
        {
            "market": bets["market"].to_numpy(),
            "gap": pred - bets["market_line"].to_numpy(),
            "realised": actual - bets["market_line"].to_numpy(),
        }
    )


def fit_gap_weight(gap: np.ndarray, realised: np.ndarray, min_games: int, z: float) -> GapWeight:
    gap, realised = np.asarray(gap, float), np.asarray(realised, float)
    n = len(gap)
    sxx = float(np.sum(gap**2))
    if n < max(min_games, 2) or sxx == 0:
        return GapWeight(weight=0.0, slope=float("nan"), se=float("nan"), n=n)
    slope = float(np.sum(gap * realised) / sxx)
    resid = realised - slope * gap
    se = float(np.sqrt(np.sum(resid**2) / (n - 1) / sxx))
    return GapWeight(weight=float(np.clip(slope - z * se, 0.0, 1.0)), slope=slope, se=se, n=n)


def fit_gap_weights(
    log: pd.DataFrame,
    results: pd.DataFrame,
    model_trained_at: str | None,
    min_games: int,
    z: float,
) -> dict[str, GapWeight]:
    """Per-market weights from the graded games of the model trained at
    `model_trained_at` (every logged game when None)."""
    if model_trained_at is not None and "model_trained_at" in log:
        log = log[log["model_trained_at"] == model_trained_at]
    games = line_games(log, results) if not log.empty else pd.DataFrame(columns=["market", "gap", "realised"])
    return {
        m: fit_gap_weight(games.loc[games["market"] == m, "gap"], games.loc[games["market"] == m, "realised"], min_games, z)
        for m in MARKETS
    }


def shrink_toward_line(pred: float, line: float, weight: float) -> float:
    """The point to price from: weight 1 is the raw prediction, 0 is the line itself."""
    return line + weight * (pred - line)
