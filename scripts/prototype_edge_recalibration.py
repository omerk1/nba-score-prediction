"""
Prototype: recalibrate the spread-cover and over probabilities that drive the
daily slate's "strong bet" list, and check whether any strong bets survive
out-of-sample.

The live job prices a side as norm.cdf((prediction - line) / sigma) and calls
it strong when that probability beats the market's implied 1/odds by 5+
points. Held-out grading showed those claimed probabilities run ~17 points
above the realised hit rate and that edge size doesn't rank bets. This fits a
recalibrator on the early part of the held-out season and evaluates it on the
rest (and the reverse, as a robustness check), so every evaluated probability
comes from a map that never saw that game.

Methods per market (spread, total):
  raw       the live job's construction (normal curve, fixed sigma)
  logistic  p = sigmoid(a + b*z), z = (prediction - line) / sigma; b near 0
            means the prediction-vs-line gap carries no information
  isotonic  monotone map from raw p to the observed rate
  coinflip  p = 0.5 for both sides (the "market is right" reference)

Input is the per-game join written by scripts/market_benchmark.py (genuinely
held-out predictions plus Polymarket lines). No odds are stored there, so
strong-bet returns use flat 1.80 odds, the live log's typical price.

Usage: venv/bin/python3 scripts/prototype_edge_recalibration.py \
           --games-csv outputs/market_benchmark_games_<tag>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Sigmas that reproduce the live job's logged cover/over probabilities
# (fit against outputs/predictions_log.csv, max abs error 1.6 points).
SIGMA = {"spread": 13.5, "total": 19.5}
ODDS = 1.80
STRONG_EDGE = 0.05
DEFAULT_SPLIT = "2026-02-01"


def build_sides(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game per market: z score and whether the home/over side won."""
    rows = []
    sp = games[games["has_spread"]]
    rows.append(
        pd.DataFrame(
            {
                "date": sp["GAME_DATE"],
                "market": "spread",
                "z": (sp["model_diff_pred"] - sp["market_diff_home"]) / SIGMA["spread"],
                "margin": sp["actual_diff"] - sp["market_diff_home"],
            }
        )
    )
    to = games[games["has_totals"]]
    rows.append(
        pd.DataFrame(
            {
                "date": to["GAME_DATE"],
                "market": "total",
                "z": (to["model_total_pred"] - to["totals_line"]) / SIGMA["total"],
                "margin": to["actual_total"] - to["totals_line"],
            }
        )
    )
    out = pd.concat(rows, ignore_index=True)
    out = out[out["margin"] != 0]  # pushes settle neither side
    out["y"] = (out["margin"] > 0).astype(int)  # home covered / went over
    out["p_raw"] = norm.cdf(out["z"])
    return out


def fit_predict(fit: pd.DataFrame, ev: pd.DataFrame) -> tuple[dict, float]:
    """Probabilities of the home/over side on `ev` per method, plus the logistic slope."""
    lr = LogisticRegression(C=1e6).fit(fit[["z"]], fit["y"])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(fit["p_raw"], fit["y"])
    probs = {
        "raw": ev["p_raw"].to_numpy(),
        "logistic": lr.predict_proba(ev[["z"]])[:, 1],
        "isotonic": iso.predict(ev["p_raw"]),
        "coinflip": np.full(len(ev), 0.5),
    }
    return probs, float(lr.coef_[0][0])


def score(p: np.ndarray, y: np.ndarray) -> dict:
    """Log loss, Brier, and the flat-odds strong-bet record for one method."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "brier": float(np.mean((p - y) ** 2)),
    }
    # Each game offers two sides; a side is strong when its probability beats 1/odds by STRONG_EDGE.
    mp = 1.0 / ODDS
    wins = np.concatenate([y[p - mp >= STRONG_EDGE] == 1, y[(1 - p) - mp >= STRONG_EDGE] == 0])
    n = len(wins)
    out.update(
        strong_bets=n,
        strong_hit=float(wins.mean()) if n else np.nan,
        strong_roi=float(np.where(wins, ODDS - 1, -1.0).mean()) if n else np.nan,
    )
    return out


def run(sides: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    results = []
    for direction, fit_mask in (("fit_early_eval_late", sides["date"] < split), ("fit_late_eval_early", sides["date"] >= split)):
        for market in ("spread", "total"):
            m = sides["market"] == market
            fit, ev = sides[m & fit_mask], sides[m & ~fit_mask]
            probs, slope = fit_predict(fit, ev)
            for method, p in probs.items():
                row = {"split": direction, "market": market, "method": method, "n_fit": len(fit), "n_eval": len(ev)}
                row.update(score(p, ev["y"].to_numpy()))
                row["logistic_slope"] = slope if method == "logistic" else np.nan
                results.append(row)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-csv", required=True, type=Path)
    parser.add_argument("--split-date", default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV for the results table")
    args = parser.parse_args()

    games = pd.read_csv(args.games_csv, parse_dates=["GAME_DATE"])
    res = run(build_sides(games), pd.Timestamp(args.split_date))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
