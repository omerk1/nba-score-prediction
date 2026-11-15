"""
Does Polymarket's pre-game line, used as model input, leave anything for the
model to add?

Polymarket spread/total lines exist only for 2025-26 (games.csv; 2024-25 has
moneyline only), too short for the 5-fold CV harness, so this is a
one-season prototype. The champion model is trained once on fold5 (data
through 2024-04), so its 2025-26 predictions are held out. The season is
split at 2026-02-01 and each approach is fit on one half and evaluated on
the other, in both directions:

  market   pred = PM line
  model    pred = champion prediction
  blend    pred = line + a * (model - line), a fit by least squares
  resid    pred = line + CatBoost(features + line + model gap -> actual - line)

The question is the "beyond the line" signal: corr(pred - line, actual - line)
and the record of bets where |pred - line| clears a threshold, at 1.80 odds
(break-even 55.6%). The PM line here is the last pre-game trade (closing
line), a slightly sharper line than one available hours earlier.

Analysis only: nothing here touches the model, features or config.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CWD = os.getcwd()  # market_benchmark chdirs to its own repo on import; data lives in the caller's checkout
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from market_benchmark import load_polymarket_games  # noqa: E402

os.chdir(CWD)

from src.evaluation.cv_harness import run_split, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

ODDS = 1.80
THRESHOLDS = (1.0, 2.0, 3.0, 4.0)
MARKETS = {
    # market: (line column, model column, actual column)
    "spread": ("market_diff_home", "model_diff", "actual_diff"),
    "total": ("market_total", "model_total", "actual_total"),
}
SEASONS = {"early": ("2025-10-01", "2026-01-31"), "late": ("2026-02-01", "2026-06-30")}


def season_frame(features: pd.DataFrame, predictor, feature_cols) -> pd.DataFrame:
    preds = predictor.predict(features[feature_cols])
    df = features[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "PTS_home", "PTS_away"]].copy()
    df["model_diff"] = preds[:, 0] - preds[:, 1]
    df["model_total"] = preds[:, 0] + preds[:, 1]
    df["actual_diff"] = df["PTS_home"] - df["PTS_away"]
    df["actual_total"] = df["PTS_home"] + df["PTS_away"]
    df["game_date_norm"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    return pd.concat([df, features[feature_cols].reset_index(drop=True).set_index(df.index)], axis=1)


def join_market(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    m = market[["home_team_id", "away_team_id", "game_date_norm", "market_diff_home", "market_total"]]
    return df.merge(
        m,
        left_on=["HOME_TEAM_ID", "AWAY_TEAM_ID", "game_date_norm"],
        right_on=["home_team_id", "away_team_id", "game_date_norm"],
        how="inner",
    )


def bet_record(gap: np.ndarray, realised: np.ndarray, threshold: float) -> dict:
    """Bet the side the gap points to when |gap| >= threshold; realised = actual - line."""
    take = np.abs(gap) >= threshold
    win = np.sign(gap[take]) == np.sign(realised[take])
    push = realised[take] == 0
    n = int(take.sum() - push.sum())
    w = int((win & ~push).sum())
    return {"n": n, "hit": w / n if n else np.nan, "roi": (w * ODDS - n) / n if n else np.nan}


def evaluate(name: str, pred: np.ndarray, line: np.ndarray, actual: np.ndarray) -> dict:
    gap, realised = pred - line, actual - line
    row = {
        "approach": name,
        "mae": float(np.mean(np.abs(actual - pred))),
        "gap_sd": float(np.std(gap)),
        "corr_beyond_line": float(np.corrcoef(gap, realised)[0, 1]) if np.std(gap) > 0 else np.nan,
    }
    for t in THRESHOLDS:
        r = bet_record(gap, realised, t)
        row[f"n_ge{t:g}"], row[f"hit_ge{t:g}"], row[f"roi_ge{t:g}"] = r["n"], r["hit"], r["roi"]
    return row


def run_direction(fit: pd.DataFrame, ev: pd.DataFrame, market: str, feature_cols: list, seed: int) -> list:
    line_c, model_c, actual_c = MARKETS[market]
    fit, ev = fit.dropna(subset=[line_c]), ev.dropna(subset=[line_c])
    rows = [
        evaluate("market", ev[line_c].values, ev[line_c].values, ev[actual_c].values),
        evaluate("model", ev[model_c].values, ev[line_c].values, ev[actual_c].values),
    ]

    g, r = (fit[model_c] - fit[line_c]).values, (fit[actual_c] - fit[line_c]).values
    a = float(np.dot(g, r) / np.dot(g, g))
    rows.append(
        {
            **evaluate(
                "blend",
                ev[line_c].values + a * (ev[model_c] - ev[line_c]).values,
                ev[line_c].values,
                ev[actual_c].values,
            ),
            "a": a,
        }
    )

    cols = feature_cols + ["pm_line", "model_gap"]

    def design(d):
        x = d[feature_cols].copy()
        x["pm_line"] = d[line_c].values
        x["model_gap"] = (d[model_c] - d[line_c]).values
        return x[cols]

    cb = CatBoostRegressor(
        iterations=400,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10,
        loss_function="MAE",
        random_seed=seed,
        verbose=0,
    )
    cb.fit(design(fit), r)
    resid = cb.predict(design(ev))
    rows.append(evaluate("resid", ev[line_c].values + resid, ev[line_c].values, ev[actual_c].values))
    for row in rows:
        row["n_eval"] = len(ev)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(None)
    validate_fold_definitions(cfg.cv.folds)
    fold = next(f for f in cfg.cv.folds if f.name == "fold5")
    print(f"Training champion on fold5 (train through {fold.train_end_date})...", flush=True)
    res = run_split(
        cfg,
        fold.train_end_date,
        fold.validation_start_date,
        fold.validation_end_date,
        fold.test_start_date,
        fold.test_end_date,
        keep_artifacts=True,
    )
    feats = pd.concat([res.val_features, res.test_features], ignore_index=True)
    assert pd.to_datetime(feats["GAME_DATE"]).min() > pd.Timestamp(
        fold.train_end_date
    ), "held-out guarantee violated"

    frame = join_market(
        season_frame(feats, res.predictor, res.feature_cols), load_polymarket_games(Path(args.games_csv))
    )
    seasons = {
        s: frame[frame["game_date_norm"].between(pd.Timestamp(a), pd.Timestamp(b))]
        for s, (a, b) in SEASONS.items()
    }
    print({s: len(d) for s, d in seasons.items()}, flush=True)

    out = []
    for fit_s, ev_s in (("early", "late"), ("late", "early")):
        for market in MARKETS:
            for row in run_direction(seasons[fit_s], seasons[ev_s], market, res.feature_cols, args.seed):
                out.append({"fit": fit_s, "eval": ev_s, "market": market, **row})
    df = pd.DataFrame(out)
    df.to_csv(args.out, index=False)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format
    ):
        print(
            df.drop(columns=[c for c in df.columns if c.startswith("n_ge") and c != "n_ge2"]).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
