"""
market_benchmark (latest fold's test season) for the baseline and both
second-stage margin corrections (src/models/residual_corrector.py), each
correction fit on every fold's cached validation predictions -- all seasons
before the test season. Wraps scripts/market_benchmark.py without changing it;
one run_split is reused for all three tags.

Usage (repo root, after scripts/run_residual_correction_screen.py --cache <dir>):
    venv/bin/python3 scripts/run_residual_correction_benchmark.py --cache <dir>
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, "."); sys.path.insert(0, "scripts")  # noqa: E702
import market_benchmark as mb
from src.evaluation.residual_correction import fit_on_seasons
from src.models.residual_corrector import build_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
CACHE = ap.parse_args().cache
hist = [pd.read_pickle(f"{CACHE}/fold{k}_val.pkl") for k in range(1, 6)]
orig = mb.generate_holdout_predictions
memo = {}

def cached(cfg, fold):
    if "r" not in memo:
        memo["r"] = orig(cfg, fold)
    return memo["r"][0].copy(), memo["r"][1]

def corrected(use_err):
    def gen(cfg, fold):
        pred_df, result = cached(cfg, fold)
        tf = result.test_features.reset_index(drop=True)
        assert (tf.GAME_ID.values == pred_df.GAME_ID.values).all()
        c = fit_on_seasons(hist, use_err)
        print("coef", dict(zip(c.columns, np.round(c.coef, 3))))
        adj = c.adjustment(build_inputs(tf, pred_df.model_diff_pred.values, use_err))
        pred_df["model_diff_pred"] = pred_df.model_diff_pred.values + adj
        pred_df["model_home_pred"] = (pred_df.model_total_pred + pred_df.model_diff_pred) / 2
        pred_df["model_away_pred"] = (pred_df.model_total_pred - pred_df.model_diff_pred) / 2
        sd = np.std(pred_df.actual_diff - pred_df.model_diff_pred) or 1.0
        pred_df["model_p_home"] = norm.cdf(pred_df.model_diff_pred / sd)
        return pred_df, result
    return gen

for tag, gen in (("rc_baseline", cached), ("rc_rest", corrected(False)), ("rc_rest_team_error", corrected(True))):
    mb.generate_holdout_predictions = gen
    sys.argv = ["market_benchmark.py", "--tag", tag]
    print(f"===== {tag}", flush=True)
    mb.main()
