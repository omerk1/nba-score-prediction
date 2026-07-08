"""
A genuinely different paradigm -- a small supervised model trained directly on
the concatenated matchup vector to predict actual_home_margin, instead of "look
up similar historical games and average their margins."

Uses CatBoost (already a project dependency, same usage convention as
train_model.py -- CatBoostRegressor, not a from-scratch implementation) on the
10-dim hand-picked, injury-adjusted (layer=2, per coordinator clarification --
comparable to the 0.281 lookup baseline) matchup vector (home_pace_score...
away_assist_rate) as the ONLY features. No historical similarity search is
involved -- this is direct regression, not nearest-neighbor averaging. Leakage
is still prevented at the vector-construction level (fingerprint.py's pre-game
rolling window discipline, reused unchanged via tuning.py's in-memory pipeline);
no additional leakage risk is introduced by fitting a model on top, since the
model only ever sees pre-game vectors and the target itself is next.

Hyperparameter selection: NOT run through the full Optuna joint search (out of
scope -- the question here is "does this paradigm beat lookup", not "what's the
optimal catboost config for this paradigm"). Instead a small,
fixed, reasonable config is used (depth=4, learning_rate=0.05, iterations=300,
early stopping on an internal chronological dev slice carved out of the tail of
the TRAIN split -- never touching the true validation split during fitting/early-
stopping, so the validation number reported is not tuned against).
"""

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.matchups.split import get_split_dates
from src.matchups.tuning import apply_injury_deltas, build_fingerprints_inmemory, build_index_inmemory, load_constants

logger = logging.getLogger(__name__)


def run_supervised_model(
    window: int = 20, halflife: float = 5.0, layer: int = 2,
    iterations: int = 300, depth: int = 4, learning_rate: float = 0.05,
    internal_dev_frac: float = 0.15, seed: int = 42,
) -> dict:
    consts = load_constants()
    fp1 = build_fingerprints_inmemory(consts["raw"], window=window, halflife=halflife)
    fp = apply_injury_deltas(fp1, consts["delta_lookup"]) if layer == 2 else fp1
    idx = build_index_inmemory(fp, consts["games"])

    vector_cols = [c for c in idx.columns if c.startswith("home_") or c.startswith("away_")]
    splits = get_split_dates()

    train_mask = (idx["game_date"] >= splits["train_start"]) & (idx["game_date"] <= splits["train_end"])
    val_mask = (idx["game_date"] >= splits["validation_start"]) & (idx["game_date"] <= splits["validation_end"])

    train_idx = idx[train_mask].sort_values("game_date").reset_index(drop=True)
    # Internal chronological dev slice (tail of TRAIN only) for early stopping --
    # the true validation split is never used for model selection.
    cutoff_pos = int(len(train_idx) * (1 - internal_dev_frac))
    fit_df = train_idx.iloc[:cutoff_pos]
    dev_df = train_idx.iloc[cutoff_pos:]

    X_fit, y_fit = fit_df[vector_cols], fit_df["actual_home_margin"]
    X_dev, y_dev = dev_df[vector_cols], dev_df["actual_home_margin"]
    X_train_full, y_train_full = train_idx[vector_cols], train_idx["actual_home_margin"]
    X_val, y_val = idx.loc[val_mask, vector_cols], idx.loc[val_mask, "actual_home_margin"]

    model = CatBoostRegressor(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        random_state=seed, verbose=False, early_stopping_rounds=30,
    )
    model.fit(X_fit, y_fit, eval_set=(X_dev, y_dev), use_best_model=True)

    pred_train = model.predict(X_train_full)
    pred_val = model.predict(X_val)

    corr_train = float(np.corrcoef(pred_train, y_train_full)[0, 1])
    corr_val = float(np.corrcoef(pred_val, y_val)[0, 1]) if len(y_val) > 1 else 0.0
    mae_train = float(np.abs(pred_train - y_train_full).mean())
    mae_val = float(np.abs(pred_val - y_val).mean()) if len(y_val) else float("nan")

    importances = dict(zip(vector_cols, model.get_feature_importance().tolist()))

    train_pred_df = train_idx[["game_id", "game_date", "actual_home_margin"]].copy()
    train_pred_df["style_score"] = pred_train
    val_pred_df = idx.loc[val_mask, ["game_id", "game_date", "actual_home_margin"]].copy()
    val_pred_df["style_score"] = pred_val

    return {
        "corr_train": corr_train,
        "corr_validation": corr_val,
        "mae_train": mae_train,
        "mae_validation": mae_val,
        "n_train": len(train_idx),
        "n_fit": len(fit_df),
        "n_internal_dev": len(dev_df),
        "n_validation": len(idx[val_mask]),
        "best_iteration": model.get_best_iteration(),
        "feature_importances": importances,
        "train_pred_df": train_pred_df,
        "val_pred_df": val_pred_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(run_supervised_model(), indent=2, default=str))
