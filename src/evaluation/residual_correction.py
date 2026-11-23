"""
Out-of-sample predictions the second-stage margin correction
(src/models/residual_corrector.py) is fit on: each CV fold's validation
season, predicted by that fold's model, which never trained on it. Shared by
the screen, the market benchmark and the fit that serving uses, so all three
see the same data. Only calls run_split; the harness itself is unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.cv_harness import run_split
from src.models.residual_corrector import ResidualCorrector, build_inputs

KEEP = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON_ID",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "POINT_DIFF",
    "home_team_back_to_back",
    "away_team_back_to_back",
    "home_team_rest_days",
    "away_team_rest_days",
    "home_team_games_in_4_nights",
    "away_team_games_in_4_nights",
]
ALPHA = 10.0


def fold_validation_predictions(cfg, fold, cache: Path) -> tuple[pd.DataFrame, dict]:
    """(validation rows with pred_diff/pred_total/act_total, the fold's val
    score and naive baselines), cached per fold in `cache`."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    p, meta_p = cache / f"{fold.name}_val.pkl", cache / f"{fold.name}_meta.json"
    if p.exists() and meta_p.exists():
        return pd.read_pickle(p), json.loads(meta_p.read_text())
    r = run_split(
        cfg,
        fold.train_end_date,
        fold.validation_start_date,
        fold.validation_end_date,
        fold.test_start_date,
        fold.test_end_date,
        keep_artifacts=True,
    )
    vf = r.val_features
    pred = r.predictor.predict(vf[r.feature_cols])
    df = vf[KEEP].copy().reset_index(drop=True)
    df["pred_diff"] = pred[:, 0] - pred[:, 1]
    df["pred_total"] = pred[:, 0] + pred[:, 1]
    y = vf[cfg.features.targets].values
    df["act_total"] = y[:, 0] + y[:, 1]
    meta = {
        "val_score": r.val_score,
        "naive_diff_mae": r.naive_val_metrics["diff_mae"],
        "naive_total_mae": r.naive_val_metrics["total_mae"],
    }
    df.to_pickle(p)
    meta_p.write_text(json.dumps(meta))
    return df, meta


def fit_on_seasons(
    seasons: list[pd.DataFrame], use_team_error: bool, alpha: float = ALPHA
) -> ResidualCorrector:
    """Fits the correction on several seasons of out-of-sample predictions
    (each with pred_diff), team errors computed within each season."""
    X = pd.concat([build_inputs(s, s.pred_diff, use_team_error) for s in seasons], ignore_index=True)
    residual = np.concatenate([s.POINT_DIFF - s.pred_diff for s in seasons])
    return ResidualCorrector.fit(X, residual, alpha=alpha)
