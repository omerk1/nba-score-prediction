"""
Runs the second-stage margin correction (src/models/residual_corrector.py)
next to the live model, so it can be judged on real picks before it changes
any. Each recommendation gets a `residual_correction` entry: the adjustment
and its parts, the corrected margin, and the spread/moneyline sides priced
off it by the same code as the live ones (totals aren't corrected). The slate
shows the corrected model's own full set of bets; the delivered picks, edge
calibration and grading are unchanged.

Point-in-time: each team's recent error comes from re-predicting this
season's already-played games with the live model (trained through
2023-24, so every such game is out of sample for it, as in the fit).
"""

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.predictive_distribution import silverman_bandwidth
from src.models.residual_corrector import (
    ERROR_MIN_GAMES,
    ERROR_WINDOW,
    TEAM_ERROR_INPUT,
    ResidualCorrector,
    build_inputs,
    latest_team_errors,
    rest_inputs,
)

logger = logging.getLogger(__name__)

@dataclass
class CorrectionResources:
    corrector: ResidualCorrector
    team_errors: dict  # team id -> mean margin error over its last games this season
    diff_residuals: np.ndarray  # corrected validation-season residuals, for pricing
    diff_bandwidth: float
    as_of: str  # team errors cover games before this date


def season_start(day: datetime.date) -> datetime.date:
    """Aug 1 of the season `day` falls in (NBA seasons start in October)."""
    return datetime.date(day.year if day.month >= 8 else day.year - 1, 8, 1)


def played_this_season(predictor, config, as_of: datetime.date) -> pd.DataFrame:
    """This season's regular-season games before `as_of`, built through the
    same feature pipeline as training, with the live model's pred_diff."""
    from src.data_processing.data_loader import NBADataLoader
    from src.feature_engineering.feature_builder import FeatureBuilder

    end = str(as_of - datetime.timedelta(days=1))
    dl = config.datasets_loading
    loader = NBADataLoader(db_path=config.data_paths.raw_db)
    try:
        history = loader.load_games(
            start_date=dl.data_start_date,
            end_date=end,
            allowed_season_types=dl.context_season_types or dl.allowed_season_types,
        )
    finally:
        loader.close()
    if history.empty:
        return history
    fb = FeatureBuilder(
        rolling_windows=config.features.rolling_windows,
        h2h_margin_window=config.features.h2h_margin_window,
        h2h_win_rate_window=config.features.h2h_win_rate_window,
    )
    feats = fb.create_all_features(history, context_end_date=end)
    played = feats[
        (pd.to_datetime(feats["GAME_DATE"]) >= pd.Timestamp(season_start(as_of)))
        & (pd.to_datetime(feats["GAME_DATE"]) < pd.Timestamp(as_of))
        & feats["SEASON_TYPE"].isin(dl.allowed_season_types)
        & feats["POINT_DIFF"].notna()
    ].reset_index(drop=True)
    if played.empty:
        return played
    pred = predictor.predict(played[predictor.feature_names])
    return played.assign(pred_diff=pred[:, 0] - pred[:, 1])


def load_correction(
    predictor, config, val_features: pd.DataFrame, val_pred_diff, as_of: datetime.date | None = None
) -> CorrectionResources | None:
    """None when the correction is off, its coefficients were fit for a
    different model, or anything fails (logged; the slate goes out without it)."""
    from src.serving.prediction_log import _model_trained_at

    rc = getattr(config, "residual_correction", None)
    if rc is None or rc.enabled is not True:
        return None
    try:
        spec = json.loads(Path(rc.coefficients_path).read_text())
        trained_at = _model_trained_at()
        if spec.get("model_trained_at") != trained_at:
            logger.warning(
                f"residual correction skipped: fit for the model trained {spec.get('model_trained_at')}, "
                f"live model trained {trained_at}; rerun scripts/fit_residual_corrector.py"
            )
            return None
        if (spec.get("error_window"), spec.get("error_min_games")) != (ERROR_WINDOW, ERROR_MIN_GAMES):
            logger.warning("residual correction skipped: coefficients fit with a different team-error window")
            return None
        corrector = ResidualCorrector.from_dict(spec)
        use_err = TEAM_ERROR_INPUT in corrector.columns

        # Pricing spread: the corrected model's own errors on the validation season.
        val_pred_diff = np.asarray(val_pred_diff, dtype=float)
        adj = corrector.adjustment(build_inputs(val_features, val_pred_diff, use_err))
        residuals = val_features["POINT_DIFF"].to_numpy(dtype=float) - (val_pred_diff + adj)

        as_of = as_of or datetime.date.today()
        team_errors = {}
        if use_err:
            played = played_this_season(predictor, config, as_of)
            team_errors = latest_team_errors(played, played["pred_diff"]) if not played.empty else {}
            logger.info(f"residual correction: team errors from {len(played)} games before {as_of}")
        return CorrectionResources(
            corrector=corrector,
            team_errors=team_errors,
            diff_residuals=residuals,
            diff_bandwidth=silverman_bandwidth(residuals),
            as_of=str(as_of),
        )
    except Exception as e:
        logger.error(f"residual correction unavailable today: {e}")
        return None


def corrected_entry(
    game_row: pd.DataFrame,
    diff_pred: float,
    corr: CorrectionResources,
    gap_weights: dict,
    margin_markets: dict,
) -> dict:
    """The `residual_correction` entry for one game: `game_row` is its live
    feature row, `margin_markets` the spread/moneyline lines and odds."""
    from src.serving.recommend import price_margin_markets

    X = rest_inputs(game_row)
    if TEAM_ERROR_INPUT in corr.corrector.columns:
        home, away = int(game_row["HOME_TEAM_ID"].iloc[0]), int(game_row["AWAY_TEAM_ID"].iloc[0])
        X[TEAM_ERROR_INPUT] = corr.team_errors.get(home, 0.0) - corr.team_errors.get(away, 0.0)
    inputs = {c: float(X[c].iloc[0]) for c in corr.corrector.columns}
    parts = {c: inputs[c] * coef for c, coef in zip(corr.corrector.columns, corr.corrector.coef)}
    adjustment = float(sum(parts.values()))
    new_diff = diff_pred + adjustment
    return {
        "adjustment": round(adjustment, 2),
        "parts": {c: round(v, 2) for c, v in parts.items()},
        "inputs": inputs,
        "predicted_diff": round(new_diff, 1),
        "team_errors_as_of": corr.as_of,
        **price_margin_markets(
            new_diff,
            corr.diff_residuals,
            corr.diff_bandwidth,
            gap_weights.get("spread", 1.0),
            **margin_markets,
        ),
    }


def corrected_rec(rec: dict) -> dict | None:
    """The recommendation as the corrected model would price it (totals as
    is), or None when this game has no correction."""
    entry = rec.get("residual_correction")
    return None if entry is None else {**rec, **entry}
