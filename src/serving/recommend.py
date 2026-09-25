"""
Turns a live matchup (plus optional market lines) into a full recommendation:
point prediction, win probability, spread/total cover probability, edge vs.
given market odds, and the margin heatmap. The "serving" half of
docs/features/serving/scope.md (the screenshot-extraction half isn't built
yet -- this takes already-structured picks).

Uses fold5's validation residuals as the calibration sample for every live
game (docs/features/serving/scope.md's "Decisions" section: fold5 is the
most recent already-validated out-of-sample set, and live games are outside
every CV fold either way). Recomputed fresh from data/features/val_features.csv
+ whichever model is currently loaded, rather than a stored constant, so a
future model refresh can't silently go stale against cached residuals.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.predictive_distribution import (
    cover_probability,
    margin_pmf,
    silverman_bandwidth,
    win_probability,
)
from src.models.score_predictor import ScorePredictor
from src.serving.live_features import build_live_game_features
from src.utils.config_loader import load_config


@dataclass
class PredictionResources:
    predictor: ScorePredictor
    diff_residuals: np.ndarray
    diff_bandwidth: float
    total_residuals: np.ndarray
    total_bandwidth: float


def load_resources(config=None) -> PredictionResources:
    """Loads the production model and derives its calibration residuals
    from data/features/val_features.csv (written by the same train_model.py
    run that produced the currently-loaded model.pkl -- both must come from
    the same run, or the residuals won't match the model's own prediction
    behavior)."""
    config = config or load_config()
    model_path = Path(config.data_paths.models) / "score_predictor.pkl"
    predictor = ScorePredictor.load(str(model_path))

    val_features_path = Path("data/features/val_features.csv")
    if not val_features_path.exists():
        raise FileNotFoundError(
            f"{val_features_path} not found -- run train_model.py (--protocol single_split) "
            "first so the currently-loaded model has a matching validation-fold residual sample."
        )
    val_features = pd.read_csv(val_features_path)
    home_col, away_col = config.features.targets[0], config.features.targets[1]
    X_val = val_features[predictor.feature_names]
    val_pred = predictor.predict(X_val)

    diff_true = (val_features[home_col] - val_features[away_col]).to_numpy()
    total_true = (val_features[home_col] + val_features[away_col]).to_numpy()
    diff_residuals = diff_true - (val_pred[:, 0] - val_pred[:, 1])
    total_residuals = total_true - (val_pred[:, 0] + val_pred[:, 1])

    return PredictionResources(
        predictor=predictor,
        diff_residuals=diff_residuals,
        diff_bandwidth=silverman_bandwidth(diff_residuals),
        total_residuals=total_residuals,
        total_bandwidth=silverman_bandwidth(total_residuals),
    )


def decimal_odds_to_probability(odds: float) -> float:
    """Implied probability from decimal (European) odds, e.g. 1.80 -> 0.556."""
    return 1.0 / odds


def recommend_game(
    home_team_id: int,
    away_team_id: int,
    resources: PredictionResources,
    game_date: str = None,
    home_spread: float = None,
    home_spread_odds: float = None,
    away_spread_odds: float = None,
    home_moneyline_odds: float = None,
    away_moneyline_odds: float = None,
    total_line: float = None,
    over_odds: float = None,
    under_odds: float = None,
    heatmap_range: int = 40,
    config=None,
) -> dict:
    """Builds one recommendation dict for a single matchup.

    `home_spread` uses the market's own sign convention (negative = home
    favored by that many points, positive = home getting points), matching
    what a bookmaker screenshot shows printed next to the home team directly
    -- callers don't need to flip signs before passing it in. Odds are
    decimal (e.g. 1.80), not American/fractional -- convert before calling
    if the source uses a different format.

    Every market argument is optional; only the fields whose inputs were
    given are included in the result, so partial information (e.g. a spread
    with no odds yet) still gets a probability, just no edge.
    """
    config = config or load_config()
    game_features = build_live_game_features(home_team_id, away_team_id, game_date, config)
    if game_features.empty:
        raise ValueError(
            f"Not enough game history to build features for {home_team_id} vs {away_team_id}"
            f"{f' on {game_date}' if game_date else ''}."
        )

    X = game_features[resources.predictor.feature_names].iloc[[-1]]
    pred = resources.predictor.predict(X)[0]
    diff_pred = float(pred[0] - pred[1])
    total_pred = float(pred[0] + pred[1])

    result = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "predicted_home_score": round(float(pred[0]), 1),
        "predicted_away_score": round(float(pred[1]), 1),
        "predicted_diff": round(diff_pred, 1),
        "predicted_total": round(total_pred, 1),
        "home_win_probability": win_probability(
            diff_pred, resources.diff_residuals, resources.diff_bandwidth
        ),
    }

    if home_moneyline_odds is not None:
        result["home_moneyline_market_probability"] = decimal_odds_to_probability(home_moneyline_odds)
        result["home_moneyline_edge"] = (
            result["home_win_probability"] - result["home_moneyline_market_probability"]
        )
    if away_moneyline_odds is not None:
        away_win_prob = 1 - result["home_win_probability"]
        result["away_win_probability"] = away_win_prob
        result["away_moneyline_market_probability"] = decimal_odds_to_probability(away_moneyline_odds)
        result["away_moneyline_edge"] = away_win_prob - result["away_moneyline_market_probability"]

    if home_spread is not None:
        home_cover_prob = cover_probability(
            diff_pred, -home_spread, resources.diff_residuals, resources.diff_bandwidth
        )
        result["home_spread"] = home_spread
        result["home_cover_probability"] = home_cover_prob
        result["away_cover_probability"] = 1 - home_cover_prob
        if home_spread_odds is not None:
            result["home_spread_market_probability"] = decimal_odds_to_probability(home_spread_odds)
            result["home_spread_edge"] = home_cover_prob - result["home_spread_market_probability"]
        if away_spread_odds is not None:
            result["away_spread_market_probability"] = decimal_odds_to_probability(away_spread_odds)
            result["away_spread_edge"] = (
                result["away_cover_probability"] - result["away_spread_market_probability"]
            )

    if total_line is not None:
        over_prob = cover_probability(
            total_pred, total_line, resources.total_residuals, resources.total_bandwidth
        )
        result["total_line"] = total_line
        result["over_probability"] = over_prob
        result["under_probability"] = 1 - over_prob
        if over_odds is not None:
            result["over_market_probability"] = decimal_odds_to_probability(over_odds)
            result["over_edge"] = over_prob - result["over_market_probability"]
        if under_odds is not None:
            result["under_market_probability"] = decimal_odds_to_probability(under_odds)
            result["under_edge"] = result["under_probability"] - result["under_market_probability"]

    margins, probs = margin_pmf(
        diff_pred, resources.diff_residuals, resources.diff_bandwidth, lo=-heatmap_range, hi=heatmap_range
    )
    result["margin_heatmap"] = {int(m): round(float(p), 5) for m, p in zip(margins, probs)}

    return result
