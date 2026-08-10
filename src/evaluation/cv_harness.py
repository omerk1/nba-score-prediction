"""
Expanding-window CV harness for the main CatBoost score-predictor pipeline.
See CLAUDE.md's "Project Rules (ML experimentation)" section for the rules
this implements.

`run_split` is the one code path both `--protocol single_split` (today's
default) and every fold of `--protocol cv` go through in train_model.py --
there is exactly one way to train+evaluate a split, not two. Each call
recomputes its own naive baseline fresh from that split's own validation/test
features; nothing is cached or reused across calls, since a shared/stale
normalizer would silently break the composite score's own definition.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_processing.data_loader import load_training_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.utils.config_loader import CVFoldConfig


def naive_baseline_metrics(features_df: pd.DataFrame, y_true: pd.DataFrame, window: int) -> dict:
    """Metrics for a naive predictor that uses each team's rolling avg score
    (`home/away_team_off_eff_L{window}`) as its prediction. THIS split's own
    values -- never a fixed constant carried in from another split/fold.

    Rolling stats are computed per-team across all seasons (no season reset --
    `_add_rolling_features` groups only by team, so a team's window at the
    start of a new season legitimately draws on their last games from the
    prior season, same as train_df has always done). NaN only remains for a
    team's genuine first-ever game in the loaded history (~1% of rows -- the
    real CatBoost model handles this natively across its whole feature set,
    but this is plain numpy, so those rows are dropped here explicitly rather
    than letting a single NaN silently propagate through every mean via
    numpy's default behavior)."""
    home_pred = features_df[f"home_team_off_eff_L{window}"].values
    away_pred = features_df[f"away_team_off_eff_L{window}"].values
    home_true = y_true.iloc[:, 0].values
    away_true = y_true.iloc[:, 1].values

    valid = ~(np.isnan(home_pred) | np.isnan(away_pred))
    if not valid.all():
        home_pred, away_pred = home_pred[valid], away_pred[valid]
        home_true, away_true = home_true[valid], away_true[valid]

    diff_true = home_true - away_true
    diff_pred = home_pred - away_pred
    total_true = home_true + away_true
    total_pred = home_pred + away_pred

    abs_diff_err = np.abs(diff_true - diff_pred)
    residual_std = np.std(diff_true - diff_pred) or 1.0
    return {
        "diff_mae": float(np.mean(abs_diff_err)),
        "diff_within_5": float(np.mean(abs_diff_err <= 5)),
        "total_mae": float(np.mean(np.abs(total_true - total_pred))),
        "win_accuracy": float(np.mean((diff_true > 0) == (diff_pred > 0))),
        "brier_score": float(
            np.mean((norm.cdf(diff_pred / residual_std) - (diff_true > 0).astype(float)) ** 2)
        ),
    }


def compute_composite_score(
    diff_mae: float,
    total_mae: float,
    naive_diff_mae: float,
    naive_total_mae: float,
    lambda_weight: float = 0.5,
) -> float:
    """Primary score (minimize), per CLAUDE.md's Metric section:
    (diff_mae / naive_diff_mae) + lambda_weight * (total_mae / naive_total_mae).
    Both terms normalized by that SAME split's own naive rolling-baseline
    values, so the two MAEs (different typical magnitude, same units) combine
    without an arbitrary scale fix."""
    return (diff_mae / naive_diff_mae) + lambda_weight * (total_mae / naive_total_mae)


def validate_fold_definitions(folds: list[CVFoldConfig]) -> None:
    """Raises ValueError if `folds` (already loaded from configs/config.yaml's
    cv.folds) isn't a valid expanding-window scheme:

    1. Each fold's own dates are chronological and non-overlapping:
       train_end <= validation_start <= validation_end <= test_start <= test_end.
    2. validation_start_date is strictly increasing fold-to-fold -- folds walk
       FORWARD in time (oldest first, newest/largest last), never backward.
    3. No fold's validation/test window predates any EARLIER fold's own
       train_end -- catches a walk-backward ordering mistake mechanically,
       not just by eye.

    (3) is mathematically implied by (1)+(2) -- if every fold is internally
    chronological and validation_start strictly increases fold-to-fold, then
    by transitivity fold[j].validation_start > fold[i].train_end for every
    i<j automatically. It's checked explicitly anyway as defense-in-depth:
    harmless when (1)/(2) hold, and it stays correct even if either of those
    is ever loosened independently later.

    Plain string comparison is safe here: every date in this codebase is an
    ISO "YYYY-MM-DD" string, so lexicographic order == chronological order.
    """
    if not folds:
        raise ValueError("No CV folds defined in configs/config.yaml's cv.folds")

    for f in folds:
        dates = [
            f.train_end_date,
            f.validation_start_date,
            f.validation_end_date,
            f.test_start_date,
            f.test_end_date,
        ]
        if dates != sorted(dates):
            raise ValueError(
                f"Fold {f.name!r} is not chronological: train_end={f.train_end_date}, "
                f"validation=({f.validation_start_date}..{f.validation_end_date}), "
                f"test=({f.test_start_date}..{f.test_end_date})"
            )

    for i in range(1, len(folds)):
        prev, cur = folds[i - 1], folds[i]
        if cur.validation_start_date <= prev.validation_start_date:
            raise ValueError(
                f"Folds must walk forward in time: {cur.name!r}'s validation_start_date "
                f"({cur.validation_start_date}) must be strictly after {prev.name!r}'s "
                f"({prev.validation_start_date}). configs/config.yaml's cv.folds must be "
                f"ordered oldest -> newest."
            )

    for i, cur in enumerate(folds):
        for prev in folds[:i]:
            if cur.validation_start_date <= prev.train_end_date or cur.test_start_date <= prev.train_end_date:
                raise ValueError(
                    f"Fold {cur.name!r}'s validation/test window predates fold {prev.name!r}'s "
                    f"own training window (train_end={prev.train_end_date}) -- a fold whose "
                    f"validation season precedes part of its own (or an earlier fold's) "
                    f"training data is invalid."
                )


@dataclass
class SplitResult:
    train_metrics: dict
    val_metrics: dict
    test_metrics: dict
    naive_val_metrics: dict
    naive_test_metrics: dict
    val_score: float
    test_score: float
    n_features: int
    n_train: int
    n_val: int
    n_test: int
    feature_cols: list = field(default_factory=list)
    # Only populated when keep_artifacts=True (single_split's own side-effect
    # artifacts -- feature CSVs, saved model, predictions preview -- need
    # these; CV mode leaves them None for all 5 folds rather than carrying 5
    # trained models + feature frames it never uses).
    predictor: object = None
    train_features: object = None
    val_features: object = None
    test_features: object = None


def run_split(
    config,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    test_start_date: str,
    test_end_date: str,
    lambda_weight: float = 0.5,
    keep_artifacts: bool = False,
) -> SplitResult:
    """Loads data, builds features, trains one model, evaluates val/test, and
    computes THIS split's own naive baseline + composite score. Used for both
    `--protocol single_split` (today's default) and every fold of
    `--protocol cv` -- one code path, not two.

    `keep_artifacts=True` (single_split only) additionally returns the
    trained predictor and feature DataFrames, so the caller doesn't have to
    re-run feature building a second time just to save CSVs/predictions."""
    train_df, val_df, test_df = load_training_data(
        db_path=config.data_paths.raw_db,
        train_start_date=config.datasets_loading.train_start_date,
        train_end_date=train_end_date,
        val_start_date=validation_start_date,
        val_end_date=validation_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        allowed_season_types=config.datasets_loading.allowed_season_types,
        data_start_date=config.datasets_loading.data_start_date,
        context_season_types=config.datasets_loading.context_season_types,
    )

    feature_builder = FeatureBuilder(
        rolling_windows=config.features.rolling_windows,
        h2h_margin_window=config.features.h2h_margin_window,
        h2h_win_rate_window=config.features.h2h_win_rate_window,
    )
    # context_end_date bounds Elo/season_motivation's own internal history
    # queries to exactly this call's own applicable end date -- a train-time
    # call must never be able to reach into val/test-period games. See
    # FeatureBuilder.create_all_features's docstring.
    train_features = feature_builder.create_all_features(train_df, context_end_date=train_end_date)
    train_features = train_features[
        train_features["GAME_DATE"] >= pd.Timestamp(config.datasets_loading.train_start_date)
    ].reset_index(drop=True)

    val_features = feature_builder.create_all_features(val_df, context_end_date=validation_end_date)
    val_features = val_features[
        (val_features["GAME_DATE"] >= pd.Timestamp(validation_start_date))
        & (val_features["SEASON_TYPE"].isin(config.datasets_loading.allowed_season_types))
    ].reset_index(drop=True)

    test_features = feature_builder.create_all_features(test_df, context_end_date=test_end_date)
    test_features = test_features[
        (test_features["GAME_DATE"] >= pd.Timestamp(test_start_date))
        & (test_features["SEASON_TYPE"].isin(config.datasets_loading.allowed_season_types))
    ].reset_index(drop=True)

    target_cols = config.features.targets
    feature_cols = feature_builder.get_feature_names(train_features)

    X_train, y_train = train_features[feature_cols], train_features[target_cols]
    X_val, y_val = val_features[feature_cols], val_features[target_cols]
    X_test, y_test = test_features[feature_cols], test_features[target_cols]

    predictor = ScorePredictor(
        model_type="catboost",
        iterations=config.model.iterations,
        early_stopping_rounds=config.model.early_stopping_rounds,
        depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=config.model.random_state,
        verbose=False,
        target_formulation=config.model.target_formulation.value,
        target_lambda_weight=config.model.target_lambda_weight,
    )
    train_metrics, val_metrics = predictor.train(X_train, y_train, X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test, dataset_name="Test")

    naive_window = config.features.naive_rolling_baseline
    naive_val_metrics = naive_baseline_metrics(val_features, y_val, naive_window)
    naive_test_metrics = naive_baseline_metrics(test_features, y_test, naive_window)

    val_score = compute_composite_score(
        val_metrics["diff_mae"],
        val_metrics["total_mae"],
        naive_val_metrics["diff_mae"],
        naive_val_metrics["total_mae"],
        lambda_weight,
    )
    test_score = compute_composite_score(
        test_metrics["diff_mae"],
        test_metrics["total_mae"],
        naive_test_metrics["diff_mae"],
        naive_test_metrics["total_mae"],
        lambda_weight,
    )

    return SplitResult(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        naive_val_metrics=naive_val_metrics,
        naive_test_metrics=naive_test_metrics,
        val_score=val_score,
        test_score=test_score,
        n_features=len(feature_cols),
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        feature_cols=feature_cols,
        predictor=predictor if keep_artifacts else None,
        train_features=train_features if keep_artifacts else None,
        val_features=val_features if keep_artifacts else None,
        test_features=test_features if keep_artifacts else None,
    )


@dataclass
class CVResult:
    fold_names: list
    fold_results: list
    val_score_mean: float
    test_score_mean: float


def run_expanding_window_cv(config, lambda_weight: float = 0.5) -> CVResult:
    """Validates fold definitions, then runs `run_split` once per fold in
    `config.cv.folds` (already ordered oldest -> newest), aggregating the
    mean val/test composite score across folds."""
    folds = config.cv.folds if config.cv else []
    validate_fold_definitions(folds)

    fold_results = [
        run_split(
            config,
            f.train_end_date,
            f.validation_start_date,
            f.validation_end_date,
            f.test_start_date,
            f.test_end_date,
            lambda_weight=lambda_weight,
        )
        for f in folds
    ]

    val_scores = [r.val_score for r in fold_results]
    test_scores = [r.test_score for r in fold_results]
    return CVResult(
        fold_names=[f.name for f in folds],
        fold_results=fold_results,
        val_score_mean=sum(val_scores) / len(val_scores),
        test_score_mean=sum(test_scores) / len(test_scores),
    )
