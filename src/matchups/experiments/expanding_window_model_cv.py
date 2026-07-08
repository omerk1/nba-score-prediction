"""
Expanding-window model CV -- does the raw-fingerprint feature redesign's finding
hold across folds, or was it a one-split artifact?

The raw-fingerprint feature redesign (see docs/a7_phase_log.md) found that
`_add_style_fingerprint_features` (raw per-team fingerprint components + explicit
home/away differentials, gated by `style_matchup.raw_features_enabled`) gets real
CatBoost feature importance -- `home_style_pace_score`/`away_style_pace_score`
outrank `elo_diff` outright -- but the accuracy effect was MIXED on the single
static train/validation/test split config.yaml's datasets_loading dates define:
`total_mae` improved clearly (val 14.905 vs 15.070 baseline; test 15.410 vs
15.654), while `win_acc` got slightly WORSE on both splits (val 0.6514 vs 0.6531;
test 0.6637 vs 0.6735).

One static split cannot distinguish "this pattern is robust" from "this pattern
fits the specific calendar quirks of one validation window" -- exactly the
reasoning that motivated `walkforward.py`'s 5-fold walk-forward CV for A7's
standalone similarity-search pipeline (the walk-forward CV check / post-fix
hyperparameter recheck). This module applies the SAME 5 fold boundaries
(`walkforward.FOLDS_WITH_FOLD5` -- train-through-cutoff / validate-on-next-season,
unmodified, reused as-is) to the FULL trained model instead: per fold, build real
features via `FeatureBuilder.create_all_features` and train the actual CatBoost
model via `ScorePredictor`, reusing `train_model.py`'s exact
data-loading/feature-building/training/metric flow and hyperparameters -- nothing
here re-tunes or reimplements any of that, it is called per fold with a
fold-specific date range substituted for `config.yaml`'s single static one.

Two configs per fold (same hyperparameters, only this one difference):
  - "baseline":      style_matchup.enabled=False, raw_features_enabled=False
  - "raw_features":  style_matchup.enabled=False, raw_features_enabled=True
(`enabled`, the OLD KNN-lookup score, stays off in both -- the KNN-score
integration test already settled that path; this module is only about the
raw-fingerprint feature redesign's raw+diff approach.)

No test split per fold: `walkforward.py`'s own fold scheme is train/validate
only (see its module docstring -- it never defines a third held-out split), so
per instructions we report only each fold's validation season, consistent with
how walkforward.py's own folds work. This is NOT the same as the single-split
comparison's val/test rows above -- here "the validation season" is fold-specific.

Toggling `raw_features_enabled` per config WITHOUT touching the committed
configs/config.yaml (which must stay `false` regardless of outcome, and must
never be left mutated mid-run in an unattended session): `_add_style_
fingerprint_features`/`_add_style_matchup_features` both call `load_config()`
with no arguments (hardcoded default path), so this module monkeypatches the
`load_config` NAME inside the `feature_builder` module object at runtime (not
the file) to return an in-memory `model_copy(update=...)` of the real config
with only `style_matchup.raw_features_enabled` flipped -- reversible in-process,
zero risk of leaving the repo dirty even if a fold crashes. `feature_builder.py`
itself is not modified, per instructions.

Placement note: this script trains the full model (not just A7's standalone
similarity-search/fingerprint pipeline), which the task instructions say may
warrant a location outside `src/matchups/experiments/` -- kept here anyway
because its entire purpose is checking the robustness of the raw-fingerprint
feature redesign's finding, it reuses walkforward.py's fold scheme directly, and
every other one-off A7 analysis script already lives in this directory; splitting
this one out would separate it from the finding it's testing for no real benefit.

Outputs (NOT outputs/experiments.csv, the shared production log):
  - outputs/a7_round9_modelcv_results.csv           (one row per fold x config)
  - outputs/a7_round9_feature_importance_fold{N}_{config}.csv (full importance table, every fold x config)
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import src.feature_engineering.feature_builder as feature_builder_module
from src.data_processing.data_loader import load_training_data
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.score_predictor import ScorePredictor
from src.matchups.walkforward import FOLDS_WITH_FOLD5
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("outputs/a7_round9_modelcv_results.csv")
IMPORTANCE_DIR = Path("outputs")

CONFIGS = {
    "baseline": {"enabled": False, "raw_features_enabled": False},
    "raw_features": {"enabled": False, "raw_features_enabled": True},
}


def _make_patched_loader(main_cfg, style_overrides: dict):
    """Returns a zero-arg callable standing in for `load_config` that returns
    `main_cfg` with `style_matchup` fields overridden per `style_overrides` --
    everything else (data_paths, features.exclude, etc.) passes through
    unchanged, since only style_matchup.{enabled,raw_features_enabled} differ
    between the two configs this round compares."""
    new_style = main_cfg.style_matchup.model_copy(update=style_overrides)
    patched_cfg = main_cfg.model_copy(update={"style_matchup": new_style})

    def _loader(*args, **kwargs):
        return patched_cfg

    return _loader


def _run_one(main_cfg, fold: dict, config_name: str) -> dict:
    style_overrides = CONFIGS[config_name]
    original_load_config = feature_builder_module.load_config
    feature_builder_module.load_config = _make_patched_loader(main_cfg, style_overrides)
    try:
        dl = main_cfg.datasets_loading
        train_start_date = dl.train_start_date
        train_end_date = fold["train_end"]
        val_start_date = fold["validation_start"]
        val_end_date = fold["validation_end"]

        # No third split in walkforward.py's fold scheme -- pass val_end_date as a
        # degenerate (empty/near-empty) "test" window purely to satisfy
        # load_training_data's required signature; the returned test_df is never used.
        train_df, val_df, _unused_test_df = load_training_data(
            db_path=main_cfg.data_paths.raw_db,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            test_start_date=val_end_date,
            test_end_date=val_end_date,
            allowed_season_types=dl.allowed_season_types,
            data_start_date=dl.data_start_date,
            context_season_types=dl.context_season_types,
        )

        feature_builder = FeatureBuilder(
            rolling_windows=main_cfg.features.rolling_windows,
            h2h_margin_window=main_cfg.features.h2h_margin_window,
            h2h_win_rate_window=main_cfg.features.h2h_win_rate_window,
        )

        train_features = feature_builder.create_all_features(train_df)
        train_features = train_features[
            train_features["GAME_DATE"] >= pd.Timestamp(train_start_date)
        ].reset_index(drop=True)

        val_features = feature_builder.create_all_features(val_df)
        val_features = val_features[
            (val_features["GAME_DATE"] >= pd.Timestamp(val_start_date)) &
            (val_features["SEASON_TYPE"].isin(dl.allowed_season_types))
        ].reset_index(drop=True)

        target_cols = main_cfg.features.targets
        feature_cols = feature_builder.get_feature_names(train_features)

        X_train = train_features[feature_cols]
        y_train = train_features[target_cols]
        X_val = val_features[feature_cols]
        y_val = val_features[target_cols]

        logger.info(
            f"[fold={fold['fold']} {fold['season']} / {config_name}] "
            f"train={len(X_train):,} val={len(X_val):,} features={len(feature_cols)}"
        )

        predictor = ScorePredictor(
            model_type="catboost",
            iterations=main_cfg.model.iterations,
            early_stopping_rounds=main_cfg.model.early_stopping_rounds,
            depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_state=main_cfg.model.random_state,
            verbose=False,
        )
        _train_metrics, val_metrics = predictor.train(X_train, y_train, X_val, y_val)

        full_importance_df = predictor.get_feature_importance(top_n=len(feature_cols))
        importance_path = IMPORTANCE_DIR / f"a7_round9_feature_importance_fold{fold['fold']}_{config_name}.csv"
        full_importance_df.to_csv(importance_path, index=False)

        return {
            "fold": fold["fold"],
            "season": fold["season"],
            "config": config_name,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(feature_cols),
            "val_diff_mae": round(val_metrics["diff_mae"], 3),
            "val_diff_within_5": round(val_metrics["diff_within_5"], 4),
            "val_total_mae": round(val_metrics["total_mae"], 3),
            "val_win_acc": round(val_metrics["win_accuracy"], 4),
            "val_brier": round(val_metrics["brier_score"], 4),
        }
    finally:
        feature_builder_module.load_config = original_load_config


def run_all() -> pd.DataFrame:
    main_cfg = load_config()
    rows = []
    for fold in FOLDS_WITH_FOLD5:
        for config_name in CONFIGS:
            row = _run_one(main_cfg, fold, config_name)
            rows.append(row)
            logger.info(f"[RESULT] {row}")

    df = pd.DataFrame(rows)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "fold", "season", "config", "n_train", "n_val", "n_features",
            "val_diff_mae", "val_diff_within_5", "val_total_mae", "val_win_acc", "val_brier",
        ])
        if write_header:
            writer.writeheader()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for row in rows:
            writer.writerow({"timestamp": ts, **row})

    logger.info(f"Results saved -> {RESULTS_CSV}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean/std across folds (the key robustness table)."""
    metrics = ["val_diff_mae", "val_diff_within_5", "val_total_mae", "val_win_acc", "val_brier"]
    return df.groupby("config")[metrics].agg(["mean", "std"])


if __name__ == "__main__":
    result = run_all()
    print(result.to_string(index=False))
    print()
    print(summarize(result).to_string())
