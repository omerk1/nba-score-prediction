"""CV-integrated CatBoost hyperparameter tuning.

Supersedes scripts/tune_model.py for hyperparameter search: that script is
broken today (imports train_model._naive_baseline_metrics, removed when the
CV harness was built) and, even before that, optimized raw single-split
validation diff_mae rather than the composite score aggregated across CV
folds -- both violate this project's current "Model selection and tuning
use aggregated VALIDATION scores only" hard constraint (CLAUDE.md). Two of
its six searched hyperparameters (l2_leaf_reg, min_data_in_leaf) also never
reached CatBoost at all -- ScorePredictor._create_model silently dropped
them; fixed alongside this script (src/models/score_predictor.py).

Reuses configs/config.yaml's existing model.tuning bounds (depth,
learning_rate, l2_leaf_reg, min_data_in_leaf, subsample, colsample_bylevel)
-- already thoughtfully set, no need to invent new ranges. Every trial
config is built via pydantic's model_copy(update=...); configs/config.yaml
on disk is never touched.

Two-stage protocol, per CLAUDE.md's "Project Rules (ML experimentation)":
  1. SCREEN: --n-trials Optuna/TPE trials, each scored by val_score_mean
     aggregated over the last --screen-folds folds only (cheap-screening
     allowance). Every trial logged to results/sessions/<session_id>.csv.
  2. CONFIRM: the top --confirm-top-n screening trials (plus the untouched
     config as an explicit same-run baseline) are re-scored under FULL
     5-fold CV -- "full CV required before an experiment is promoted or
     declared a new best." Only confirmed rows can be promoted.

At session end, appends to outputs/experiments_v2.csv (a) the best
confirmed row by full-CV val_score_mean and (b) any other confirmed row
that beats the baseline -- everything else stays in the session CSV only.

Usage:
    venv/bin/python3 scripts/tune_model_cv.py --run-name hp-tuning-cv --n-trials 30
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.cv_harness import run_expanding_window_cv, validate_fold_definitions  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from train_model import _save_experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _aggregate_cv_metrics(cv_result) -> tuple:
    """Mean val/test metrics dicts + per-fold val_score string, same
    aggregation pattern train_model._run_cv already uses for its own
    experiments_v2.csv row."""
    n = len(cv_result.fold_results)
    keys = ["diff_mae", "diff_within_5", "total_mae", "win_accuracy", "brier_score"]
    val_metrics = {k: sum(r.val_metrics[k] for r in cv_result.fold_results) / n for k in keys}
    test_metrics = {k: sum(r.test_metrics[k] for r in cv_result.fold_results) / n for k in keys}
    val_score_per_fold = ",".join(f"{r.val_score:.4f}" for r in cv_result.fold_results)
    return val_metrics, test_metrics, val_score_per_fold


def _sample_params(trial: optuna.Trial, tuning) -> dict:
    return {
        "depth": trial.suggest_int("depth", *tuning.depth),
        "learning_rate": trial.suggest_float("learning_rate", *tuning.learning_rate, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *tuning.l2_leaf_reg, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *tuning.min_data_in_leaf),
        "subsample": trial.suggest_float("subsample", *tuning.subsample),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *tuning.colsample_bylevel),
    }


def _build_trial_config(cfg, params: dict, folds: list):
    return cfg.model_copy(
        update={
            "model": cfg.model.model_copy(update=params),
            "cv": cfg.cv.model_copy(update={"folds": folds}),
        }
    )


def _log_row(session_csv: Path, run_name: str, notes: str, cfg, cv_result, session_id: str) -> None:
    val_metrics, test_metrics, val_score_per_fold = _aggregate_cv_metrics(cv_result)
    _save_experiment(
        run_name,
        notes,
        cfg,
        val_metrics,
        test_metrics,
        n_features=cv_result.fold_results[-1].n_features,
        protocol="cv",
        session_id=session_id,
        val_score_mean=cv_result.val_score_mean,
        val_score_per_fold=val_score_per_fold,
        test_score_mean=cv_result.test_score_mean,
        experiments_csv=str(session_csv),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", required=True, help="Session slug, e.g. 'hp-tuning-cv'")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--screen-folds", type=int, default=3, help="Last N folds for cheap screening")
    parser.add_argument(
        "--confirm-top-n", type=int, default=2, help="Top screening trials to confirm on full CV"
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--experiments-csv", default="outputs/experiments_v2.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Resume a previous session (its results/sessions/<id>.csv and Optuna storage) instead of "
        "starting a fresh one -- e.g. after an interruption. Default: fresh YYYYMMDD_HHMM_<run-name>.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_fold_definitions(cfg.cv.folds)
    if cfg.model.tuning is None:
        logger.error("No 'tuning' section found in config.yaml under 'model'. Add it before running.")
        return

    session_id = args.session_id or f"{datetime.now(timezone.utc):%Y%m%d_%H%M}_{args.run_name}"
    session_csv = Path("results/sessions") / f"{session_id}.csv"
    session_csv.parent.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///results/sessions/{session_id}_optuna.db"
    logger.info(
        f"Session {session_id!r} -- logging every trial to {session_csv}, Optuna storage {optuna_storage}"
    )

    screen_folds = cfg.cv.folds[-args.screen_folds :]
    logger.info(
        f"Screening on last {args.screen_folds} fold(s): {[f.name for f in screen_folds]} "
        f"(full {len(cfg.cv.folds)}-fold CV reserved for confirming the top {args.confirm_top_n} trials)"
    )

    baseline_run_name = f"{args.run_name}_baseline"
    baseline_score = None
    if session_csv.exists():
        existing = pd.read_csv(session_csv)
        match = existing[existing["run_name"] == baseline_run_name]
        if len(match):
            baseline_score = float(match.iloc[0]["val_score_mean"])
            logger.info(f"Resuming: baseline already logged, val_score_mean = {baseline_score:.4f}")
    if baseline_score is None:
        logger.info("Running untouched config as this session's own same-run baseline (full CV)...")
        baseline_result = run_expanding_window_cv(cfg)
        _log_row(
            session_csv,
            baseline_run_name,
            "same-run baseline, untouched config",
            cfg,
            baseline_result,
            session_id,
        )
        baseline_score = baseline_result.val_score_mean
        logger.info(f"Baseline val_score_mean = {baseline_score:.4f}")

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, cfg.model.tuning)
        trial_config = _build_trial_config(cfg, params, screen_folds)
        cv_result = run_expanding_window_cv(trial_config)
        notes = f"screen ({args.screen_folds}-fold): {params}" + (f" | {args.notes}" if args.notes else "")
        _log_row(
            session_csv,
            f"{args.run_name}_trial{trial.number}",
            notes,
            trial_config,
            cv_result,
            session_id,
        )
        return cv_result.val_score_mean

    study = optuna.create_study(
        study_name=args.run_name, direction="minimize", storage=optuna_storage, load_if_exists=True
    )
    n_done = len(study.trials)
    if n_done:
        logger.info(f"Resuming study: {n_done} trial(s) already complete, running {args.n_trials} more")
    logger.info(f"Starting Optuna study: {args.n_trials} trials")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    ranked = sorted(study.trials, key=lambda t: t.value)
    top = ranked[: args.confirm_top_n]
    logger.info(
        f"Top {len(top)} screening trial(s) by val_score_mean: "
        + ", ".join(f"trial{t.number}={t.value:.4f}" for t in top)
    )

    confirmed = []
    for t in top:
        trial_config = _build_trial_config(cfg, t.params, cfg.cv.folds)
        cv_result = run_expanding_window_cv(trial_config)
        run_name = f"{args.run_name}_confirm_trial{t.number}"
        _log_row(
            session_csv,
            run_name,
            f"full 5-fold confirmation: {t.params}",
            trial_config,
            cv_result,
            session_id,
        )
        confirmed.append((run_name, t.params, trial_config, cv_result))
        logger.info(f"  Confirmed trial{t.number}: full-CV val_score_mean = {cv_result.val_score_mean:.4f}")

    print("\n" + "=" * 78)
    print(f"SESSION SUMMARY -- {session_id}")
    print("=" * 78)
    print(f"Baseline (untouched config, full CV): val_score_mean = {baseline_score:.4f}")
    print(f"\nScreening ({args.screen_folds}-fold), all {len(ranked)} trials, best to worst:")
    for t in ranked:
        print(f"  trial{t.number}: val_score_mean={t.value:.4f}  params={t.params}")
    print(f"\nConfirmed (full 5-fold CV):")
    for run_name, params, _, cv_result in confirmed:
        delta = cv_result.val_score_mean - baseline_score
        print(
            f"  {run_name}: val_score_mean={cv_result.val_score_mean:.4f}  delta={delta:+.4f}  params={params}"
        )
    print("=" * 78)

    best_run_name, best_params, best_config, best_result = min(confirmed, key=lambda c: c[3].val_score_mean)
    _log_row(
        Path(args.experiments_csv),
        best_run_name,
        f"best confirmed, session {session_id}: {best_params}",
        best_config,
        best_result,
        session_id,
    )
    logger.info(f"Logged best confirmed row ({best_run_name}) to {args.experiments_csv}")

    for run_name, params, trial_config, cv_result in confirmed:
        if run_name == best_run_name:
            continue
        if cv_result.val_score_mean < baseline_score:
            _log_row(
                Path(args.experiments_csv),
                run_name,
                f"beats baseline, session {session_id}: {params}",
                trial_config,
                cv_result,
                session_id,
            )
            logger.info(f"Logged additional champion-beating row ({run_name}) to {args.experiments_csv}")

    if best_result.val_score_mean < baseline_score:
        logger.info(
            f"Best confirmed val_score_mean ({best_result.val_score_mean:.4f}) beats baseline "
            f"({baseline_score:.4f}) -- consider promoting these hyperparameters to configs/config.yaml's "
            f"model section after reviewing per-fold deltas (CLAUDE.md's fold-majority guardrail)."
        )
    else:
        logger.info(
            f"Best confirmed val_score_mean ({best_result.val_score_mean:.4f}) does not beat baseline "
            f"({baseline_score:.4f}) -- no promotion. configs/config.yaml's current hyperparameters stand."
        )


if __name__ == "__main__":
    main()
