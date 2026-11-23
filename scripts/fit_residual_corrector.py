"""
Fits the second-stage margin correction that serving shows alongside the
live picks (src/serving/residual_shadow.py), and saves it next to the model.

Fit on every CV fold's validation predictions (seasons 2020-21 to 2024-25,
each from a model that never trained on it) with the selected variant (rest
plus each team's recent error), exactly as scripts/run_residual_correction_benchmark.py
did. The file is tagged with the live model's train_date: serving skips the
correction when the model has been retrained since, so rerun this after every
train_model.py run that replaces data/models/score_predictor.pkl.

Usage (repo root):
    venv/bin/python3 scripts/fit_residual_corrector.py --cache <dir>
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.residual_correction import (  # noqa: E402
    ALPHA,
    fit_on_seasons,
    fold_validation_predictions,
)
from src.models.residual_corrector import ERROR_MIN_GAMES, ERROR_WINDOW  # noqa: E402
from src.serving.prediction_log import _model_trained_at  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="directory for per-fold validation predictions")
    args = ap.parse_args()

    cfg = load_config()
    rc = cfg.residual_correction
    trained_at = _model_trained_at()
    if trained_at is None:
        sys.exit("data/models/training_metadata.json missing: train the live model first")

    seasons = [fold_validation_predictions(cfg, f, Path(args.cache))[0] for f in cfg.cv.folds]
    corrector = fit_on_seasons(seasons, use_team_error=True, alpha=ALPHA)
    out = {
        **corrector.to_dict(),
        "alpha": ALPHA,
        "error_window": ERROR_WINDOW,
        "error_min_games": ERROR_MIN_GAMES,
        "fitted_on": [f.name for f in cfg.cv.folds],
        "fit_games": int(sum(len(s) for s in seasons)),
        "model_trained_at": trained_at,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    path = Path(rc.coefficients_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
