"""
Screen the second-stage margin correction (src/models/residual_corrector.py)
on folds 4-5 (--first-fold 2 adds folds 2-3).

For fold k the correction is fit on the validation predictions of folds < k
(seasons inside fold k's training window, each predicted by a model that never
trained on it) and applied to fold k's own validation predictions. The CV
harness itself is unchanged: this only calls run_split and
compute_composite_score. Test folds are never used.

Usage (repo root):
    venv/bin/python3 scripts/run_residual_correction_screen.py --cache <dir> [--session-csv <path>]
Validation predictions are cached per fold in --cache, so reruns only refit the
correction.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.cv_harness import compute_composite_score  # noqa: E402
from src.evaluation.residual_correction import fit_on_seasons, fold_validation_predictions  # noqa: E402
from src.models.residual_corrector import build_inputs  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

VARIANTS = {"rest": False, "rest_team_error": True}


def team_side(df: pd.DataFrame, pred_diff) -> pd.DataFrame:
    """One row per team per game: signed margin error from that team's side."""
    err = df["POINT_DIFF"].values - np.asarray(pred_diff)
    n = np.arange(len(df))
    t = pd.concat(
        [
            pd.DataFrame(
                {
                    "row": n,
                    "date": df.GAME_DATE.values,
                    "season": df.SEASON_ID.values,
                    "team": df.HOME_TEAM_ID.values,
                    "err": err,
                    "b2b": df.home_team_back_to_back.values,
                }
            ),
            pd.DataFrame(
                {
                    "row": n,
                    "date": df.GAME_DATE.values,
                    "season": df.SEASON_ID.values,
                    "team": df.AWAY_TEAM_ID.values,
                    "err": -err,
                    "b2b": df.away_team_back_to_back.values,
                }
            ),
        ]
    ).sort_values(["date", "row"], kind="stable")
    t["n"] = t.groupby(["season", "team"]).cumcount()
    return t


def bias_metrics(df: pd.DataFrame, pred_diff) -> dict:
    t = team_side(df, pred_diff)
    ts = t.groupby(["season", "team"]).err
    noise_var = (ts.var() / ts.size()).mean()
    half = t.n.groupby([t.season, t.team]).transform("size") // 2
    first = t[t.n < half].groupby(["season", "team"]).err.mean()
    second = t[t.n >= half].groupby(["season", "team"]).err.mean()
    return {
        "b2b_bias": t[t.b2b == 1].err.mean(),
        "team_excess_sd": float(np.sqrt(max(ts.mean().var() - noise_var, 0))),
        "team_half_r": first.corr(second),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--session-csv")
    ap.add_argument("--session-id", default="")
    # fold1 has no earlier fold to fit on; folds 2-3 were also the seasons the bias scan was run on.
    ap.add_argument("--first-fold", type=int, default=4)
    args = ap.parse_args()
    cache = Path(args.cache)

    cfg = load_config()
    folds = cfg.cv.folds
    preds = {f.name: fold_validation_predictions(cfg, f, cache) for f in folds}

    rows = []
    for k in range(args.first_fold - 1, len(folds)):
        fold = folds[k]
        val, meta = preds[fold.name]
        hist = [preds[f.name][0] for f in folds[:k]]
        diff_mae_base = np.abs(val.POINT_DIFF - val.pred_diff).mean()
        total_mae = np.abs(val.act_total - val.pred_total).mean()
        base_score = compute_composite_score(
            diff_mae_base, total_mae, meta["naive_diff_mae"], meta["naive_total_mae"]
        )
        assert abs(base_score - meta["val_score"]) < 1e-6, (base_score, meta["val_score"])
        rows.append(
            {
                "variant": "baseline",
                "fold": fold.name,
                "val_score": base_score,
                "diff_mae": diff_mae_base,
                "total_mae": total_mae,
                "coef": "",
                **bias_metrics(val, val.pred_diff),
            }
        )
        for name, use_err in VARIANTS.items():
            corr = fit_on_seasons(hist, use_err)
            new_diff = val.pred_diff + corr.adjustment(build_inputs(val, val.pred_diff, use_err))
            diff_mae = np.abs(val.POINT_DIFF - new_diff).mean()
            score = compute_composite_score(
                diff_mae, total_mae, meta["naive_diff_mae"], meta["naive_total_mae"]
            )
            rows.append(
                {
                    "variant": name,
                    "fold": fold.name,
                    "val_score": score,
                    "diff_mae": diff_mae,
                    "total_mae": total_mae,
                    "coef": json.dumps(dict(zip(corr.columns, np.round(corr.coef, 3)))),
                    **bias_metrics(val, new_diff),
                }
            )
    out = pd.DataFrame(rows)
    base = out[out.variant == "baseline"].set_index("fold").val_score
    out["delta"] = out.val_score - out.fold.map(base)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    if args.session_csv:
        out.assign(session_id=args.session_id).to_csv(args.session_csv, index=False)


if __name__ == "__main__":
    main()
