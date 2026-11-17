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
from src.evaluation.cv_harness import compute_composite_score, run_split  # noqa: E402
from src.models.residual_corrector import ResidualCorrector, build_inputs  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

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
VARIANTS = {"rest": False, "rest_team_error": True}
ALPHA = 10.0


def fold_predictions(cfg, fold, cache: Path) -> tuple[pd.DataFrame, dict]:
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
    cache.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    folds = cfg.cv.folds
    preds = {f.name: fold_predictions(cfg, f, cache) for f in folds}

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
            Xh = pd.concat([build_inputs(h, h.pred_diff, use_err) for h in hist], ignore_index=True)
            rh = np.concatenate([h.POINT_DIFF - h.pred_diff for h in hist])
            corr = ResidualCorrector.fit(Xh, rh, alpha=ALPHA)
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
