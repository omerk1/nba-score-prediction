"""
Intrinsic evaluation of availability estimators (P(plays) for Questionable /
Doubtful listings) against box-score labels.

Protocol: expanding by season. For each evaluation season S from the second
season onward, every estimator is fit on labeled rows from seasons < S and
scored on season S. Metrics are reported per season, pooled over all evaluated
seasons, and on the post-cutoff slice (rows with report_date >= --cutoff,
default 2025-06-01, i.e. after the training cutoff of the default LLM), which
is the only slice where an LLM's number can be trusted against memorization.

Output: one row per (estimator, slice) appended to outputs/availability_eval.csv,
plus a reliability table per estimator on the pooled slice printed to stdout.

Usage:
  venv/bin/python3 scripts/run_availability_eval.py [--tag phase0] [--cutoff 2025-06-01]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.availability.baselines import all_baselines
from src.availability.labels import build_history, build_labels, load_game_log
from src.availability.retrieval import build_context, load_games, load_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = Path("outputs/availability_eval.csv")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {
        "n": int(len(y)),
        "play_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
    }
    # expected calibration error over 10 quantile bins
    bins = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    out["ece"] = float(((g["n"] / len(df)) * (g["y"] - g["p"]).abs()).sum())
    return out


def reliability_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    return df.groupby("b").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injury-db", default="data/raw/injury_features.sqlite")
    ap.add_argument("--cutoff", default="2025-06-01", help="start of the post-LLM-cutoff slice")
    ap.add_argument("--tag", default="phase0")
    ap.add_argument(
        "--context-cache", default=None, help="optional parquet path to cache the built context frame"
    )
    args = ap.parse_args()

    cache = Path(args.context_cache) if args.context_cache else None
    if cache and cache.exists():
        ctx = pd.read_parquet(cache)
        logger.info(f"loaded context from {cache}: {len(ctx)} rows")
    else:
        labels, report = build_labels(args.injury_db)
        logger.info(f"labels: {report} resolved_share={report.resolved_share:.3f}")
        history = build_history(args.injury_db)
        ctx = build_context(labels, history, load_game_log(), load_importance(args.injury_db), load_games())
        if cache:
            ctx.to_parquet(cache)

    seasons = sorted(ctx["season"].unique())
    eval_seasons = seasons[1:]
    logger.info(f"seasons {seasons}; evaluating {eval_seasons}")

    rows, pooled = [], {}
    for est in all_baselines():
        preds = []
        for s in eval_seasons:
            train = ctx[ctx["season"] < s]
            test = ctx[ctx["season"] == s]
            est.fit(train)
            p = est.predict(test)
            preds.append(pd.DataFrame({"idx": test.index, "p": p}))
            rows.append(
                {"estimator": est.name, "slice": f"season_{s}", **metrics(test["played"].to_numpy(), p)}
            )
        pred = pd.concat(preds).set_index("idx")["p"].reindex(ctx.index)
        mask = ctx["season"].isin(eval_seasons)
        pooled[est.name] = pred
        rows.append(
            {
                "estimator": est.name,
                "slice": "pooled",
                **metrics(ctx.loc[mask, "played"].to_numpy(), pred[mask].to_numpy()),
            }
        )
        post = mask & (ctx["report_date"] >= args.cutoff)
        rows.append(
            {
                "estimator": est.name,
                "slice": "post_cutoff",
                **metrics(ctx.loc[post, "played"].to_numpy(), pred[post].to_numpy()),
            }
        )
        for status in ("Questionable", "Doubtful"):
            sm = mask & (ctx["status"] == status)
            rows.append(
                {
                    "estimator": est.name,
                    "slice": f"status_{status}",
                    **metrics(ctx.loc[sm, "played"].to_numpy(), pred[sm].to_numpy()),
                }
            )

    res = pd.DataFrame(rows)
    res.insert(0, "tag", args.tag)
    res.insert(1, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    logger.info(f"appended {len(res)} rows to {OUT_CSV}")

    pd.set_option("display.width", 200)
    show = res[res["slice"].isin(["pooled", "post_cutoff", "status_Questionable", "status_Doubtful"])]
    print(show.pivot(index="estimator", columns="slice", values="brier").round(4).to_string())
    print()
    print(
        res[res["slice"] == "pooled"]
        .set_index("estimator")[["n", "brier", "log_loss", "auc", "ece"]]
        .round(4)
        .to_string()
    )
    mask = ctx["season"].isin(eval_seasons)
    for name, pred in pooled.items():
        print(f"\nreliability ({name}, pooled):")
        print(reliability_table(ctx.loc[mask, "played"].to_numpy(), pred[mask].to_numpy()).to_string())


if __name__ == "__main__":
    main()
