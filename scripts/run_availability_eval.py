"""
Intrinsic evaluation of availability estimators (P(plays) for Questionable /
Doubtful listings) against box-score labels.

Protocol: expanding by season. For each evaluation season S from the second
season onward, every trained estimator is fit on labeled rows from seasons < S
and scored on season S. The LLM estimators need no training and are scored on
the same rows. Metrics are reported per season, pooled, per status, and on two
date slices: pre_cutoff and post_cutoff (rows with report_date >= --cutoff,
default 2025-06-01, after the default LLM's training cutoff). Only post_cutoff
is trusted for an LLM; the named-vs-anonymized gap on pre_cutoff is the
memorization check.

Output: one row per (estimator, slice) appended to outputs/availability_eval.csv,
plus a reliability table per estimator on the pooled slice printed to stdout.

Usage:
  venv/bin/python3 scripts/run_availability_eval.py --tag phase0 --estimators baselines
  venv/bin/python3 scripts/run_availability_eval.py --tag pilot --estimators baselines,llm --sample 60
  venv/bin/python3 scripts/run_availability_eval.py --tag phase1 --estimators baselines,llm,llm_named
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()  # GOOGLE_API_KEY for the llm estimators; harmless otherwise

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.availability.baselines import StatusPrior, all_baselines
from src.availability.labels import build_history, build_labels, load_game_log
from src.availability.retrieval import build_context, load_games, load_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = Path("outputs/availability_eval.csv")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0:
        return {
            "n": 0,
            "play_rate": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "auc": np.nan,
            "ece": np.nan,
        }
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {
        "n": int(len(y)),
        "play_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
    }
    bins = pd.qcut(pd.Series(p), min(10, len(p)), labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    out["ece"] = float(((g["n"] / len(df)) * (g["y"] - g["p"]).abs()).sum())
    return out


def reliability_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(pd.Series(p), min(10, len(p)), labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    return df.groupby("b").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)


def build_estimators(names: list[str]) -> list:
    ests = []
    for n in names:
        if n == "baselines":
            ests.extend(all_baselines())
        elif n in ("llm", "llm_named"):
            from src.availability.llm_estimator import LLMEstimator

            ests.append(LLMEstimator(variant="anonymized" if n == "llm" else "named"))
        else:
            raise SystemExit(f"unknown estimator {n}")
    return ests


def load_context(args) -> pd.DataFrame:
    cache = Path(args.context_cache) if args.context_cache else None
    if cache and cache.exists():
        ctx = pd.read_parquet(cache)
        logger.info(f"loaded context from {cache}: {len(ctx)} rows")
        return ctx
    labels, report = build_labels(args.injury_db)
    logger.info(f"labels: {report} resolved_share={report.resolved_share:.3f}")
    history = build_history(args.injury_db)
    ctx = build_context(labels, history, load_game_log(), load_importance(args.injury_db), load_games())
    if cache:
        ctx.to_parquet(cache)
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injury-db", default="data/raw/injury_features.sqlite")
    ap.add_argument("--cutoff", default="2025-06-01", help="start of the post-LLM-cutoff slice")
    ap.add_argument("--tag", default="phase0")
    ap.add_argument("--estimators", default="baselines", help="comma list: baselines,llm,llm_named")
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="evaluate on a random sample of N rows per evaluated season (0 = all); "
        "all estimators are scored on the same rows",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-cache", default=None, help="optional parquet path to cache the context frame")
    args = ap.parse_args()

    ctx = load_context(args)
    seasons = sorted(ctx["season"].unique())
    eval_seasons = seasons[1:]
    logger.info(f"seasons {seasons}; evaluating {eval_seasons}")

    eval_mask = ctx["season"].isin(eval_seasons)
    if args.sample:
        rng = np.random.default_rng(args.seed)
        keep = []
        for s in eval_seasons:
            idx = ctx.index[ctx["season"] == s]
            keep.extend(rng.choice(idx, size=min(args.sample, len(idx)), replace=False))
        eval_mask = ctx.index.isin(keep)
        logger.info(f"sampled {int(eval_mask.sum())} evaluation rows")

    rows, pooled, details = [], {}, {}
    for est in build_estimators(args.estimators.split(",")):
        pred = pd.Series(np.nan, index=ctx.index)
        for s in eval_seasons:
            train = ctx[ctx["season"] < s]
            test_mask = (ctx["season"] == s) & eval_mask
            if not test_mask.any():
                continue
            est.fit(train)
            p = est.predict(ctx[test_mask])
            fallback = StatusPrior().fit(train).predict(ctx[test_mask])
            n_nan = int(np.isnan(p).sum())
            if n_nan:
                logger.warning(f"{est.name} season {s}: {n_nan} missing predictions filled from status prior")
                p = np.where(np.isnan(p), fallback, p)
            pred[test_mask] = p
            rows.append(
                {
                    "estimator": est.name,
                    "slice": f"season_{s}",
                    "n_failed": n_nan,
                    **metrics(ctx.loc[test_mask, "played"].to_numpy(), p),
                }
            )
            if getattr(est, "last_details", None) is not None:
                details.setdefault(est.name, []).append(est.last_details.assign(season=s))
        pooled[est.name] = pred
        y_all = ctx.loc[eval_mask, "played"].to_numpy()
        p_all = pred[eval_mask].to_numpy()
        rows.append({"estimator": est.name, "slice": "pooled", **metrics(y_all, p_all)})
        for name, extra in [
            ("post_cutoff", ctx["report_date"] >= args.cutoff),
            ("pre_cutoff", ctx["report_date"] < args.cutoff),
            ("status_Questionable", ctx["status"] == "Questionable"),
            ("status_Doubtful", ctx["status"] == "Doubtful"),
        ]:
            m = eval_mask & extra
            rows.append(
                {
                    "estimator": est.name,
                    "slice": name,
                    **metrics(ctx.loc[m, "played"].to_numpy(), pred[m].to_numpy()),
                }
            )

    res = pd.DataFrame(rows)
    res.insert(0, "tag", args.tag)
    res.insert(1, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    res.insert(2, "sample", args.sample)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    logger.info(f"appended {len(res)} rows to {OUT_CSV}")

    pd.set_option("display.width", 200)
    show = res[
        res["slice"].isin(["pooled", "pre_cutoff", "post_cutoff", "status_Questionable", "status_Doubtful"])
    ]
    print(show.pivot(index="estimator", columns="slice", values="brier").round(4).to_string())
    print()
    print(
        res[res["slice"] == "pooled"]
        .set_index("estimator")[["n", "brier", "log_loss", "auc", "ece"]]
        .round(4)
        .to_string()
    )
    for name, pred in pooled.items():
        print(f"\nreliability ({name}, pooled):")
        print(
            reliability_table(ctx.loc[eval_mask, "played"].to_numpy(), pred[eval_mask].to_numpy()).to_string()
        )
    for name, parts in details.items():
        d = pd.concat(parts)
        out = Path(f"outputs/availability_llm_details_{args.tag}_{name}.csv")
        ctx.loc[d.index, ["report_date", "game_date", "player_name", "status", "reason", "played"]].join(
            d
        ).to_csv(out, index=False)
        logger.info(f"wrote {len(d)} LLM rows with rationales to {out}")


if __name__ == "__main__":
    main()
