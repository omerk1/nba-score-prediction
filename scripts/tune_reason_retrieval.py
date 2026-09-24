"""
Tune and then test semantic retrieval over injury-report reason text.

Protocol, set up before any knob was turned:

- TUNING seasons (2022-23, 2023-24, 2024-25) are the only data the sweep sees.
  Every retrieval parameter is chosen on them.
- The 2025-26 season is SEALED. It is scored once, after the sweep has chosen,
  and it also falls after the embedding model's training cutoff.
- Within either split, a season is predicted by a model fit on earlier seasons
  only, so nothing is ever scored on data it was fit on.

The question is not whether the retrieved feature predicts on its own; it is
whether adding it to the tabular model that already wins improves that model.
So every setting is scored as the Brier delta of CatBoost with the new columns
against the identical CatBoost without them, on the same rows.

Usage:
  venv/bin/python3 scripts/tune_reason_retrieval.py --sweep
  venv/bin/python3 scripts/tune_reason_retrieval.py --test --k 25 --min-sim 0.9 --power 3 --prior-weight 10
"""

import argparse
import itertools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.availability.baselines import CatBoostBaseline
from src.availability.labels import DATED_DB_PATH, build_history_dated, build_labels_dated, load_game_log
from src.availability.reason_retrieval import (
    FEATURE_COLUMNS,
    GeminiEmbedder,
    ReasonIndex,
    RetrievalParams,
    build_reason_features,
)
from src.availability.retrieval import (
    CONTEXT_COLUMNS,
    build_context,
    load_games,
    load_importance,
    normalize_reason,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TUNING_SEASONS = ["2022-23", "2023-24", "2024-25"]
SEALED_SEASON = "2025-26"
OUT_CSV = Path("outputs/reason_retrieval_tuning.csv")


def representative_texts(history: pd.DataFrame) -> dict[str, str]:
    """normalized reason -> its most common readable spelling.

    Normalization collapses the PDF's spacing variants, but the normalized form
    ("injuryillnessrightanklesprain") is poor input for an embedding model. Key
    on the normalized form, embed the readable one.
    """
    h = history.assign(norm=history["reason"].map(normalize_reason))
    h = h[h["norm"] != ""]
    counts = h.groupby(["norm", "reason"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("norm")
    return dict(zip(best["norm"], best["reason"]))


def load_all(injury_db: str) -> tuple[pd.DataFrame, pd.DataFrame, ReasonIndex]:
    labels, report = build_labels_dated(injury_db)
    logger.info(f"labels: {report}")
    history = build_history_dated(injury_db)
    logger.info(f"history: {len(history)} listings (Out included)")

    # 21 rows resolve to no player league-wide; they are all played=0 but have no
    # minutes history, so the context builder cannot describe them. Dropped here
    # and counted, rather than silently.
    keep = labels["player_id"].notna()
    if (~keep).any():
        logger.info(f"dropping {int((~keep).sum())} rows with no resolvable player for the context build")
    labels = labels[keep].copy()
    labels["player_id"] = labels["player_id"].astype(int)
    hist_ctx = history[history["player_id"].notna()].copy()
    hist_ctx["player_id"] = hist_ctx["player_id"].astype(int)

    ctx = build_context(
        labels, hist_ctx, load_game_log(), load_importance("data/raw/injury_features.sqlite"), load_games()
    )

    reps = representative_texts(history)
    vectors_raw = GeminiEmbedder().embed(list(reps.values()))
    vectors = {norm: vectors_raw[raw] for norm, raw in reps.items() if raw in vectors_raw}
    logger.info(f"reason vocabulary: {len(vectors)} embedded")
    return ctx, history, ReasonIndex(vectors)


def score_setting(
    ctx: pd.DataFrame, history: pd.DataFrame, index: ReasonIndex, params: RetrievalParams, seasons: list[str]
) -> dict:
    """Brier for CatBoost with and without the retrieved columns, same rows."""
    feats = build_reason_features(ctx, history, index, params)
    full = pd.concat([ctx, feats], axis=1)

    base_p = pd.Series(np.nan, index=full.index)
    aug_p = pd.Series(np.nan, index=full.index)
    for s in seasons:
        train = full[full["season"] < s]
        test = full[full["season"] == s]
        if train.empty or test.empty:
            continue
        base_p[test.index] = CatBoostBaseline(columns=CONTEXT_COLUMNS).fit(train).predict(test)
        aug_p[test.index] = (
            CatBoostBaseline(columns=CONTEXT_COLUMNS + FEATURE_COLUMNS).fit(train).predict(test)
        )

    m = full["season"].isin(seasons) & base_p.notna()
    y = full.loc[m, "played"].to_numpy()
    b, a = base_p[m].to_numpy(), aug_p[m].to_numpy()
    cov = float(feats.loc[m, "reason_nbr_play_rate"].notna().mean())
    return {
        **params.__dict__,
        "n": int(m.sum()),
        "coverage": cov,
        "base_brier": float(brier_score_loss(y, b)),
        "aug_brier": float(brier_score_loss(y, a)),
        "delta": float(brier_score_loss(y, a) - brier_score_loss(y, b)),
        "base_auc": float(roc_auc_score(y, b)),
        "aug_auc": float(roc_auc_score(y, a)),
    }


def paired_ci(ctx, history, index, params, seasons, n_boot=10000, seed=42):
    feats = build_reason_features(ctx, history, index, params)
    full = pd.concat([ctx, feats], axis=1)
    base_p, aug_p = pd.Series(np.nan, index=full.index), pd.Series(np.nan, index=full.index)
    for s in seasons:
        train, test = full[full["season"] < s], full[full["season"] == s]
        if train.empty or test.empty:
            continue
        base_p[test.index] = CatBoostBaseline(columns=CONTEXT_COLUMNS).fit(train).predict(test)
        aug_p[test.index] = (
            CatBoostBaseline(columns=CONTEXT_COLUMNS + FEATURE_COLUMNS).fit(train).predict(test)
        )
    m = full["season"].isin(seasons) & base_p.notna()
    y = full.loc[m, "played"].to_numpy()
    d = (aug_p[m].to_numpy() - y) ** 2 - (base_p[m].to_numpy() - y) ** 2
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int(len(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injury-db", default=DATED_DB_PATH)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--test", action="store_true", help="score the sealed season once")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--min-sim", type=float, default=0.90)
    ap.add_argument("--power", type=float, default=3.0)
    ap.add_argument("--prior-weight", type=float, default=10.0)
    args = ap.parse_args()

    ctx, history, index = load_all(args.injury_db)

    if args.sweep:
        grid = [
            RetrievalParams(k=k, min_sim=s, power=p, prior_weight=w)
            for k, s, p, w in itertools.product(
                [10, 25, 100],
                [0.80, 0.90, 0.95],
                [1.0, 3.0, 8.0],
                [5.0, 20.0],
            )
        ]
        logger.info(f"sweeping {len(grid)} settings on {TUNING_SEASONS}")
        rows = []
        for i, p in enumerate(grid, 1):
            r = score_setting(ctx, history, index, p, TUNING_SEASONS)
            rows.append(r)
            logger.info(f"[{i}/{len(grid)}] {p.tag()} delta={r['delta']:+.5f} cov={r['coverage']:.2f}")
        res = pd.DataFrame(rows).sort_values("delta")
        res["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        res["split"] = "tuning"
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
        pd.set_option("display.width", 200)
        print(
            res.head(12)[
                [
                    "k",
                    "min_sim",
                    "power",
                    "prior_weight",
                    "coverage",
                    "base_brier",
                    "aug_brier",
                    "delta",
                    "aug_auc",
                ]
            ]
            .round(5)
            .to_string(index=False)
        )
        best = res.iloc[0]
        print(
            f"\nbest on tuning: k={int(best.k)} min_sim={best.min_sim} power={best.power} "
            f"prior_weight={best.prior_weight}  delta={best.delta:+.5f}"
        )

    if args.test:
        p = RetrievalParams(args.k, args.min_sim, args.power, args.prior_weight)
        logger.info(f"scoring SEALED season {SEALED_SEASON} once with {p.tag()}")
        r = score_setting(ctx, history, index, p, [SEALED_SEASON])
        d, lo, hi, n = paired_ci(ctx, history, index, p, [SEALED_SEASON])
        r.update(
            {
                "split": "sealed",
                "ci_low": lo,
                "ci_high": hi,
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        pd.DataFrame([r]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
        verdict = "helps" if hi < 0 else "hurts" if lo > 0 else "no significant effect"
        print(f"\nsealed {SEALED_SEASON}: n={n} coverage={r['coverage']:.3f}")
        print(f"  brier without retrieval: {r['base_brier']:.4f}")
        print(f"  brier with retrieval:    {r['aug_brier']:.4f}")
        print(f"  delta {d:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")
        print(f"  auc {r['base_auc']:.4f} -> {r['aug_auc']:.4f}")


if __name__ == "__main__":
    main()
