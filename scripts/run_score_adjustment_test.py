"""
Does a language model add anything on top of the trained model's own point
prediction?

Post-hoc and read-only, in the same spirit as src/evaluation/conformal.py: it
calls run_split for one fold, takes the held-out predictions that come back,
and never changes training, features, folds, or the metric.

Per game it sends the model's predicted differential plus a compact block of
pre-game facts, and asks for a bounded adjustment in [-max, +max] points. It
then compares differential MAE with and without the adjustment on the same
games, with a paired bootstrap over per-game absolute-error differences.

Memorization: fold5's test window (2025-10-21 onward) falls entirely after the
default model's training cutoff, so team names and dates are included. That
gives the language model its best shot (real basketball knowledge) without
opening a recall path to these specific results. Use --anonymize to strip them.

Usage:
  venv/bin/python3 scripts/run_score_adjustment_test.py --fold fold5 --max-adjust 4
"""

import argparse
import hashlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as nba_teams

from src.availability.db import get_conn
from src.availability.llm_estimator import GeminiClient
from src.evaluation.cv_harness import run_split, validate_fold_definitions
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TEAM_ABBR = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}
OUT_CSV = Path("outputs/score_adjustment_test.csv")

SYSTEM = (
    "You are adjusting a gradient-boosted NBA model's predicted point differential "
    "(home minus away). The model is well calibrated overall and beats a rolling "
    "baseline, so the prior on any adjustment is ZERO. Suggest a non-zero adjustment "
    "only when the listed facts point to something the model is likely to be "
    "underweighting. Most games should get 0.0. "
    'Return JSON with exactly: "adjustment" (float, points to ADD to the model\'s '
    'differential, bounded as stated), "confidence" (float 0-1), "rationale" (one sentence).'
)

# Pre-game facts shown to the model. Every one is a feature the trained model
# already sees; the question is whether the LLM weighs them differently.
FACT_SPECS = [
    ("elo_diff", "Elo differential (home minus away)", "{:+.0f}"),
    ("strength_differential_L20", "strength differential over last 20", "{:+.2f}"),
    ("home_team_off_eff_L20", "home offensive efficiency L20", "{:.1f}"),
    ("home_team_def_eff_L20", "home defensive efficiency L20", "{:.1f}"),
    ("away_team_off_eff_L20", "away offensive efficiency L20", "{:.1f}"),
    ("away_team_def_eff_L20", "away defensive efficiency L20", "{:.1f}"),
    ("home_team_rest_days", "home rest days", "{:.0f}"),
    ("away_team_rest_days", "away rest days", "{:.0f}"),
    ("home_team_back_to_back", "home on a back-to-back", "{:.0f}"),
    ("away_team_back_to_back", "away on a back-to-back", "{:.0f}"),
    ("home_team_n_out", "home players ruled out", "{:.0f}"),
    ("away_team_n_out", "away players ruled out", "{:.0f}"),
    ("team_deficit_diff", "injury deficit differential", "{:+.3f}"),
    ("h2h_home_win_pct", "home win pct in recent head-to-head", "{:.2f}"),
    ("season_progress", "fraction of season elapsed", "{:.2f}"),
]


def build_prompt(row: pd.Series, max_adjust: float, anonymize: bool) -> str:
    lines = []
    if not anonymize:
        lines.append(
            f"{_TEAM_ABBR.get(int(row['AWAY_TEAM_ID']), '?')} at "
            f"{_TEAM_ABBR.get(int(row['HOME_TEAM_ID']), '?')}, {str(row['GAME_DATE'])[:10]}."
        )
    lines.append(
        f"Model prediction: home {row['model_home_pred']:.1f}, away {row['model_away_pred']:.1f} "
        f"(differential {row['model_diff_pred']:+.1f}, total {row['model_total_pred']:.1f})."
    )
    lines.append("Pre-game facts:")
    for col, label, fmt in FACT_SPECS:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        lines.append(f"- {label}: {fmt.format(v)}")
    lines.append(
        f"\nGive an adjustment between {-max_adjust:+.1f} and {max_adjust:+.1f} points, "
        "or 0.0 if the model's number already looks right."
    )
    return "\n".join(lines)


def _parse(text: str, max_adjust: float) -> dict:
    raw = json.loads(text)
    adj = float(raw["adjustment"])
    return {
        "adjustment": float(np.clip(adj, -max_adjust, max_adjust)),
        "confidence": float(raw.get("confidence", np.nan)),
        "rationale": str(raw.get("rationale", ""))[:300],
    }


def call_all(
    prompts: list[str], model: str, db_path: str, workers: int, rpm: int, max_adjust: float, tag: str
) -> list[dict | None]:
    client = GeminiClient(model, rpm)
    conn = get_conn(db_path)
    keys = [hashlib.sha256(f"{model}\n{tag}\n{p}".encode()).hexdigest() for p in prompts]
    cached: dict[str, dict] = {}
    for i in range(0, len(keys), 500):
        chunk = keys[i : i + 500]
        q = f"SELECT prompt_hash, response_json FROM llm_cache WHERE prompt_hash IN ({','.join('?' * len(chunk))})"
        for h, js in conn.execute(q, chunk):
            cached[h] = json.loads(js)
    todo = [(k, p) for k, p in dict(zip(keys, prompts)).items() if k not in cached]
    logger.info(f"{len(prompts)} games, {len(cached)} cached, {len(todo)} to call")

    lock, done = threading.Lock(), 0

    def work(item):
        nonlocal done
        k, p = item
        parsed = None
        for attempt in range(4):
            try:
                raw = client._client.models.generate_content(
                    model=model,
                    contents=p,
                    config=client._types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        response_mime_type="application/json",
                        temperature=0.0,
                        thinking_config=client._types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                client._limiter.wait()
                parsed = _parse(raw.text, max_adjust)
                break
            except Exception as e:
                if "PerDay" in str(e):
                    raise RuntimeError(f"daily quota exhausted: {e}") from e
                if attempt < 3:
                    time.sleep(5 * (2**attempt))
                else:
                    logger.warning(f"call failed: {e}")
        with lock:
            if parsed is not None:
                cached[k] = parsed
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?)",
                    (k, model, tag, p, json.dumps(parsed), datetime.now(timezone.utc).isoformat()),
                )
            done += 1
            if done % 200 == 0:
                conn.commit()
                logger.info(f"{done}/{len(todo)} calls done")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
        conn.commit()
    conn.close()
    return [cached.get(k) for k in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold5")
    ap.add_argument("--max-adjust", type=float, default=4.0)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    tag = args.tag or f"score_adjust_{args.fold}_max{args.max_adjust:g}{'_anon' if args.anonymize else ''}"

    cfg = load_config()
    validate_fold_definitions(cfg.cv.folds)
    fold = next(f for f in cfg.cv.folds if f.name == args.fold)
    logger.info(f"{fold.name}: test window {fold.test_start_date} to {fold.test_end_date}")

    result = run_split(
        cfg,
        fold.train_end_date,
        fold.validation_start_date,
        fold.validation_end_date,
        fold.test_start_date,
        fold.test_end_date,
        keep_artifacts=True,
    )
    tf = result.test_features.copy()
    preds = result.predictor.predict(tf[result.feature_cols])
    tf["model_home_pred"], tf["model_away_pred"] = preds[:, 0], preds[:, 1]
    tf["model_diff_pred"] = tf["model_home_pred"] - tf["model_away_pred"]
    tf["model_total_pred"] = tf["model_home_pred"] + tf["model_away_pred"]
    tf["actual_diff"] = tf["PTS_home"] - tf["PTS_away"]

    in_window = pd.to_datetime(tf["GAME_DATE"]).between(
        pd.Timestamp(fold.test_start_date), pd.Timestamp(fold.test_end_date)
    )
    if not in_window.all():
        raise AssertionError("predicted games outside the fold's test window; aborting")
    logger.info(f"{len(tf)} held-out games")

    prompts = [build_prompt(r, args.max_adjust, args.anonymize) for _, r in tf.iterrows()]
    logger.info("example prompt:\n" + prompts[0])
    out = call_all(
        prompts,
        cfg.availability_agent.llm_model,
        cfg.availability_agent.db_path,
        cfg.availability_agent.parallel_workers,
        cfg.availability_agent.api_calls_per_minute,
        args.max_adjust,
        tag,
    )

    tf["adjustment"] = [o["adjustment"] if o else np.nan for o in out]
    tf["confidence"] = [o["confidence"] if o else np.nan for o in out]
    tf["rationale"] = [o["rationale"] if o else "" for o in out]
    n_failed = int(tf["adjustment"].isna().sum())
    tf["adjustment"] = tf["adjustment"].fillna(0.0)
    tf["adjusted_diff_pred"] = tf["model_diff_pred"] + tf["adjustment"]

    base_err = (tf["model_diff_pred"] - tf["actual_diff"]).abs()
    adj_err = (tf["adjusted_diff_pred"] - tf["actual_diff"]).abs()
    d = (adj_err - base_err).to_numpy()  # negative => adjustment helps
    rng = np.random.default_rng(42)
    boot = d[rng.integers(0, len(d), size=(args.n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    nz = tf["adjustment"] != 0
    summary = {
        "tag": tag,
        "fold": fold.name,
        "n_games": len(tf),
        "n_failed": n_failed,
        "base_diff_mae": float(base_err.mean()),
        "adjusted_diff_mae": float(adj_err.mean()),
        "mae_delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "pct_nonzero_adjustment": float(nz.mean()),
        "mean_abs_adjustment": float(tf["adjustment"].abs().mean()),
        "adjustment_corr_with_residual": float(
            np.corrcoef(tf["adjustment"], tf["actual_diff"] - tf["model_diff_pred"])[0, 1]
        ),
        "base_win_acc": float(((tf["model_diff_pred"] > 0) == (tf["actual_diff"] > 0)).mean()),
        "adjusted_win_acc": float(((tf["adjusted_diff_pred"] > 0) == (tf["actual_diff"] > 0)).mean()),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    verdict = "adjustment helps" if hi < 0 else "adjustment hurts" if lo > 0 else "no significant effect"
    summary["verdict"] = verdict

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    detail = Path(f"outputs/score_adjustment_games_{tag}.csv")
    tf[
        [
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "model_diff_pred",
            "adjustment",
            "confidence",
            "adjusted_diff_pred",
            "actual_diff",
            "rationale",
        ]
    ].to_csv(detail, index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nper-game detail: {detail}")
    if nz.any():
        print("\nadjustment size vs. whether it helped:")
        b = pd.cut(tf.loc[nz, "adjustment"].abs(), [0, 1, 2, 3, 4], include_lowest=True)
        print(
            tf.loc[nz]
            .groupby(b, observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "n": len(g),
                        "helped_pct": float(
                            (
                                (g["adjusted_diff_pred"] - g["actual_diff"]).abs()
                                < (g["model_diff_pred"] - g["actual_diff"]).abs()
                            ).mean()
                        ),
                    }
                ),
                include_groups=False,
            )
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
