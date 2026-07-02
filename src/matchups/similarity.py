"""
Layer 3: Historical Matchup Similarity.

Runs on top of Layer 2 (injury-adjusted) matchup vectors, per the design doc's
corrected layer order. Two similarity methods are implemented and compared
(pre-settled decision: try both, keep the better one — see Phase 3 in the phase
log for the actual comparison and winner):

  cosine  — keep all historical games above `similarity_threshold` (default 0.70)
  knn     — keep the K most similar historical games regardless of score (default 30)

CRITICAL (data leakage): for a target game at position i (rows sorted ascending by
game_date), the search space is restricted to rows with game_date STRICTLY BEFORE
the target's game_date (via np.searchsorted on the sorted date array — this
correctly excludes every other game on the same date, not just earlier row
positions, since multiple games share a date and none of them may see each other).

Confidence: min(n_similar / full_confidence_sample, 1.0). Below
min_confidence_sample, falls back to the A2 H2H score (low_confidence_fallback:
h2h, pre-settled) instead of returning 0.
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.baseline_a2 import build_a2_h2h_scores
from src.matchups.config import load_style_matchup_config
from src.matchups.matchup_index import build_matchup_index

logger = logging.getLogger(__name__)


def _prep_sorted_index(layer: int) -> pd.DataFrame:
    idx = build_matchup_index(layer=layer)
    idx = idx.sort_values("game_date").reset_index(drop=True)
    return idx


def run_similarity_search(
    layer: int = 2,
    method: str = "cosine",
    threshold: float = 0.70,
    k: int = 30,
    min_confidence_sample: int = 10,
    full_confidence_sample: int = 50,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
) -> pd.DataFrame:
    """Returns one row per evaluated game: style_score, confidence, n_similar,
    fallback_used, plus h2h_score and actual_home_margin for comparison."""
    idx = _prep_sorted_index(layer=layer)
    vector_cols = [c for c in idx.columns if c.startswith("home_") or c.startswith("away_")]
    V = idx[vector_cols].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(V, axis=1)
    norms[norms == 0] = 1e-9
    dates = idx["game_date"].to_numpy()

    h2h = build_a2_h2h_scores()[["game_id", "h2h_score"]]
    idx = idx.merge(h2h, on="game_id", how="left")
    idx["h2h_score"] = idx["h2h_score"].fillna(0.0)

    if eval_start_date:
        eval_mask = idx["game_date"] >= eval_start_date
    else:
        eval_mask = pd.Series(True, index=idx.index)
    if eval_end_date:
        eval_mask &= idx["game_date"] <= eval_end_date
    eval_positions = np.where(eval_mask.to_numpy())[0]

    results = []
    for i in eval_positions:
        target_date = dates[i]
        end_pos = int(np.searchsorted(dates, target_date, side="left"))  # exclusive of same-date games
        if end_pos == 0:
            n_similar = 0
            style_score = None
        else:
            hist_V = V[:end_pos]
            hist_norms = norms[:end_pos]
            sims = (hist_V @ V[i]) / (hist_norms * norms[i])

            if method == "cosine":
                keep = np.where(sims >= threshold)[0]
            elif method == "knn":
                n_take = min(k, len(sims))
                keep = np.argpartition(-sims, n_take - 1)[:n_take] if n_take > 0 else np.array([], dtype=int)
            else:
                raise ValueError(f"Unknown similarity_method: {method}")

            n_similar = len(keep)
            style_score = float(idx["actual_home_margin"].to_numpy()[:end_pos][keep].mean()) if n_similar else None

        confidence = min(n_similar / full_confidence_sample, 1.0) if n_similar else 0.0
        fallback_used = n_similar < min_confidence_sample
        final_score = idx.at[i, "h2h_score"] if fallback_used else style_score
        if final_score is None:
            final_score = idx.at[i, "h2h_score"]

        results.append({
            "game_id": idx.at[i, "game_id"],
            "game_date": idx.at[i, "game_date"],
            "actual_home_margin": idx.at[i, "actual_home_margin"],
            "h2h_score": idx.at[i, "h2h_score"],
            "raw_style_score": style_score,
            "n_similar": n_similar,
            "confidence": confidence,
            "fallback_used": fallback_used,
            "style_score": final_score,
        })

    out = pd.DataFrame(results)
    logger.info(
        f"Similarity search (layer={layer}, method={method}): {len(out)} games evaluated, "
        f"fallback_rate={out['fallback_used'].mean():.3f}, mean_confidence={out['confidence'].mean():.3f}"
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_style_matchup_config()
    for method, kw in [("cosine", {"threshold": cfg["similarity_threshold"]}), ("knn", {"k": cfg["knn_k"]})]:
        out = run_similarity_search(
            layer=2, method=method,
            min_confidence_sample=cfg["min_confidence_sample"],
            full_confidence_sample=cfg["full_confidence_sample"],
            eval_start_date="2023-10-01",
            **kw,
        )
        corr = out["style_score"].corr(out["actual_home_margin"])
        print(method, "n=", len(out), "fallback_rate=", out["fallback_used"].mean(),
              "mean_conf=", out["confidence"].mean(), "corr=", corr)
