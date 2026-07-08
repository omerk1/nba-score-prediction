"""
Built for the KNN-score integration test: precomputes `style_matchup_score` (+
`confidence`, `fallback_used`, `n_similar`) for every game_id in nba_api.sqlite's
`game` table, caching the result in a new `style_matchup_scores` table in
outputs/a7_matchups_cache.sqlite (our own additive cache DB -- see db.py's
docstring; `game`/`nba_api.sqlite` itself is opened strictly read-only and
never written to).

This is the "precompute + join" half of that integration test: FeatureBuilder's
new `_add_style_matchup_features` (src/feature_engineering/feature_builder.py)
left-joins this table onto the training dataframe by GAME_ID rather than
running the KNN similarity search live -- keeps that diff small and reuses
this already-validated pipeline as-is.

Config: reuses the validated hyperparameters now written into configs/config.yaml's
style_matchup block (fingerprint_window=37, decay_halflife=13.2, encoding=
hand_picked, similarity_method=knn, knn_k=81, min_confidence_sample=21,
full_confidence_sample=82, layer=2/injury-adjusted,
injury_calibration_decay_halflife_games=20) via
src/matchups/config.py:load_style_matchup_config() -- see docs/a7_phase_log.md's
KNN-Score Integration Test section for why that yaml block previously still held
the untuned defaults and was corrected as part of that run. Nothing is re-tuned
here.

Z-score leakage: similarity.run_similarity_search is called with
zscore_cutoff_date=<main config's datasets_loading.train_end_date> so the
matchup-vector normalization stats are fit only on fingerprint rows strictly
before the model's own train/validation boundary -- mirrors the
zscore_point_in_time mechanism validated in the phase log's Post-Fix
Hyperparameter Recheck section (there applied per-fold/per-static-split for a
hyperparameter search; here applied once for a single precompute pass covering
train+val+test). The KNN/cosine search itself is unaffected -- already fully
leakage-safe per-game via np.searchsorted on game_date (see similarity.py's
docstring); this only
concerns the vector normalization constants, not which historical games are
visible to a given target game.

Scores the FULL game_id range (~12,000+ games), not just a recent evaluation
window, since FeatureBuilder needs a value for every row in the training
dataframe (train from 2018 onward, plus val/test). Games with fewer than
`min_games_played` prior team-games (early-season, no valid fingerprint) are
not scored -- FeatureBuilder's left join leaves those as NaN, same "kept,
CatBoost handles natively" convention already used for every other feature.
"""

import logging

import pandas as pd

from src.matchups.config import load_style_matchup_config
from src.matchups.db import cache_conn, table_exists
from src.matchups.fingerprint import build_fingerprint_cache
from src.matchups.injury_layer import build_injury_adjusted_fingerprints
from src.matchups.players import build_archetype_cache, build_name_resolution_cache
from src.matchups.similarity import run_similarity_search
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS style_matchup_scores (
        game_id TEXT PRIMARY KEY,
        style_matchup_score REAL,
        confidence REAL,
        fallback_used INTEGER,
        n_similar INTEGER
    )
"""


def precompute_and_cache() -> dict:
    main_cfg = load_config()
    sm_cfg = load_style_matchup_config()

    # Prerequisites for Layer 2 (injury adjustment): player_name_resolution and
    # player_archetypes are read by injury_layer.py's _out_players_with_reason but
    # NOT built by it -- a fresh outputs/a7_matchups_cache.sqlite (e.g. a new
    # worktree) starts with both tables empty, which silently makes Layer 2 a
    # no-op (0 games adjusted) rather than erroring. Build them here if missing so
    # this script is self-contained; skip if already populated (they're
    # independent of fingerprint_window/decay_halflife, so no need to rebuild on
    # every run once cached).
    conn = cache_conn()
    has_names = table_exists(conn, "player_name_resolution") and conn.execute(
        "SELECT COUNT(*) FROM player_name_resolution"
    ).fetchone()[0] > 0
    has_archetypes = table_exists(conn, "player_archetypes") and conn.execute(
        "SELECT COUNT(*) FROM player_archetypes"
    ).fetchone()[0] > 0
    conn.close()

    if not has_names:
        logger.info("player_name_resolution cache empty — building...")
        logger.info(f"Name resolution: {build_name_resolution_cache()}")
    if not has_archetypes:
        logger.info("player_archetypes cache empty — building...")
        logger.info(f"Archetype classification: {build_archetype_cache()}")

    logger.info(
        "Building layer=1 fingerprint cache (window=%s, halflife=%s)...",
        sm_cfg["fingerprint_window"], sm_cfg["decay_halflife"],
    )
    build_fingerprint_cache(layer=1, min_games=main_cfg.features.min_games_played)

    logger.info("Building layer=2 injury-adjusted fingerprints...")
    build_injury_adjusted_fingerprints()

    zscore_cutoff = main_cfg.datasets_loading.train_end_date
    logger.info(
        "Running similarity search (method=%s, k=%s, min_conf=%s, full_conf=%s) "
        "over full history, zscore_cutoff_date=%s...",
        sm_cfg["similarity_method"], sm_cfg["knn_k"],
        sm_cfg["min_confidence_sample"], sm_cfg["full_confidence_sample"], zscore_cutoff,
    )
    out = run_similarity_search(
        layer=2,
        method=sm_cfg["similarity_method"],
        threshold=sm_cfg["similarity_threshold"],
        k=sm_cfg["knn_k"],
        min_confidence_sample=sm_cfg["min_confidence_sample"],
        full_confidence_sample=sm_cfg["full_confidence_sample"],
        zscore_cutoff_date=zscore_cutoff,
    )

    conn = cache_conn()
    conn.execute("DROP TABLE IF EXISTS style_matchup_scores")
    conn.execute(TABLE_DDL)
    rows = [
        (
            r.game_id,
            float(r.style_score) if r.style_score is not None and pd.notna(r.style_score) else None,
            float(r.confidence),
            int(r.fallback_used),
            int(r.n_similar),
        )
        for r in out.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO style_matchup_scores
           (game_id, style_matchup_score, confidence, fallback_used, n_similar)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()

    summary = {
        "n_games_scored": len(out),
        "fallback_rate": float(out["fallback_used"].mean()) if len(out) else None,
        "mean_confidence": float(out["confidence"].mean()) if len(out) else None,
        "score_min": float(out["style_score"].min()) if len(out) else None,
        "score_max": float(out["style_score"].max()) if len(out) else None,
        "nan_scores": int(out["style_score"].isna().sum()) if len(out) else 0,
    }
    logger.info(f"Precompute done: {summary}")
    return summary


if __name__ == "__main__":
    print(precompute_and_cache())
