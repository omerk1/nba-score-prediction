"""
Builds the searchable historical matchup-vector index described in the design doc:

    game_id | date | matchup_vector (10 values) | actual_home_margin

matchup_vector = [*home_norm, *away_norm] where each side's normalized fingerprint
is built from Layer-N fingerprints (layer=1 for Layer 1 only, layer=2 once Layer 2
injury adjustment is applied — see injury_layer.py).

Normalization: z-score using mean/std computed across the full available fingerprint
history for that layer (all team-games with a valid rolling window). This is a
design-doc-underspecified choice ("normalize before concatenating" — no method given);
z-score was chosen over min-max because cosine similarity is invariant to per-dimension
scale differences primarily through variance, and min-max is more sensitive to outlier
games early in a team's rolling window (small n_games_in_window -> noisier extremes).
"""

import logging

import numpy as np
import pandas as pd

from src.matchups.db import cache_conn, nba_api_conn

logger = logging.getLogger(__name__)

FINGERPRINT_METRICS = [
    "pace_score", "three_pt_reliance", "paint_activity", "defensive_rating", "assist_rate",
]


def _load_games() -> pd.DataFrame:
    with nba_api_conn() as conn:
        games = pd.read_sql_query(
            "SELECT game_id, game_date, team_id_home, team_id_away, pts_home, pts_away "
            "FROM game WHERE pts_home IS NOT NULL AND pts_away IS NOT NULL",
            conn,
        )
    games["actual_home_margin"] = games["pts_home"] - games["pts_away"]
    return games


def _load_fingerprints(layer: int) -> pd.DataFrame:
    conn = cache_conn()
    fp = pd.read_sql_query(
        "SELECT game_id, team_id, game_date, " + ", ".join(FINGERPRINT_METRICS) +
        ", n_games_in_window FROM matchup_fingerprints WHERE layer = ?",
        conn, params=(layer,),
    )
    conn.close()
    return fp


def _zscore_stats(fp: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats = {}
    for m in FINGERPRINT_METRICS:
        mu, sd = fp[m].mean(), fp[m].std()
        stats[m] = (mu, sd if sd > 1e-9 else 1.0)
    return stats


def build_matchup_index(layer: int = 1) -> pd.DataFrame:
    """One row per game with the 10-dim matchup vector (5 home + 5 away, z-scored) and
    actual_home_margin. Only games where BOTH teams have a valid fingerprint for the
    requested layer are included (early-season games with <5 prior games are dropped
    upstream in fingerprint.py)."""
    games = _load_games()
    fp = _load_fingerprints(layer)
    stats = _zscore_stats(fp)

    fp_norm = fp.copy()
    for m in FINGERPRINT_METRICS:
        mu, sd = stats[m]
        fp_norm[m] = (fp_norm[m] - mu) / sd

    home = fp_norm.rename(columns={m: f"home_{m}" for m in FINGERPRINT_METRICS})
    away = fp_norm.rename(columns={m: f"away_{m}" for m in FINGERPRINT_METRICS})

    merged = games.merge(
        home[["game_id", "team_id"] + [f"home_{m}" for m in FINGERPRINT_METRICS]],
        left_on=["game_id", "team_id_home"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])
    merged = merged.merge(
        away[["game_id", "team_id"] + [f"away_{m}" for m in FINGERPRINT_METRICS]],
        left_on=["game_id", "team_id_away"], right_on=["game_id", "team_id"], how="inner",
    ).drop(columns=["team_id"])

    vector_cols = [f"home_{m}" for m in FINGERPRINT_METRICS] + [f"away_{m}" for m in FINGERPRINT_METRICS]
    merged["matchup_vector"] = merged[vector_cols].values.tolist()

    logger.info(
        f"Matchup index (layer={layer}): {len(merged)} games "
        f"(of {len(games)} total games with a final score)"
    )
    return merged[
        ["game_id", "game_date", "team_id_home", "team_id_away", "actual_home_margin",
         "matchup_vector"] + vector_cols
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    idx = build_matchup_index(layer=1)
    print(idx.head(3))
    print(f"Total rows: {len(idx)}")
