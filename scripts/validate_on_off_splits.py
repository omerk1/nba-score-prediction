"""
Validation deliverable for the player on/off splits feature (see
docs/on_off_splits_log.md).

Produces:
  1. outputs/on_off_splits_results.csv -- the most recent 50 real (Regular
     Season) games with the new on/off features alongside actual margins.
  2. Sanity checks printed to stdout: NaN rate on games with sufficient
     history, and a flag on any per-player on_off_plus_minus beyond +/-20
     (the raw per-player cache value -- the team-level SUMMED feature can
     legitimately exceed +/-20 when multiple players are out at once, so the
     +/-20 sanity bound is applied at the player level, not the team-sum
     level).

Usage:
    python scripts/validate_on_off_splits.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.data_loader import NBADataLoader
from src.feature_engineering.feature_builder import FeatureBuilder
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("outputs/on_off_splits_results.csv")
ON_OFF_COLS = [
    "home_team_missing_player_on_off_impact", "home_team_n_out_total", "home_team_n_out_resolved_on_off",
    "away_team_missing_player_on_off_impact", "away_team_n_out_total", "away_team_n_out_resolved_on_off",
    "missing_player_on_off_impact_diff",
]


def main():
    cfg = load_config()
    loader = NBADataLoader(db_path=cfg.data_paths.raw_db)
    try:
        # Load a wide context window (rolling features need history) but only
        # keep the most recent 50 Regular Season games for the deliverable.
        all_games = loader.load_games(
            start_date="2024-01-01",
            end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
            allowed_season_types=["Regular Season"],
        )
    finally:
        loader.close()

    all_games = all_games.sort_values("GAME_DATE").reset_index(drop=True)
    fb = FeatureBuilder(
        rolling_windows=cfg.features.rolling_windows,
        h2h_margin_window=cfg.features.h2h_margin_window,
        h2h_win_rate_window=cfg.features.h2h_win_rate_window,
    )
    feats = fb.create_all_features(all_games)

    recent_50 = feats.sort_values("GAME_DATE").tail(50).copy()
    recent_50["actual_margin"] = recent_50["PTS_home"] - recent_50["PTS_away"]

    out_cols = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "PTS_home", "PTS_away", "actual_margin"] + ON_OFF_COLS
    result = recent_50[out_cols]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Saved {len(result)} rows -> {RESULTS_PATH}")

    # --- Sanity checks ---
    n_missing = result[ON_OFF_COLS].isna().any(axis=1).sum()
    logger.info(f"Sanity check 1 -- NaN rows in the last 50 games: {n_missing} / {len(result)}")

    logger.info("Sanity check 2 -- team-level missing_player_on_off_impact_diff range in last 50 games: "
                f"min={result['missing_player_on_off_impact_diff'].min():.1f}, "
                f"max={result['missing_player_on_off_impact_diff'].max():.1f}, "
                f"mean={result['missing_player_on_off_impact_diff'].mean():.2f}")

    on_off_db = Path(cfg.on_off_splits.db_path)
    if on_off_db.exists():
        with sqlite3.connect(f"file:{on_off_db}?mode=ro", uri=True) as conn:
            splits = pd.read_sql_query(
                "SELECT player_id, player_name, team_id, split_type, as_of_date, "
                "min_on, min_off, on_off_plus_minus FROM player_on_off_splits "
                "WHERE on_off_plus_minus IS NOT NULL",
                conn,
            )
        min_side = splits[["min_on", "min_off"]].min(axis=1)
        trusted = splits[min_side >= cfg.on_off_splits.min_on_off_minutes]
        beyond_20 = trusted[trusted["on_off_plus_minus"].abs() > 20]
        logger.info(
            f"Sanity check 3 -- per-player on_off_plus_minus beyond +/-20 (min_on_off_minutes-gated "
            f"rows only): {len(beyond_20)} / {len(trusted)} ({100 * len(beyond_20) / max(len(trusted), 1):.1f}%)"
        )
        if not beyond_20.empty:
            logger.info(
                "Examples beyond +/-20:\n" +
                beyond_20.sort_values("on_off_plus_minus", key=lambda s: s.abs(), ascending=False)
                .head(10)[["player_name", "team_id", "split_type", "as_of_date", "min_on", "min_off", "on_off_plus_minus"]]
                .to_string(index=False)
            )
    else:
        logger.warning(f"on/off cache not found at {on_off_db} -- skipping per-player sanity check")


if __name__ == "__main__":
    main()
