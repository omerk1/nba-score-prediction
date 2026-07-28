"""
Validation deliverable for the season motivation / seeding-incentive features
(see docs/SEASON_MOTIVATION_LOG.md).

Produces:
  1. outputs/season_motivation_results.csv -- the most recent 50 real (Regular
     Season) games with the new features alongside actual margins.
  2. Sanity checks printed to stdout: no NaN, standings_pressure in [0,1],
     clinch values non-negative.

Usage:
    python scripts/validate_season_motivation.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.data_loader import NBADataLoader
from src.feature_engineering.feature_builder import FeatureBuilder
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("outputs/season_motivation_results.csv")
MOTIVATION_COLS = [
    "home_team_standings_pressure", "home_team_roster_behavior_score",
    "home_team_games_to_clinch_ceiling", "home_team_games_to_clinch_floor",
    "away_team_standings_pressure", "away_team_roster_behavior_score",
    "away_team_games_to_clinch_ceiling", "away_team_games_to_clinch_floor",
    "standings_pressure_diff", "roster_behavior_score_diff",
    "games_to_clinch_ceiling_diff", "games_to_clinch_floor_diff",
]


def main():
    cfg = load_config()
    loader = NBADataLoader(db_path=cfg.data_paths.raw_db)
    try:
        # Load a wide context window (rolling/standings features need history)
        # but only keep the most recent 50 Regular Season games for the deliverable.
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

    out_cols = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "PTS_home", "PTS_away", "actual_margin"] + MOTIVATION_COLS
    result = recent_50[out_cols]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Saved {len(result)} rows -> {RESULTS_PATH}")

    # --- Sanity checks (over the full featurized history, not just the last 50) ---
    n_missing = feats[MOTIVATION_COLS].isna().any(axis=1).sum()
    logger.info(f"Sanity check 1 -- NaN rows (full history, {len(feats)} games): {n_missing}")

    score_cols = ["home_team_standings_pressure", "away_team_standings_pressure"]
    out_of_range = feats[
        (feats[score_cols] < 0).any(axis=1) | (feats[score_cols] > 1).any(axis=1)
    ]
    logger.info(f"Sanity check 2 -- standings_pressure outside [0,1]: {len(out_of_range)} / {len(feats)}")

    clinch_cols = [
        "home_team_games_to_clinch_ceiling", "home_team_games_to_clinch_floor",
        "away_team_games_to_clinch_ceiling", "away_team_games_to_clinch_floor",
    ]
    negative_clinch = feats[(feats[clinch_cols] < 0).any(axis=1)]
    logger.info(f"Sanity check 3 -- negative clinch values: {len(negative_clinch)} / {len(feats)}")

    logger.info(
        "Sanity check 4 -- home_team_standings_pressure distribution: "
        f"min={feats['home_team_standings_pressure'].min():.3f}, "
        f"mean={feats['home_team_standings_pressure'].mean():.3f}, "
        f"max={feats['home_team_standings_pressure'].max():.3f}"
    )
    locked_both = (
        (feats["home_team_games_to_clinch_ceiling"] == 0) & (feats["home_team_games_to_clinch_floor"] == 0)
    )
    logger.info(
        f"Sanity check 5 -- rows where home team has nothing left to play for "
        f"(ceiling==0 and floor==0): {locked_both.sum()} / {len(feats)} ({100 * locked_both.mean():.1f}%)"
    )


if __name__ == "__main__":
    main()
