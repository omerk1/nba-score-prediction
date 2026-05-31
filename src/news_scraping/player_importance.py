"""
Player importance scores for injury impact estimation.

importance = (minutes_share × w1) + (usage_rate_norm × w2) + (pts_share × w3)

# TODO (ablation candidate): replace with on/off point differential from nba_api
# (leaguedashlineups). Current formula measures volume of activity, not net marginal
# value — a 30 PPG player may only add ~8 net pts if teammates fill in.
# After implementing: retrain and compare val_diff_mae in outputs/experiments.csv.

All inputs are normalized per-team to 0–1 so players on different teams are comparable.
Raw per-game numbers don't capture relative value — a 15 PPG player on a 90-pt team
matters more than the same on a 120-pt team.

Computed weekly (as_of_date = snapshot date) so historical predictions only use stats
available before each game. At join time, use latest as_of_date < game_date.

The LLM extractor uses importance > 0.5 to flag star_out — a separate binary feature
because losing a star has disproportionate impact beyond the linear score.
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from src.news_scraping.db import get_conn, init_db
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_SEASON_START_MONTH = 10  # NBA regular season starts in October


def _season_start(season_year: int) -> date:
    return date(season_year, _SEASON_START_MONTH, 1)


def _fetch_stats(season: str, date_from: date, date_to: date) -> pd.DataFrame:
    """Fetch cumulative per-game base + advanced stats between two dates."""
    kwargs = dict(
        season=season,
        date_from_nullable=date_from.strftime("%m/%d/%Y"),
        date_to_nullable=date_to.strftime("%m/%d/%Y"),
        per_mode_detailed="PerGame",
    )
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        **kwargs, measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    time.sleep(1)  # nba_api rate limit

    advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        **kwargs, measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    time.sleep(1)

    merged = base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "MIN", "PTS"]].merge(
        advanced[["PLAYER_ID", "TEAM_ID", "USG_PCT"]], on=["PLAYER_ID", "TEAM_ID"]
    )
    merged.columns = ["player_id", "player_name", "team_id", "minutes_per_game", "pts_per_game", "usage_rate"]
    return merged


def _compute_importance(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Normalize per team, then compute weighted importance score (0–1)."""
    for col, share_col in [("minutes_per_game", "minutes_share"), ("pts_per_game", "pts_share")]:
        team_total = df.groupby("team_id")[col].transform("sum")
        df[share_col] = (df[col] / team_total.replace(0, 1)).clip(0, 1)

    # usage_rate is already team-relative; normalize to team max for a consistent 0–1 scale
    team_max_usg = df.groupby("team_id")["usage_rate"].transform("max")
    df["usage_rate_norm"] = (df["usage_rate"] / team_max_usg.replace(0, 1)).clip(0, 1)

    w = weights
    df["importance_score"] = (
        df["minutes_share"] * w["minutes_share"]
        + df["usage_rate_norm"] * w["usage_rate"]
        + df["pts_share"] * w["pts_share"]
    ).clip(0, 1)
    return df


def compute_and_store(season: str, as_of_date: date) -> int:
    """Compute importance for all players up to as_of_date and upsert to DB. Returns row count."""
    cfg = load_config()
    season_year = int(season[:4])
    date_from = _season_start(season_year)

    if as_of_date < date_from:
        return 0

    try:
        df = _fetch_stats(season, date_from, as_of_date)
    except Exception as e:
        logger.warning(f"Skipping {season} as of {as_of_date} — API error: {e}")
        return 0

    if df.empty:
        logger.debug(f"Skipping {season} as of {as_of_date} — no games yet")
        return 0

    df = _compute_importance(df, cfg.injury_features.importance_weights.model_dump())
    df["as_of_date"] = str(as_of_date)

    df["updated_at"] = datetime.now(timezone.utc).isoformat()
    rows = df[
        ["player_id", "player_name", "team_id", "as_of_date",
         "importance_score", "minutes_per_game", "pts_per_game", "usage_rate", "updated_at"]
    ].to_dict("records")

    init_db(cfg.injury_features.db_path)
    with get_conn(cfg.injury_features.db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO player_importance
               (player_id, player_name, team_id, as_of_date, importance_score,
                minutes_per_game, pts_per_game, usage_rate, updated_at)
               VALUES (:player_id, :player_name, :team_id, :as_of_date, :importance_score,
                       :minutes_per_game, :pts_per_game, :usage_rate, :updated_at)""",
            rows,
        )

    logger.info(f"Upserted {len(rows)} importance rows for {season} as of {as_of_date}")
    return len(rows)


def backfill_season(season: str, interval_days: int = 7) -> None:
    """Compute weekly importance snapshots across a full season. Skips dates already in DB."""
    cfg = load_config()
    init_db(cfg.injury_features.db_path)

    season_year = int(season[:4])
    cursor = _season_start(season_year) + timedelta(days=interval_days)
    today = date.today()

    while cursor <= today:
        with get_conn(cfg.injury_features.db_path) as conn:
            already_stored = conn.execute(
                "SELECT 1 FROM player_importance WHERE as_of_date = ? LIMIT 1",
                (str(cursor),),
            ).fetchone()

        if already_stored:
            logger.debug(f"Skipping {season} as of {cursor} — already in DB")
        else:
            logger.info(f"Computing importance for {season} as of {cursor}")
            compute_and_store(season, cursor)

        cursor += timedelta(days=interval_days)