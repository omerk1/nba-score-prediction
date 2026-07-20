"""
Backfill player_on_off_splits with historical on/off-court splits from nba_api.

Data source: nba_api's TeamPlayerOnOffSummary endpoint (see
docs/on_off_splits_decisions.md for the empirical investigation this is based on).
Key facts that shape this script, all confirmed via real API calls, not assumed:

- `LastNGames` does NOT compose with `DateFrom`/`DateTo` (returns an empty
  dataframe when both are set) and silently ignores `DateFrom` when passed alone
  -- so it is NEVER used here. `DateTo` alone gives a genuine, leakage-safe
  season-cumulative-to-date snapshot.
- `DateTo` is INCLUSIVE of games played on that exact calendar date (confirmed:
  Boston's cumulative GP jumped from 39 to 40 exactly on 2024-01-15, the date of
  an actual BOS game -- not the day after). So every checkpoint/fetch in this
  script that is meant to inform a prediction for calendar date D must pass
  `date_to_nullable = D - 1 day`, never `D` itself.
- `OpponentTeamID` and `Location` (Home/Road) both compose correctly with
  `DateTo` (confirmed empirically) -- used for the 'vs_opponent' and
  'home'/'away' split types respectively.

Checkpoint cadence (not per-game): the endpoint is called ONCE PER TEAM and
returns a season-cumulative number, so a fresh snapshot is only needed every
`checkpoint_cadence_days` (config: on_off_splits.checkpoint_cadence_days), not
once per game -- the value barely moves between two adjacent games. See the
decisions doc's cost analysis: per-game precision would cost ~10x more API calls
for a number that changes by a fraction of a percent between checkpoints.

Split types backfilled at full checkpoint cadence: 'overall', 'home', 'away' (3
calls per team per checkpoint). 'vs_opponent' is deliberately NOT backfilled at
the same cadence x all 29 possible opponents (combinatorially infeasible, ~200k+
calls for full history -- see decisions doc's cost analysis) -- instead it's
fetched lazily, only for opponent pairings that actually occurred in real games,
via `--vs-opponent --recent-games N`.

Resumable/incremental: before each API call, checks whether a row already exists
for that (team_id, split_type, opponent_team_id, as_of_date) and skips it unless
--force is passed -- same idempotent-by-primary-key pattern
scripts/backfill_player_stats.py already uses (INSERT OR REPLACE).

Usage:
    # Checkpoint backfill (overall/home/away) for a set of seasons:
    python scripts/backfill_on_off_splits.py --seasons 2023-24,2024-25,2025-26

    # Lazy vs-opponent backfill for the most recent N real games:
    python scripts/backfill_on_off_splits.py --vs-opponent --recent-games 50

    # Resume an interrupted run (skips already-cached rows automatically; --force
    # to refetch anyway):
    python scripts/backfill_on_off_splits.py --seasons 2023-24,2024-25,2025-26 --update
"""

import argparse
import datetime
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import TeamPlayerOnOffSummary

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.fetch_data import _date_to_season
from src.migrations.migration_create_player_on_off_splits import migrate_player_on_off_splits
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

SLEEP_SECONDS = 0.7  # matches box_scores.py / fetch_data.py's rate-limit convention
BACKFILL_LOG_PATH = Path("outputs/on_off_backfill_log.csv")
BACKFILL_LOG_COLUMNS = [
    "timestamp", "team_id", "split_type", "opponent_team_id", "as_of_date",
    "season", "status", "n_players", "elapsed_seconds",
]


def _log_row(row: dict) -> None:
    BACKFILL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BACKFILL_LOG_PATH.exists()
    pd.DataFrame([row])[BACKFILL_LOG_COLUMNS].to_csv(
        BACKFILL_LOG_PATH, mode="a", header=write_header, index=False
    )


def _existing_keys(db_path: str) -> set[tuple]:
    """(team_id, split_type, opponent_team_id, as_of_date) already cached."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT DISTINCT team_id, split_type, opponent_team_id, as_of_date "
            "FROM player_on_off_splits"
        )
        return {tuple(r) for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _all_team_ids() -> list[int]:
    with sqlite3.connect("file:data/raw/nba_api.sqlite?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT DISTINCT team_id_home FROM game").fetchall()
    return sorted({r[0] for r in rows})


def _season_checkpoints(season: str, cadence_days: int) -> list[str]:
    """Weekly(ish)-cadence calendar-date checkpoints spanning the Regular Season
    games actually present in `game` for this season string (e.g. '2023-24')."""
    with sqlite3.connect("file:data/raw/nba_api.sqlite?mode=ro", uri=True) as conn:
        dates = pd.read_sql_query(
            "SELECT DISTINCT game_date FROM game WHERE season_type = 'Regular Season'",
            conn,
        )["game_date"]
    in_season = dates[dates.map(lambda d: _date_to_season(d) == season)]
    if in_season.empty:
        return []
    start, end = in_season.min(), in_season.max()
    checkpoints = pd.date_range(start, end, freq=f"{cadence_days}D")
    return checkpoints.strftime("%Y-%m-%d").tolist()


def _parse_on_off_summary(
    resp: TeamPlayerOnOffSummary,
) -> pd.DataFrame:
    """Merge the On/Off dataframes into one row per player with on_off_plus_minus."""
    dfs = resp.get_data_frames()
    off_court, on_court = dfs[1], dfs[2]  # load_response() order: overall, off, on
    if off_court.empty or on_court.empty:
        return pd.DataFrame()

    keep = ["VS_PLAYER_ID", "VS_PLAYER_NAME", "GP", "MIN", "PLUS_MINUS", "NET_RATING"]
    off_r = off_court[keep].rename(columns={
        "GP": "gp_off", "MIN": "min_off", "PLUS_MINUS": "plus_minus_off", "NET_RATING": "net_rating_off",
    })
    on_r = on_court[keep].rename(columns={
        "GP": "gp_on", "MIN": "min_on", "PLUS_MINUS": "plus_minus_on", "NET_RATING": "net_rating_on",
    })
    merged = on_r.merge(off_r, on=["VS_PLAYER_ID", "VS_PLAYER_NAME"], how="outer")
    merged["on_off_plus_minus"] = merged["plus_minus_on"] - merged["plus_minus_off"]
    merged["on_off_net_rating"] = merged["net_rating_on"] - merged["net_rating_off"]
    return merged.rename(columns={"VS_PLAYER_ID": "player_id", "VS_PLAYER_NAME": "player_name"})


def _fetch_split(
    team_id: int,
    season: str,
    as_of_date: str,
    split_type: str,
    opponent_team_id: Optional[int] = None,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, float]:
    """One API call for one (team, split_type, as_of_date[, opponent]). Returns
    (parsed dataframe, elapsed_seconds). Empty dataframe on failure/no data."""
    kwargs = dict(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season",
        date_to_nullable=as_of_date,
    )
    if split_type == "home":
        kwargs["location_nullable"] = "Home"
    elif split_type == "away":
        kwargs["location_nullable"] = "Road"
    elif split_type == "vs_opponent":
        kwargs["opponent_team_id"] = opponent_team_id

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = TeamPlayerOnOffSummary(**kwargs)
            df = _parse_on_off_summary(resp)
            return df, time.time() - t0
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(
                f"Fetch failed team={team_id} split={split_type} as_of={as_of_date} "
                f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s..."
            )
            time.sleep(wait)
    return pd.DataFrame(), 0.0


def _insert_rows(
    db_path: str,
    df: pd.DataFrame,
    team_id: int,
    split_type: str,
    opponent_team_id: Optional[int],
    as_of_date: str,
    season: str,
) -> int:
    if df.empty:
        return 0
    now = datetime.datetime.now().isoformat()
    rows = [
        (
            int(r.player_id), r.player_name, team_id, split_type, opponent_team_id,
            as_of_date, season,
            r.gp_on if pd.notna(r.gp_on) else None, r.gp_off if pd.notna(r.gp_off) else None,
            r.min_on if pd.notna(r.min_on) else None, r.min_off if pd.notna(r.min_off) else None,
            r.plus_minus_on if pd.notna(r.plus_minus_on) else None,
            r.plus_minus_off if pd.notna(r.plus_minus_off) else None,
            r.net_rating_on if pd.notna(r.net_rating_on) else None,
            r.net_rating_off if pd.notna(r.net_rating_off) else None,
            r.on_off_plus_minus if pd.notna(r.on_off_plus_minus) else None,
            r.on_off_net_rating if pd.notna(r.on_off_net_rating) else None,
            now,
        )
        for r in df.itertuples(index=False)
        if pd.notna(r.player_id)
    ]
    if not rows:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.executemany(
            """INSERT OR REPLACE INTO player_on_off_splits
               (player_id, player_name, team_id, split_type, opponent_team_id, as_of_date, season,
                gp_on, gp_off, min_on, min_off, plus_minus_on, plus_minus_off,
                net_rating_on, net_rating_off, on_off_plus_minus, on_off_net_rating, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def backfill_checkpoints(
    seasons: list[str],
    cadence_days: int,
    db_path: str,
    split_types: list[str] = ("overall", "home", "away"),
    force: bool = False,
    dry_run: bool = False,
    max_calls: Optional[int] = None,
) -> dict:
    migrate_player_on_off_splits(db_path)
    team_ids = _all_team_ids()
    existing = set() if force else _existing_keys(db_path)

    n_calls = 0
    n_written = 0
    n_skipped = 0
    n_empty = 0
    n_errors = 0

    for season in seasons:
        checkpoints = _season_checkpoints(season, cadence_days)
        logger.info(f"Season {season}: {len(checkpoints)} checkpoints x {len(team_ids)} teams x {len(split_types)} splits")
        for as_of_date in checkpoints:
            for team_id in team_ids:
                for split_type in split_types:
                    key = (team_id, split_type, None, as_of_date)
                    if key in existing:
                        n_skipped += 1
                        continue
                    if max_calls is not None and n_calls >= max_calls:
                        logger.info(f"Reached --max-calls={max_calls}, stopping.")
                        return _summary(n_calls, n_written, n_skipped, n_empty, n_errors)

                    n_calls += 1
                    if dry_run:
                        logger.info(f"[DRY RUN] would fetch team={team_id} split={split_type} as_of={as_of_date} season={season}")
                        continue

                    df, elapsed = _fetch_split(team_id, season, as_of_date, split_type)
                    status = "ok" if not df.empty else "empty"
                    if df.empty:
                        n_empty += 1
                    else:
                        n = _insert_rows(db_path, df, team_id, split_type, None, as_of_date, season)
                        n_written += n
                    _log_row({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "team_id": team_id, "split_type": split_type, "opponent_team_id": "",
                        "as_of_date": as_of_date, "season": season, "status": status,
                        "n_players": len(df), "elapsed_seconds": round(elapsed, 3),
                    })
                    time.sleep(SLEEP_SECONDS)

                    if n_calls % 100 == 0:
                        logger.info(f"Progress: {n_calls} calls made, {n_written} rows written, {n_skipped} skipped (cached)")

    return _summary(n_calls, n_written, n_skipped, n_empty, n_errors)


def _summary(n_calls, n_written, n_skipped, n_empty, n_errors) -> dict:
    s = {
        "n_calls": n_calls, "n_rows_written": n_written, "n_skipped_cached": n_skipped,
        "n_empty_responses": n_empty, "n_errors": n_errors,
    }
    logger.info(f"Backfill summary: {s}")
    return s


def backfill_vs_opponent_for_recent_games(
    n_games: int,
    db_path: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Lazy vs-opponent backfill: only for opponent pairings that actually occur
    in the `n_games` most recent real (Regular Season) games in `game`, one
    as_of_date per team = that game's date minus 1 day (strict leakage guard --
    DateTo was confirmed INCLUSIVE of games on that exact date, see module
    docstring)."""
    migrate_player_on_off_splits(db_path)
    with sqlite3.connect("file:data/raw/nba_api.sqlite?mode=ro", uri=True) as conn:
        games = pd.read_sql_query(
            "SELECT game_id, game_date, team_id_home, team_id_away FROM game "
            "WHERE season_type = 'Regular Season' ORDER BY game_date DESC LIMIT ?",
            conn, params=(n_games,),
        )

    existing = set() if force else _existing_keys(db_path)
    n_calls = 0
    n_written = 0
    n_empty = 0

    for _, g in games.iterrows():
        game_date = pd.to_datetime(g["game_date"])
        as_of_date = (game_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        season = _date_to_season(g["game_date"])
        for team_id, opp_id in [
            (g["team_id_home"], g["team_id_away"]),
            (g["team_id_away"], g["team_id_home"]),
        ]:
            key = (int(team_id), "vs_opponent", int(opp_id), as_of_date)
            if key in existing:
                continue
            n_calls += 1
            if dry_run:
                logger.info(f"[DRY RUN] would fetch vs_opponent team={team_id} opp={opp_id} as_of={as_of_date}")
                continue
            df, elapsed = _fetch_split(int(team_id), season, as_of_date, "vs_opponent", opponent_team_id=int(opp_id))
            status = "ok" if not df.empty else "empty"
            if df.empty:
                n_empty += 1
            else:
                n_written += _insert_rows(db_path, df, int(team_id), "vs_opponent", int(opp_id), as_of_date, season)
            _log_row({
                "timestamp": datetime.datetime.now().isoformat(),
                "team_id": int(team_id), "split_type": "vs_opponent", "opponent_team_id": int(opp_id),
                "as_of_date": as_of_date, "season": season, "status": status,
                "n_players": len(df), "elapsed_seconds": round(elapsed, 3),
            })
            time.sleep(SLEEP_SECONDS)

    return _summary(n_calls, n_written, len(existing), n_empty, 0)


def main():
    parser = argparse.ArgumentParser(description="Backfill player_on_off_splits from nba_api")
    parser.add_argument("--seasons", type=str, default=None, help="Comma-separated seasons, e.g. 2023-24,2024-25")
    parser.add_argument("--cadence-days", type=int, default=None, help="Overrides config.on_off_splits.checkpoint_cadence_days")
    parser.add_argument("--force", action="store_true", help="Refetch even if already cached")
    parser.add_argument("--update", action="store_true", help="Alias for incremental (default behavior already skips cached rows)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=None, help="Safety cap on number of API calls this run")
    parser.add_argument("--vs-opponent", action="store_true", help="Run the lazy vs-opponent backfill instead of the checkpoint backfill")
    parser.add_argument("--recent-games", type=int, default=50, help="For --vs-opponent: how many of the most recent real games to cover")
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.on_off_splits.db_path
    cadence = args.cadence_days or cfg.on_off_splits.checkpoint_cadence_days

    if args.vs_opponent:
        result = backfill_vs_opponent_for_recent_games(
            n_games=args.recent_games, db_path=db_path, force=args.force, dry_run=args.dry_run,
        )
    else:
        seasons = args.seasons.split(",") if args.seasons else [_date_to_season(datetime.date.today().isoformat())]
        result = backfill_checkpoints(
            seasons=seasons, cadence_days=cadence, db_path=db_path,
            force=args.force, dry_run=args.dry_run, max_calls=args.max_calls,
        )
    print(result)


if __name__ == "__main__":
    main()
