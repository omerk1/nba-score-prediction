"""
Backfill raw play-by-play events (nba_api PlayByPlayV3) and build the
possession table for one or more seasons.

Resumable: every game's fetch outcome is recorded in pbp_fetch_log, and games
already marked 'ok' are skipped unless --force is passed. Failed games are
retried on the next run. Parsing is idempotent (INSERT OR REPLACE on
(game_id, poss_idx)) and can be re-run without re-fetching via --parse-only.

Usage:
    venv/bin/python3 scripts/backfill_pbp.py --seasons 2024-25
    venv/bin/python3 scripts/backfill_pbp.py --seasons 2024-25 --limit 20      # smoke test
    venv/bin/python3 scripts/backfill_pbp.py --seasons 2024-25 --parse-only    # re-parse cached events
    venv/bin/python3 scripts/backfill_pbp.py --seasons 2024-25 --report        # parse-quality summary only
"""

import argparse
import datetime
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.fetch_data import _date_to_season
from src.pbp import collector
from src.pbp.db import connect
from src.pbp.possessions import build_possessions
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _season_games(raw_db: str, seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    with sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True) as conn:
        placeholders = ",".join("?" * len(season_types))
        games = pd.read_sql_query(
            "SELECT game_id, game_date, team_id_home, team_id_away, pts_home, pts_away "
            f"FROM game WHERE season_type IN ({placeholders}) ORDER BY game_date, game_id",
            conn, params=season_types,
        )
    games["season"] = games["game_date"].map(_date_to_season)
    return games[games["season"].isin(seasons)].reset_index(drop=True)


def fetch_games(conn: sqlite3.Connection, games: pd.DataFrame, force: bool, limit: int | None) -> None:
    done = set() if force else collector.fetched_game_ids(conn, "ok")
    todo = games[~games.game_id.isin(done)]
    n_cached = len(games) - len(todo)
    if limit:
        todo = todo.head(limit)
    logger.info(f"Fetching {len(todo):,} games ({n_cached:,} already cached, {len(games):,} in season)")
    n_ok = n_err = 0
    t_start = time.time()
    for i, g in enumerate(todo.itertuples(index=False), 1):
        t0 = time.time()
        try:
            events = collector.fetch_game_events(g.game_id)
            n = collector.store_game_events(conn, events)
            status = "ok" if n else "empty"
            collector.log_fetch(conn, g.game_id, g.season, g.game_date, status, n, time.time() - t0)
            n_ok += int(n > 0)
        except Exception as e:  # noqa: BLE001 -- recorded for retry on the next run
            collector.log_fetch(conn, g.game_id, g.season, g.game_date, "error", 0, time.time() - t0, str(e)[:500])
            n_err += 1
        conn.commit()
        if i % 50 == 0 or i == len(todo):
            rate = (time.time() - t_start) / i
            logger.info(f"{i:,}/{len(todo):,} fetched | ok={n_ok} err={n_err} | {rate:.2f}s/game | "
                        f"eta {rate * (len(todo) - i) / 60:.1f} min")


def parse_games(conn: sqlite3.Connection, games: pd.DataFrame, force: bool) -> None:
    fetched = collector.fetched_game_ids(conn, "ok")
    parsed = set() if force else {r[0] for r in conn.execute("SELECT game_id FROM possession_game_summary")}
    todo = games[games.game_id.isin(fetched) & ~games.game_id.isin(parsed)]
    logger.info(f"Parsing {len(todo):,} games")
    n_bad = 0
    for i, g in enumerate(todo.itertuples(index=False), 1):
        events = collector.load_game_events(conn, g.game_id)
        try:
            poss, s = build_possessions(events, int(g.team_id_home), int(g.team_id_away))
        except Exception as e:  # noqa: BLE001
            logger.error(f"parse failed for {g.game_id}: {e}")
            n_bad += 1
            continue
        conn.execute("DELETE FROM possessions WHERE game_id = ?", (g.game_id,))
        if len(poss):
            cols = list(poss.columns)
            conn.executemany(
                f"INSERT INTO possessions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [tuple(None if pd.isna(v) else v for v in r) for r in poss.itertuples(index=False)],
            )
        reconciled = int(
            s["pts_home_poss"] + s["tech_ft_pts_home"] == g.pts_home
            and s["pts_away_poss"] + s["tech_ft_pts_away"] == g.pts_away
        )
        conn.execute(
            "INSERT OR REPLACE INTO possession_game_summary (game_id, home_team_id, away_team_id, "
            "n_poss_home, n_poss_away, pts_home_poss, pts_away_poss, tech_ft_pts_home, tech_ft_pts_away, "
            "pts_home_box, pts_away_box, points_reconciled, lineup_complete_rate, n_events, parsed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (g.game_id, s["home_team_id"], s["away_team_id"], s["n_poss_home"], s["n_poss_away"],
             s["pts_home_poss"], s["pts_away_poss"], s["tech_ft_pts_home"], s["tech_ft_pts_away"],
             g.pts_home, g.pts_away, reconciled, s["lineup_complete_rate"], s["n_events"],
             datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
        )
        if i % 100 == 0:
            conn.commit()
            logger.info(f"{i:,}/{len(todo):,} parsed")
    conn.commit()
    if n_bad:
        logger.warning(f"{n_bad} games failed to parse")


def report(conn: sqlite3.Connection, games: pd.DataFrame) -> None:
    ids = tuple(games.game_id)
    log = pd.read_sql_query("SELECT * FROM pbp_fetch_log", conn)
    log = log[log.game_id.isin(ids)]
    summ = pd.read_sql_query("SELECT * FROM possession_game_summary", conn)
    summ = summ[summ.game_id.isin(ids)]
    print(f"games in season(s): {len(games):,}")
    print(f"fetch status: {log.status.value_counts().to_dict()}  "
          f"(mean {log.elapsed_s.mean():.2f}s/game, events/game {log.n_events.mean():.0f})" if len(log) else "no fetches")
    if summ.empty:
        print("no games parsed")
        return
    poss_per_team = pd.concat([summ.n_poss_home, summ.n_poss_away])
    print(f"parsed: {len(summ):,} | points reconciled: {summ.points_reconciled.mean():.1%} "
          f"({int((1 - summ.points_reconciled).sum())} off)")
    print(f"possessions per team-game: mean {poss_per_team.mean():.1f}, "
          f"p5 {poss_per_team.quantile(.05):.0f}, p95 {poss_per_team.quantile(.95):.0f}")
    print(f"lineup-complete rate: mean {summ.lineup_complete_rate.mean():.1%}, "
          f"games fully complete {(summ.lineup_complete_rate == 1).mean():.1%}, "
          f"games < 90% {(summ.lineup_complete_rate < 0.9).mean():.1%}")
    bad = summ[summ.points_reconciled == 0]
    if len(bad):
        bad = bad.assign(dh=bad.pts_home_poss + bad.tech_ft_pts_home - bad.pts_home_box,
                         da=bad.pts_away_poss + bad.tech_ft_pts_away - bad.pts_away_box)
        print("unreconciled games (game_id, home delta, away delta):")
        print(bad[["game_id", "dh", "da"]].head(20).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", required=True, help="comma-separated, e.g. 2024-25,2025-26")
    p.add_argument("--season-types", default="Regular Season", help="comma-separated season_type values")
    p.add_argument("--limit", type=int, default=None, help="fetch at most N games (smoke test)")
    p.add_argument("--force", action="store_true", help="refetch/reparse games already done")
    p.add_argument("--parse-only", action="store_true", help="skip fetching; parse cached events")
    p.add_argument("--report", action="store_true", help="print parse-quality summary only")
    args = p.parse_args()

    cfg = load_config()
    db_path = cfg.pbp.db_path if cfg.pbp else "data/raw/pbp.sqlite"
    seasons = [s.strip() for s in args.seasons.split(",")]
    season_types = [s.strip() for s in args.season_types.split(",")]
    games = _season_games(cfg.data_paths.raw_db, seasons, season_types)
    conn = connect(db_path)
    try:
        if args.report:
            report(conn, games)
            return
        if not args.parse_only:
            fetch_games(conn, games, args.force, args.limit)
        parse_games(conn, games, args.force)
        report(conn, games)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
