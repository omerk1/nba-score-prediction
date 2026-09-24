"""
Re-parse the NBA injury report PDFs, keeping each row's own game date.

`player_injuries.game_date` is really the date of the report that contained the
row, and an 11PM report covers both that evening's late games and the next
day's, so most rows are currently filed against the wrong game
(docs/PIPELINE_AUDIT.md, 2026-09-17). The PDFs carry a 'Game Date' column that
the parser never read; it is now read and carried forward.

This writes to its own database (data/raw/injury_dates.sqlite) and never
touches injury_features.sqlite, so the current champion stays reproducible
until the downstream ablation decides. Fetched PDFs are cached to disk, so
re-parsing after a parser change costs no downloads.

Tables written:
  player_injuries_dated(game_date, report_date, report_time, team_id,
                        player_name, status, reason)
  extraction_log(report_date, ...telemetry..., fetched_at)

Usage:
  venv/bin/python3 scripts/rebuild_injury_dates.py                # every report date on record
  venv/bin/python3 scripts/rebuild_injury_dates.py --limit 30     # smoke test
  venv/bin/python3 scripts/rebuild_injury_dates.py --validate     # compare against the schedule
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.news_scraping.scrapers.nba_injury_pdf import fetch_injuries_for_date
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DB = "data/raw/injury_dates.sqlite"
PDF_CACHE = "data/raw/injury_pdfs"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_injuries_dated (
    game_date   TEXT NOT NULL,
    report_date TEXT NOT NULL,
    report_time TEXT,
    team_id     INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT,
    date_source TEXT NOT NULL DEFAULT 'pdf',
    PRIMARY KEY (game_date, report_date, team_id, player_name)
);
CREATE INDEX IF NOT EXISTS idx_pid_game_date ON player_injuries_dated(game_date, team_id);

CREATE TABLE IF NOT EXISTS extraction_log (
    report_date TEXT PRIMARY KEY,
    report_time TEXT,
    stats_json  TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);
"""


def get_conn(path: str = OUT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def report_dates(injury_db: str) -> list[date]:
    with sqlite3.connect(injury_db) as c:
        rows = c.execute(
            "SELECT DISTINCT game_date FROM scrape_log WHERE source='pdf' ORDER BY game_date"
        ).fetchall()
    return [date.fromisoformat(r[0]) for r in rows]


def team_id_map() -> dict[str, int]:
    from nba_api.stats.static import teams as nba_teams

    return {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}


def schedule_index(raw_db: str) -> dict[tuple[int, str], bool]:
    """{(team_id, game_date): True} for every scheduled team-game."""
    with sqlite3.connect(raw_db) as c:
        g = pd.read_sql_query(
            "SELECT substr(game_date,1,10) AS d, team_id_home, team_id_away FROM game "
            "WHERE game_date >= '2021-06-01'",
            c,
        )
    idx = set()
    for _, r in g.iterrows():
        idx.add((int(r["team_id_home"]), r["d"]))
        idx.add((int(r["team_id_away"]), r["d"]))
    return {k: True for k in idx}


def resolve_from_schedule(team_id: int, report_date: str, sched: dict) -> tuple[str | None, str]:
    """Date a listing when the PDF gave no date, using the schedule.

    A report dated D covers D's games and D+1's. If the team is scheduled on
    exactly one of those, that is the game, and `validate` scores this branch at
    100.00% over 34,397 rows whose PDFs carry a real date.

    If the team is scheduled on both, it is on a back-to-back and the row could
    belong to either. Measured on those same real-date rows, D is right 97.9% of
    the time and D+1 only 2.1%, so D is the tie-break. This is an empirical
    result, not a deduction: the first version of this rule assumed D+1 on the
    grounds that a late report previews the next day, and validation showed that
    backwards. The branch stays labelled so its share and accuracy remain
    measurable.
    """
    nxt = (date.fromisoformat(report_date) + pd.Timedelta(days=1)).isoformat()
    on_d = sched.get((team_id, report_date), False)
    on_next = sched.get((team_id, nxt), False)
    if on_d and on_next:
        return report_date, "schedule_ambiguous"
    if on_next:
        return nxt, "schedule"
    if on_d:
        return report_date, "schedule"
    return None, "unresolved"


def rebuild(dates: list[date], out_db: str, cache_dir: str, force: bool, raw_db: str) -> None:
    conn = get_conn(out_db)
    abbr_to_id = team_id_map()
    sched = schedule_index(raw_db)
    done = {r[0] for r in conn.execute("SELECT report_date FROM extraction_log")}
    for i, d in enumerate(dates, 1):
        iso = d.isoformat()
        if iso in done and not force:
            continue
        entries, report_time, stats = fetch_injuries_for_date(d, cache_dir=cache_dir)
        rows = []
        for e in entries:
            tid = abbr_to_id.get(e["team_abbreviation"])
            if tid is None:
                continue
            gd, src = e.get("game_date"), "pdf"
            if not gd:
                gd, src = resolve_from_schedule(tid, iso, sched)
            if not gd:
                continue
            rows.append((gd, iso, report_time, tid, e["player_name"], e["status"], e["reason"], src))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO player_injuries_dated "
                "(game_date, report_date, report_time, team_id, player_name, status, reason, date_source) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
        conn.execute(
            "INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?)",
            (iso, report_time, json.dumps(stats), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        if i % 25 == 0 or i == len(dates):
            logger.info(f"{i}/{len(dates)} reports processed")
    n = conn.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM player_injuries_dated").fetchone()
    logger.info(f"player_injuries_dated: {n[0]} rows, game dates {n[1]} to {n[2]}")
    conn.close()


def validate(out_db: str, raw_db: str) -> None:
    """Gate: extracted game dates must line up with the schedule."""
    with sqlite3.connect(out_db) as c:
        df = pd.read_sql_query(
            "SELECT game_date, report_date, team_id, player_name, status, date_source FROM player_injuries_dated",
            c,
        )
        logs = pd.read_sql_query("SELECT report_date, stats_json FROM extraction_log", c)
    if df.empty:
        raise SystemExit("nothing to validate; run the rebuild first")

    with sqlite3.connect(raw_db) as c:
        g = pd.read_sql_query(
            "SELECT substr(game_date,1,10) AS game_date, team_id_home, team_id_away FROM game "
            "WHERE game_date >= '2021-06-01'",
            c,
        )
    sched = (
        pd.concat(
            [
                g[["game_date", "team_id_home"]].rename(columns={"team_id_home": "team_id"}),
                g[["game_date", "team_id_away"]].rename(columns={"team_id_away": "team_id"}),
            ]
        )
        .drop_duplicates()
        .assign(scheduled=1)
    )

    m = df.merge(sched, on=["game_date", "team_id"], how="left")
    m["scheduled"] = m["scheduled"].fillna(0).astype(int)
    agree = m["scheduled"].mean()

    offset = (pd.to_datetime(m["game_date"]) - pd.to_datetime(m["report_date"])).dt.days
    stats = pd.DataFrame([json.loads(s) for s in logs["stats_json"]])

    print(f"rows: {len(m)}   reports: {len(logs)}")
    print(f"game date matches a scheduled game for that team: {agree:.4f}   (gate: >= 0.99)")
    print("\nwhere each row's date came from:")
    print(m["date_source"].value_counts(normalize=True).round(4).to_string())

    # The schedule rule is only needed from 2023-24 on, where the PDFs stopped
    # exposing a date column. Score it against the earlier seasons, whose PDFs
    # do carry a real date, so the rule is measured rather than assumed.
    truth = m[m["date_source"] == "pdf"].copy()
    if len(truth):
        sched_idx = {(int(t), d): True for d, t in zip(sched["game_date"], sched["team_id"])}
        guessed, kinds = [], []
        for _, r in truth.iterrows():
            g, k = resolve_from_schedule(int(r["team_id"]), r["report_date"], sched_idx)
            guessed.append(g)
            kinds.append(k)
        truth["guess"], truth["kind"] = guessed, kinds
        ok = truth["guess"] == truth["game_date"]
        print(f"\nschedule rule scored against {len(truth)} rows with a real PDF date:")
        print(f"  overall agreement: {ok.mean():.4f}")
        print(truth.assign(correct=ok).groupby("kind")["correct"].agg(["size", "mean"]).round(4).to_string())
    print("\ngame date minus report date, share of rows:")
    print(offset.value_counts(normalize=True).sort_index().round(4).to_string())
    print("\nextraction telemetry totals:")
    for col in [
        "pages",
        "pages_no_table",
        "pages_no_header",
        "pages_missing_columns",
        "rows_seen",
        "rows_kept",
        "rows_no_team",
        "rows_no_date",
    ]:
        if col in stats:
            print(f"  {col}: {int(stats[col].sum())}")
    if "has_date_column" in stats:
        print(f"  reports with a Game Date column: {int(stats['has_date_column'].sum())}/{len(stats)}")
    bad = m[m["scheduled"] == 0]
    if len(bad):
        print(f"\n{len(bad)} rows whose game date has no scheduled game; sample:")
        print(bad.head(8).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-db", default=OUT_DB)
    ap.add_argument("--cache-dir", default=PDF_CACHE)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N report dates")
    ap.add_argument("--force", action="store_true", help="re-parse reports already logged")
    ap.add_argument("--validate", action="store_true", help="validate only, no fetching")
    args = ap.parse_args()

    cfg = load_config()
    if not args.validate:
        dates = report_dates(cfg.injury_features.db_path)
        if args.limit:
            dates = dates[: args.limit]
        logger.info(f"{len(dates)} report dates to process")
        rebuild(dates, args.out_db, args.cache_dir, args.force, cfg.data_paths.raw_db)
    validate(args.out_db, cfg.data_paths.raw_db)


if __name__ == "__main__":
    main()
