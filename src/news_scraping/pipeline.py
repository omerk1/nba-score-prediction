"""
Injury features pipeline: scrape → extract → store.

Two entry points:
  run_nightly()    — fetch ESPN injury data for today, extract impact, store
  run_historical() — backfill from prosports-transactions.com for a date range

Run player_importance.backfill_season() before run_historical() so importance
scores are available for the LLM context.
"""

import logging
import math
from datetime import date, datetime, timedelta, timezone

from src.news_scraping.db import get_conn, init_db
from src.news_scraping.extractors.llm_extractor import extract_impact
from src.news_scraping.scrapers.espn_injuries import fetch_current_injuries
from src.news_scraping.scrapers.prosports_transactions import fetch_season_transactions, snapshot_at_date
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def _resolve_team_id(abbreviation: str) -> int | None:
    """Map team abbreviation to NBA team_id via nba_api static data."""
    from nba_api.stats.static import teams
    match = next((t for t in teams.get_teams() if t["abbreviation"] == abbreviation), None)
    return match["id"] if match else None


def _get_importance_map(db_path: str, team_id: int, game_date: str) -> dict[str, float]:
    """Latest importance snapshot before game_date for each player on the team."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT player_name, importance_score
               FROM player_importance
               WHERE team_id = ? AND as_of_date < ?
               GROUP BY player_id
               HAVING as_of_date = MAX(as_of_date)""",
            (team_id, game_date),
        ).fetchall()
    return {r["player_name"]: r["importance_score"] for r in rows}


def _upsert_injury_feature(
    db_path: str, game_date: str, team_id: int, impact: dict, raw: str
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO injury_features
               (game_date, team_id, impact_score, n_out, n_questionable, star_out, raw_report, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_date, team_id,
                impact["impact_score"], impact["n_out"], impact["n_questionable"],
                int(impact["star_out"]), raw,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _absence_decay(players: list[dict], importance_map: dict[str, float], rolling_window: int) -> float:
    """
    Importance-weighted average of per-player decay factors.

    The rolling stats absorb a player's absence over time, so the marginal injury
    impact shrinks the longer they've been out. Uses days_out as a proxy for games
    missed (days_out / 2.5 ≈ games missed for a typical NBA schedule).

    Half-life = rolling_window games. At 1× window: factor ≈ 0.37. Floor at 0.1.
    Stars who've been out longer carry more weight in the aggregate decay.
    ESPN entries lack days_out → defaults to 0 (full impact, no decay).
    """
    weights, factors = [], []
    for p in players:
        if p.get("status") != "Out":
            continue
        days = p.get("days_out", 0)
        importance = importance_map.get(p["player_name"], 0.1)
        games_missed = days / 2.5
        factor = max(0.1, math.exp(-games_missed / rolling_window))
        weights.append(importance)
        factors.append(factor)

    if not weights:
        return 1.0
    return sum(w * f for w, f in zip(weights, factors)) / sum(weights)


def _process_team(
    db_path: str, team_id: int, team_name: str, game_date: str, players: list[dict]
) -> None:
    cfg = load_config()
    importance_map = _get_importance_map(db_path, team_id, game_date)
    impact = extract_impact(team_name, game_date, players, importance_map)

    decay = _absence_decay(players, importance_map, cfg.features.rolling_window)
    impact = dict(impact)
    impact["impact_score"] = round(impact["impact_score"] * decay, 2)

    _upsert_injury_feature(db_path, game_date, team_id, impact, str(players))
    logger.info(
        f"  {team_name}: impact={impact['impact_score']:.1f} | "
        f"out={impact['n_out']} | star={impact['star_out']}"
    )


def _iter_seasons(start_date: date, end_date: date):
    """Yield (season_start, season_end) pairs that overlap with the given range."""
    year = start_date.year if start_date.month >= 10 else start_date.year - 1
    while True:
        season_start = date(year, 10, 1)
        season_end = date(year + 1, 6, 30)
        overlap_start = max(season_start, start_date)
        overlap_end = min(season_end, end_date)
        if overlap_start > end_date:
            break
        yield season_start, season_end, overlap_start, overlap_end
        year += 1


def run_nightly(game_date: date | None = None) -> None:
    """Fetch ESPN injury data for game_date (default: today) and store impact scores."""
    cfg = load_config()
    if not cfg.injury_features or not cfg.injury_features.enabled:
        logger.info("Injury features disabled — skipping nightly run")
        return

    game_date = game_date or date.today()
    game_date_str = str(game_date)
    db_path = cfg.injury_features.db_path
    init_db(db_path)

    entries = fetch_current_injuries()
    by_team: dict[str, list] = {}
    for e in entries:
        by_team.setdefault(e["team_abbreviation"], []).append(e)

    for abbr, players in by_team.items():
        team_id = _resolve_team_id(abbr)
        if not team_id:
            logger.warning(f"Unknown team abbreviation: {abbr}")
            continue
        _process_team(db_path, team_id, abbr, game_date_str, players)


def run_historical(start_date: date, end_date: date) -> None:
    """
    Backfill injury_features for every day in [start_date, end_date].

    Fetches prosports transactions once per season (not once per date),
    then replays in memory for each game date — much faster than per-date fetching.
    Requires player_importance table to be populated first.
    """
    cfg = load_config()
    if not cfg.injury_features or not cfg.injury_features.enabled:
        logger.info("Injury features disabled — skipping historical run")
        return

    db_path = cfg.injury_features.db_path
    init_db(db_path)

    for season_start, season_end, overlap_start, overlap_end in _iter_seasons(start_date, end_date):
        logger.info(f"Season starting {season_start.year}: fetching transactions once")
        transactions = fetch_season_transactions(season_start, season_end)

        cursor = overlap_start
        while cursor <= overlap_end:
            snapshot = snapshot_at_date(transactions, cursor)
            if snapshot:
                date_str = str(cursor)
                logger.info(f"Processing {date_str} ({len(snapshot)} teams with injuries)")
                for abbr, players in snapshot.items():
                    team_id = _resolve_team_id(abbr)
                    if not team_id:
                        continue
                    _process_team(db_path, team_id, abbr, date_str, players)
            cursor += timedelta(days=1)
