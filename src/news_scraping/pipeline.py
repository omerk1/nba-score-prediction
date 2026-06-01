"""
Injury features pipeline: scrape → extract → store.

Two entry points:
  run_nightly()    — fetch ESPN injury data for today, extract impact, store
  run_historical() — backfill from NBA official injury report PDFs for a date range

Scoring is controlled by injury_features.scorer in config:
  formula — deterministic weighted sum, fast, fully reproducible
  llm     — Gemini call, richer reasoning, requires GOOGLE_API_KEY

Raw player injury records are always stored in player_injuries regardless of scorer,
so switching scorer and re-running only recomputes injury_features — no re-scraping needed.
Run player_importance.backfill_season() before run_historical() so importance
scores are available for scoring.
"""

import logging
import math
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

from src.news_scraping.db import get_conn, init_db
from src.news_scraping.scrapers.espn_injuries import fetch_current_injuries
from src.news_scraping.scrapers.nba_injury_pdf import fetch_injuries_for_date
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def _resolve_team_id(abbreviation: str) -> int | None:
    """Map team abbreviation to NBA team_id via nba_api static data."""
    from nba_api.stats.static import teams
    match = next((t for t in teams.get_teams() if t["abbreviation"] == abbreviation), None)
    return match["id"] if match else None


def _get_importance_map(db_path: str, team_id: int, game_date: str) -> dict[str, float]:
    """Compute importance scores on the fly from raw stats for each player on the team.

    Reads the latest snapshot before game_date, then applies the weighted formula
    using current config weights — so changing weights requires no re-fetch.
    """
    cfg = load_config()
    w = cfg.injury_features.importance_weights

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT player_name, minutes_per_game, pts_per_game, usage_rate
               FROM player_importance
               WHERE team_id = ? AND as_of_date < ?
               GROUP BY player_id
               HAVING as_of_date = MAX(as_of_date)""",
            (team_id, game_date),
        ).fetchall()

    if not rows:
        return {}

    players = [dict(r) for r in rows]
    total_min = sum(p["minutes_per_game"] for p in players) or 1
    total_pts = sum(p["pts_per_game"] for p in players) or 1
    max_usg = max(p["usage_rate"] for p in players) or 1

    return {
        p["player_name"]: min(max(
            (p["minutes_per_game"] / total_min) * w.minutes_share
            + (p["usage_rate"] / max_usg) * w.usage_rate
            + (p["pts_per_game"] / total_pts) * w.pts_share,
            0), 1)
        for p in players
    }


def _store_player_injuries(db_path: str, game_date: str, team_id: int, players: list[dict]) -> None:
    """Persist raw injury records — always called regardless of scorer."""
    rows = [
        (game_date, team_id, p["player_name"], p.get("status", ""), p.get("reason", ""), p.get("days_out", 0))
        for p in players
    ]
    with get_conn(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO player_injuries
               (game_date, team_id, player_name, status, reason, days_out)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )


def _upsert_injury_feature(
    db_path: str, game_date: str, team_id: int, scorer: str, impact: dict
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO injury_features
               (game_date, team_id, scorer, impact_score, n_out, n_questionable, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                game_date, team_id, scorer,
                impact["impact_score"], impact["n_out"], impact["n_questionable"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _get_games_out_map(db_path: str, game_date: str, team_id: int, player_names: list[str]) -> dict[str, int]:
    """
    Count prior game-dates each player already appears as Out in player_injuries.
    Called after _store_player_injuries, so today is excluded via game_date < current.
    A player appearing for the first time returns 0 (full impact, no decay).
    """
    if not player_names:
        return {}
    placeholders = ",".join("?" * len(player_names))
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT player_name, COUNT(DISTINCT game_date) "
            f"FROM player_injuries "
            f"WHERE team_id = ? AND game_date < ? AND status = 'Out' "
            f"AND player_name IN ({placeholders}) "
            f"GROUP BY player_name",
            (team_id, game_date, *player_names),
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def _absence_decay(players: list[dict], importance_map: dict[str, float], games_out_map: dict[str, int], rolling_window: int) -> float:
    """
    Importance-weighted average of per-player decay factors.

    The rolling stats absorb a player's absence over time, so the marginal injury
    impact shrinks the longer they've been out. Uses actual prior game appearances
    from player_injuries as the games-missed count — correctly treats a player
    appearing for the first time the same as a one-day absence regardless of how
    long they were listed in consecutive reports.

    Half-life = rolling_window games. At 1× window: factor ≈ 0.37. Floor at 0.1.
    """
    weights, factors = [], []
    for p in players:
        if p.get("status") != "Out":
            continue
        games_missed = games_out_map.get(p["player_name"], 0)
        importance = importance_map.get(p["player_name"], 0.0)
        factor = max(0.1, math.exp(-games_missed / rolling_window))
        weights.append(importance)
        factors.append(factor)

    total_weight = sum(weights)
    if not weights or total_weight == 0.0:
        return 1.0
    return sum(w * f for w, f in zip(weights, factors)) / total_weight


def _score_team(scorer: str, team_name: str, game_date: str, players: list[dict], importance_map: dict, cfg) -> dict:
    """Dispatch to formula or LLM scorer based on config."""
    if scorer == "llm":
        from src.news_scraping.extractors.llm_extractor import extract_impact
        return extract_impact(team_name, game_date, players, importance_map)
    from src.news_scraping.extractors.formula_scorer import score_team
    return score_team(players, importance_map, cfg.injury_features.formula_weights)


def _process_team(
    db_path: str, team_id: int, team_name: str, game_date: str, players: list[dict]
) -> None:
    cfg = load_config()
    scorer = cfg.injury_features.scorer

    _store_player_injuries(db_path, game_date, team_id, players)

    with get_conn(db_path) as conn:
        already_scored = conn.execute(
            "SELECT 1 FROM injury_features WHERE game_date = ? AND team_id = ? AND scorer = ? LIMIT 1",
            (game_date, team_id, scorer),
        ).fetchone()
    if already_scored:
        logger.debug(f"  {team_name} [{scorer}]: already scored for {game_date}, skipping")
        return

    importance_map = _get_importance_map(db_path, team_id, game_date)
    out_names = [p["player_name"] for p in players if p.get("status") == "Out"]
    games_out_map = _get_games_out_map(db_path, game_date, team_id, out_names)

    impact = _score_team(scorer, team_name, game_date, players, importance_map, cfg)
    decay = _absence_decay(players, importance_map, games_out_map, cfg.features.rolling_window)
    impact = dict(impact)
    impact["impact_score"] = round(impact["impact_score"] * decay, 2)

    _upsert_injury_feature(db_path, game_date, team_id, scorer, impact)
    logger.debug(
        f"  {team_name} [{scorer}]: impact={impact['impact_score']:.2f} | "
        f"out={impact['n_out']} | questionable={impact['n_questionable']}"
    )


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

    Downloads NBA official injury report PDFs (available from the 2021-22 season).
    Dates with no report (pre-2021, off-season, no games) are silently skipped.
    Requires player_importance table to be populated first.
    """
    cfg = load_config()
    if not cfg.injury_features or not cfg.injury_features.enabled:
        logger.info("Injury features disabled — skipping historical run")
        return

    db_path = cfg.injury_features.db_path
    init_db(db_path)

    # Only process dates that have actual games — skips off-season, all-star break, etc.
    allowed_types = tuple(cfg.datasets_loading.allowed_season_types)
    placeholders = ",".join("?" * len(allowed_types))
    with sqlite3.connect(cfg.data_paths.raw_db) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT game_date FROM game "
            f"WHERE game_date >= ? AND game_date <= ? AND season_type IN ({placeholders}) "
            f"ORDER BY game_date",
            (str(start_date), str(end_date), *allowed_types),
        ).fetchall()
    game_dates = [date.fromisoformat(row[0][:10]) for row in rows]
    logger.info(f"Found {len(game_dates)} game dates to process")

    for game_date in game_dates:
        # Injury PDFs are pre-game filings (submitted ≥30 min before tip-off).
        # Using the latest report for game_date gives the most complete pre-game picture
        # with no data leakage — we never use date+1 information for date's games.
        entries = fetch_injuries_for_date(game_date)
        if entries:
            date_str = str(game_date)
            by_team: dict[str, list] = {}
            for e in entries:
                by_team.setdefault(e["team_abbreviation"], []).append(e)

            logger.info(f"Processing {date_str} ({len(by_team)} teams with injuries)")
            for abbr, players in by_team.items():
                team_id = _resolve_team_id(abbr)
                if not team_id:
                    logger.warning(f"Unknown team abbreviation: {abbr}")
                    continue
                _process_team(db_path, team_id, abbr, date_str, players)
