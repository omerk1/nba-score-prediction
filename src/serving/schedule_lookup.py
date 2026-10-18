"""
Authoritative home/away resolution via the NBA's own published schedule
(nba_api's live ScoreboardV3 endpoint) -- not inferred from a screenshot's
layout.

Vision-based home/away extraction proved unreliable in practice: the same
real screenshot flip-flopped its home/away assignment across repeated
extraction calls, even with an explicit layout rule ("right side is home")
added to the prompt. Most likely cause -- the source screenshot doesn't
actually encode home/away anywhere (no "@" marker, no explicit label, just
two teams' odds boxes side by side); a bookmaker's spread/odds display has
no reason to. Since this model's home_advantage is a large, tuned Elo term,
getting this backwards silently produces a materially wrong prediction, not
just noise -- not something to leave to a vision model's guess when the
real schedule is one API call away.
"""

import datetime
import logging

logger = logging.getLogger(__name__)


def resolve_home_away(team_a_id: int, team_b_id: int, game_date: str) -> tuple[int, int] | None:
    """Returns (home_team_id, away_team_id) for the real game between these
    two teams on game_date, per the NBA's own published schedule. Returns
    None if no matching game is found (wrong date, a game not covered by
    the live scoreboard, or the API call itself failing) -- caller decides
    the fallback."""
    from nba_api.stats.endpoints import scoreboardv3

    try:
        games = scoreboardv3.ScoreboardV3(game_date=game_date, timeout=10).get_dict()["scoreboard"]["games"]
    except Exception as e:
        logger.warning(f"Schedule lookup failed for {game_date}: {type(e).__name__}: {e}")
        return None

    teams = {team_a_id, team_b_id}
    for g in games:
        home_id, away_id = g["homeTeam"]["teamId"], g["awayTeam"]["teamId"]
        if {home_id, away_id} == teams:
            return home_id, away_id

    logger.warning(f"No scheduled game found between {team_a_id} and {team_b_id} on {game_date}")
    return None


def upcoming_games(now: datetime.datetime | None = None, hours: int = 24) -> list[dict]:
    """Real NBA games tipping off in the next `hours` hours, as
    {"home_team_id", "away_team_id", "game_date"} dicts (game_date is the
    NBA's US calendar date for the game). Raises if the schedule can't be
    fetched: the daily job treats the schedule as the source of truth for
    which games exist, so an outage must fail the run loudly, not fall
    back to whatever a screenshot reader claims is on the page."""
    from nba_api.stats.endpoints import scoreboardv3

    now = now or datetime.datetime.now(datetime.timezone.utc)
    end = now + datetime.timedelta(hours=hours)
    # US dates covering the window, padded a day each side for the
    # UTC/ET offset; the tip-off time check does the real filtering.
    days = {(now + datetime.timedelta(days=d)).date() for d in (-1, 0, 1)} | {end.date()}
    games = []
    for day in sorted(days):
        raw = scoreboardv3.ScoreboardV3(game_date=day.isoformat(), timeout=10).get_dict()["scoreboard"]["games"]
        for g in raw:
            tipoff = datetime.datetime.fromisoformat(g["gameTimeUTC"].replace("Z", "+00:00"))
            if now <= tipoff <= end:
                games.append(
                    {
                        "home_team_id": g["homeTeam"]["teamId"],
                        "away_team_id": g["awayTeam"]["teamId"],
                        "game_date": day.isoformat(),
                    }
                )
    return games
