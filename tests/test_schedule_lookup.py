import datetime
import sys
import types

from src.serving import schedule_lookup


def _fake_scoreboard(monkeypatch, by_date):
    class Fake:
        def __init__(self, game_date, timeout):
            self.d = game_date

        def get_dict(self):
            return {"scoreboard": {"games": by_date.get(self.d, [])}}

    mod = types.SimpleNamespace(ScoreboardV3=Fake)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints.scoreboardv3", mod)
    monkeypatch.setattr("nba_api.stats.endpoints.scoreboardv3", mod, raising=False)


def _game(home, away, tip):
    return {"homeTeam": {"teamId": home}, "awayTeam": {"teamId": away}, "gameTimeUTC": tip}


def test_upcoming_games_keeps_only_next_24h(monkeypatch):
    now = datetime.datetime(2026, 10, 20, 13, 0, tzinfo=datetime.timezone.utc)
    _fake_scoreboard(
        monkeypatch,
        {
            "2026-10-20": [_game(1, 2, "2026-10-20T23:00:00Z"), _game(3, 4, "2026-10-20T12:00:00Z")],
            "2026-10-21": [_game(5, 6, "2026-10-21T23:00:00Z")],
        },
    )
    games = schedule_lookup.upcoming_games(now)
    assert [(g["home_team_id"], g["game_date"]) for g in games] == [(1, "2026-10-20")]
