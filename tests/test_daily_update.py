import datetime
import sqlite3

import pandas as pd
import pytest

from src.data_processing import daily_update
from src.data_processing.daily_update import (
    FreshnessResult,
    check_game_table_freshness,
    run_daily_update,
    should_run_importance_snapshot,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(daily_update.time, "sleep", lambda s: None)


def _game_db(tmp_path, game_dates):
    db_path = tmp_path / "nba_api.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE game (game_id TEXT PRIMARY KEY, game_date TEXT)")
    conn.executemany(
        "INSERT INTO game VALUES (?, ?)",
        [(f"g{i}", d) for i, d in enumerate(game_dates)],
    )
    conn.commit()
    conn.close()
    return db_path


def _patch_scoreboard(monkeypatch, scheduled_by_date):
    """fetch_upcoming_games stub: date iso string -> number of scheduled games."""
    calls = []

    def fake(target_date=None):
        calls.append(target_date)
        n = scheduled_by_date.get(target_date, 0)
        return pd.DataFrame({"game_id": [f"s{i}" for i in range(n)]})

    import src.data_processing.fetch_data as fetch_data

    monkeypatch.setattr(fetch_data, "fetch_upcoming_games", fake)
    return calls


TODAY = datetime.date(2026, 1, 10)  # mid-season


class TestRetry:
    def test_succeeds_after_transient_failures(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient")

        daily_update._retry("step", flaky)
        assert len(attempts) == 3

    def test_raises_after_max_retries(self):
        attempts = []

        def broken():
            attempts.append(1)
            raise RuntimeError("down")

        with pytest.raises(RuntimeError, match="down"):
            daily_update._retry("step", broken)
        assert len(attempts) == daily_update.MAX_RETRIES


class TestFreshnessGate:
    def test_current_table_passes_without_scoreboard_call(self, tmp_path, monkeypatch):
        db = _game_db(tmp_path, ["2026-01-09"])
        calls = _patch_scoreboard(monkeypatch, {})
        result = check_game_table_freshness(db_path=db, today=TODAY)
        assert result.fresh
        assert calls == []

    def test_empty_table_fails(self, tmp_path):
        db = _game_db(tmp_path, [])
        result = check_game_table_freshness(db_path=db, today=TODAY)
        assert not result.fresh
        assert "empty" in result.reason

    def test_stale_with_games_scheduled_yesterday_fails(self, tmp_path, monkeypatch):
        db = _game_db(tmp_path, ["2026-01-07"])
        calls = _patch_scoreboard(monkeypatch, {"2026-01-09": 5})
        result = check_game_table_freshness(db_path=db, today=TODAY)
        assert not result.fresh
        assert "2026-01-09" in result.reason
        assert calls == ["2026-01-09"]  # fails on the first scoreboard hit

    def test_stale_but_no_scheduled_games_passes(self, tmp_path, monkeypatch):
        db = _game_db(tmp_path, ["2026-01-07"])
        calls = _patch_scoreboard(monkeypatch, {})
        result = check_game_table_freshness(db_path=db, today=TODAY)
        assert result.fresh
        # checks back only to the day after the last stored game
        assert calls == ["2026-01-09", "2026-01-08"]

    def test_gap_behind_empty_yesterday_detected(self, tmp_path, monkeypatch):
        # yesterday had no games, but the day before did and is missing
        db = _game_db(tmp_path, ["2026-01-06"])
        _patch_scoreboard(monkeypatch, {"2026-01-08": 3})
        result = check_game_table_freshness(db_path=db, today=TODAY)
        assert not result.fresh
        assert "2026-01-08" in result.reason

    def test_lookback_cap_limits_scoreboard_calls(self, tmp_path, monkeypatch):
        db = _game_db(tmp_path, ["2025-11-01"])  # long gap (offseason-like)
        calls = _patch_scoreboard(monkeypatch, {})
        result = check_game_table_freshness(db_path=db, today=TODAY, lookback_days=3)
        assert result.fresh
        assert calls == ["2026-01-09", "2026-01-08", "2026-01-07"]


class TestRunDailyUpdate:
    @pytest.fixture
    def tracked_steps(self, monkeypatch):
        ran = []
        for attr in (
            "refresh_games",
            "refresh_style_cache",
            "refresh_injuries",
            "refresh_player_importance",
        ):
            monkeypatch.setattr(daily_update, attr, lambda a=attr: ran.append(a))
        monkeypatch.setattr(
            daily_update,
            "check_game_table_freshness",
            lambda: FreshnessResult(True, "stubbed"),
        )
        return ran

    def test_all_ok_returns_zero_in_order(self, tracked_steps):
        assert run_daily_update() == 0
        assert tracked_steps == ["refresh_games", "refresh_style_cache", "refresh_injuries"]

    def test_importance_snapshot_included_when_requested(self, tracked_steps):
        assert run_daily_update(include_importance_snapshot=True) == 0
        assert tracked_steps[-1] == "refresh_player_importance"

    def test_failed_step_does_not_stop_later_steps(self, tracked_steps, monkeypatch):
        def broken():
            raise RuntimeError("API down")

        monkeypatch.setattr(daily_update, "refresh_games", broken)
        assert run_daily_update() == 1
        assert tracked_steps == ["refresh_style_cache", "refresh_injuries"]

    def test_freshness_failure_fails_run(self, tracked_steps, monkeypatch):
        monkeypatch.setattr(
            daily_update,
            "check_game_table_freshness",
            lambda: FreshnessResult(False, "stale"),
        )
        assert run_daily_update() == 1

    def test_freshness_check_exception_fails_run(self, tracked_steps, monkeypatch):
        def boom():
            raise sqlite3.OperationalError("no such table: game")

        monkeypatch.setattr(daily_update, "check_game_table_freshness", boom)
        assert run_daily_update() == 1


def test_importance_snapshot_weekly_on_monday():
    assert should_run_importance_snapshot(datetime.date(2026, 1, 5))  # Monday
    assert not should_run_importance_snapshot(datetime.date(2026, 1, 6))
    assert not should_run_importance_snapshot(datetime.date(2026, 1, 11))  # Sunday
