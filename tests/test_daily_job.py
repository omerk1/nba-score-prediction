import datetime
from pathlib import Path

import pytest

from src.serving import daily_job
from src.serving.daily_job import acquire_lock, prune_old_logs, run_daily_job
from src.serving.notify_telegram import SendStatus

TODAY = datetime.date(2026, 10, 6)
PICK = {"home_team_id": 1, "away_team_id": 2}


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    """Happy-path step stubs; individual tests break specific steps.
    Records call order and every notice sent."""
    calls = {"order": [], "notices": []}
    monkeypatch.setattr(daily_job, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(daily_job, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(daily_job, "refresh_data", lambda: calls["order"].append("refresh") or 0)
    monkeypatch.setattr(
        daily_job, "capture", lambda: calls["order"].append("capture") or Path("shot.png")
    )
    monkeypatch.setattr(
        daily_job, "extract", lambda p: calls["order"].append("extract") or [PICK, PICK]
    )
    monkeypatch.setattr(
        daily_job,
        "recommend_all",
        lambda picks: calls["order"].append("recommend") or ([{"rec": 1}] * len(picks), []),
    )
    monkeypatch.setattr(
        daily_job,
        "notify",
        lambda recs: calls["order"].append("notify") or SendStatus.SENT,
    )
    monkeypatch.setattr(
        daily_job, "notice", lambda text: calls["notices"].append(text) or SendStatus.SENT
    )
    return calls


class TestLockAndLogs:
    def test_lock_prevents_second_acquirer(self, tmp_path):
        first = acquire_lock(tmp_path / "lock")
        assert first is not None
        assert acquire_lock(tmp_path / "lock") is None
        first.close()
        second = acquire_lock(tmp_path / "lock")
        assert second is not None
        second.close()

    def test_prune_removes_only_expired_job_logs(self, tmp_path):
        (tmp_path / "daily_job_2026-08-01.log").write_text("old")
        (tmp_path / "daily_job_2026-10-01.log").write_text("recent")
        (tmp_path / "launchd_daily_job.err.log").write_text("keep")
        assert prune_old_logs(tmp_path, retention_days=30, today=TODAY) == 1
        assert not (tmp_path / "daily_job_2026-08-01.log").exists()
        assert (tmp_path / "daily_job_2026-10-01.log").exists()
        assert (tmp_path / "launchd_daily_job.err.log").exists()


class TestRunDailyJob:
    def test_happy_path_order_and_exit_zero(self, tracked):
        assert run_daily_job() == 0
        assert tracked["order"] == ["refresh", "capture", "extract", "recommend", "notify"]
        assert tracked["notices"] == []

    def test_lock_held_exits_quietly_without_running(self, tracked, monkeypatch, tmp_path):
        held = acquire_lock(tmp_path / "lock")
        assert run_daily_job() == 0
        assert tracked["order"] == []
        held.close()

    def test_refresh_failure_aborts_before_capture_with_notice(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "refresh_data", lambda: 1)
        assert run_daily_job() == 1
        assert "capture" not in tracked["order"]
        assert any("refresh" in n.lower() for n in tracked["notices"])

    def test_capture_failure_sends_notice(self, tracked, monkeypatch):
        def boom():
            raise RuntimeError("Imperva said no")

        monkeypatch.setattr(daily_job, "capture", boom)
        assert run_daily_job() == 1
        assert any("capture FAILED" in n for n in tracked["notices"])
        assert "extract" not in tracked["order"]

    def test_extraction_failure_sends_notice(self, tracked, monkeypatch):
        def boom(path):
            raise ValueError("model returned garbage")

        monkeypatch.setattr(daily_job, "extract", boom)
        assert run_daily_job() == 1
        assert any("extraction FAILED" in n for n in tracked["notices"])

    def test_empty_slate_is_a_normal_notice_exit_zero(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "extract", lambda p: [])
        assert run_daily_job() == 0
        assert any("no NBA games" in n for n in tracked["notices"])
        assert "recommend" not in tracked["order"]

    def test_partial_recommend_failures_still_send_with_note(self, tracked, monkeypatch):
        monkeypatch.setattr(
            daily_job, "recommend_all", lambda picks: ([{"rec": 1}], ["1 vs 2: no history"])
        )
        assert run_daily_job() == 0
        assert "notify" in tracked["order"]
        assert any("skipped" in n for n in tracked["notices"])

    def test_all_recommends_failed_notice_and_exit_one(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "recommend_all", lambda picks: ([], ["a", "b"]))
        assert run_daily_job() == 1
        assert "notify" not in tracked["order"]
        assert any("FAILED to recommend" in n for n in tracked["notices"])

    def test_send_failed_exits_one(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "notify", lambda recs: SendStatus.FAILED)
        assert run_daily_job() == 1

    def test_send_skipped_disabled_channel_exits_zero(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "notify", lambda recs: SendStatus.SKIPPED)
        assert run_daily_job() == 0


class TestRecommendAll:
    def test_per_pick_isolation(self, monkeypatch):
        import src.serving.recommend as recommend

        monkeypatch.setattr(recommend, "load_resources", lambda: "RES")

        def flaky(pick, resources, game_date=None):
            if pick["home_team_id"] == 1:
                raise ValueError("not enough history")
            return {"ok": pick["home_team_id"]}

        monkeypatch.setattr(recommend, "recommend_pick", flaky)
        recs, errors = daily_job.recommend_all(
            [{"home_team_id": 1, "away_team_id": 2}, {"home_team_id": 3, "away_team_id": 4}]
        )
        assert recs == [{"ok": 3}]
        assert len(errors) == 1 and "1 vs 2" in errors[0]
