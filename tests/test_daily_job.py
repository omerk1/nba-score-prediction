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
    monkeypatch.setattr(daily_job, "SUCCESS_MARKER", tmp_path / "marker")
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
    calls["persisted_dates"] = []
    monkeypatch.setattr(
        daily_job,
        "persist",
        lambda recs, run_date: calls["order"].append("persist")
        or calls["persisted_dates"].append(run_date),
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

    def test_prune_skips_invalid_date_name_without_crashing(self, tmp_path):
        (tmp_path / "daily_job_2026-99-99.log").write_text("stray")
        assert prune_old_logs(tmp_path, retention_days=30, today=TODAY) == 0
        assert (tmp_path / "daily_job_2026-99-99.log").exists()

    def test_oversized_err_log_truncated_small_kept(self, tmp_path, monkeypatch):
        monkeypatch.setattr(daily_job, "ERR_LOG_MAX_BYTES", 100)
        big = tmp_path / "launchd_daily_job.err.log"
        big.write_text("x" * 200)
        prune_old_logs(tmp_path, today=TODAY)
        assert big.read_text() == ""
        big.write_text("small")
        prune_old_logs(tmp_path, today=TODAY)
        assert big.read_text() == "small"


class TestRunDailyJob:
    def test_happy_path_order_and_exit_zero(self, tracked):
        assert run_daily_job() == 0
        assert tracked["order"] == ["refresh", "capture", "extract", "recommend", "persist", "notify"]
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

    def test_persist_failure_still_sends_with_note_and_exits_zero(self, tracked, monkeypatch):
        def broken(recs, run_date):
            raise OSError("disk full")

        monkeypatch.setattr(daily_job, "persist", broken)
        assert run_daily_job() == 0
        assert tracked["order"][-1] == "notify"
        assert len(tracked["notices"]) == 1 and "NOT saved" in tracked["notices"][0]

    def test_send_failed_exits_one(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "notify", lambda recs: SendStatus.FAILED)
        assert run_daily_job() == 1

    def test_send_skipped_disabled_channel_exits_zero(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "notify", lambda recs: SendStatus.SKIPPED)
        assert run_daily_job() == 0

    def test_unexpected_crash_sends_notice_and_exits_one(self, tracked, monkeypatch):
        def boom():
            raise ImportError("catboost went missing")

        monkeypatch.setattr(daily_job, "refresh_data", boom)
        assert run_daily_job() == 1
        assert any("CRASHED" in n for n in tracked["notices"])

    def test_failed_notice_on_empty_slate_exits_one(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "extract", lambda p: [])
        monkeypatch.setattr(
            daily_job,
            "notice",
            lambda text: tracked["notices"].append(text) or SendStatus.FAILED,
        )
        assert run_daily_job() == 1

    def test_success_marker_blocks_same_day_rerun_unless_forced(self, tracked):
        assert run_daily_job() == 0
        assert (daily_job.SUCCESS_MARKER).exists()
        first_run = list(tracked["order"])
        assert run_daily_job() == 0  # marker: skipped quietly
        assert tracked["order"] == first_run
        assert run_daily_job(force=True) == 0  # --force re-runs
        assert len(tracked["order"]) == 2 * len(first_run)

    def test_failed_send_does_not_write_success_marker(self, tracked, monkeypatch):
        monkeypatch.setattr(daily_job, "notify", lambda recs: SendStatus.FAILED)
        assert run_daily_job() == 1
        assert not (daily_job.SUCCESS_MARKER).exists()


def test_split_pick_is_the_single_filter_home():
    from src.serving.recommend import RECOMMEND_KWARGS, split_pick

    pick = {"home_team_id": 1, "away_team_id": 2, "push_odds": 9.0}
    kwargs, dropped = split_pick(pick)
    assert set(kwargs) <= RECOMMEND_KWARGS
    assert dropped == {"push_odds": 9.0}
    assert set(kwargs) | set(dropped) == set(pick)


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


class TestScheduledWindowAndMarkerDate:
    @pytest.fixture
    def clock(self, tracked, monkeypatch):
        """Settable fake clock for daily_job._now."""
        now = {"t": datetime.datetime(2026, 10, 15, 16, 0, 5)}
        monkeypatch.setattr(daily_job, "_now", lambda: now["t"])
        return now

    def test_marker_records_start_date_for_run_straddling_midnight(
        self, tracked, clock, monkeypatch
    ):
        clock["t"] = datetime.datetime(2026, 10, 15, 23, 50)

        def slow_refresh():
            tracked["order"].append("refresh")
            clock["t"] = datetime.datetime(2026, 10, 16, 0, 30)
            return 0

        monkeypatch.setattr(daily_job, "refresh_data", slow_refresh)
        assert run_daily_job() == 0
        assert daily_job.SUCCESS_MARKER.read_text() == "2026-10-15"

        # next day's real run is not mistaken for already done
        monkeypatch.setattr(daily_job, "refresh_data", lambda: tracked["order"].append("refresh") or 0)
        clock["t"] = datetime.datetime(2026, 10, 16, 16, 0, 5)
        tracked["order"].clear()
        assert run_daily_job(scheduled=True) == 0
        assert tracked["order"][0] == "refresh"

    def test_scheduled_run_inside_window_runs_normally(self, tracked, clock):
        assert run_daily_job(scheduled=True) == 0
        assert tracked["order"][-1] == "notify"
        assert tracked["notices"] == []

    @pytest.mark.parametrize(
        "started, missed",
        [
            (datetime.datetime(2026, 10, 15, 23, 55), "2026-10-15"),
            (datetime.datetime(2026, 10, 16, 3, 15), "2026-10-15"),
        ],
    )
    def test_scheduled_run_starting_outside_window_reports_missed(
        self, tracked, clock, started, missed
    ):
        clock["t"] = started
        assert run_daily_job(scheduled=True) == 1
        assert tracked["order"] == []
        assert len(tracked["notices"]) == 1
        assert "MISSED" in tracked["notices"][0] and missed in tracked["notices"][0]
        assert not daily_job.SUCCESS_MARKER.exists()

    def test_manual_run_is_never_gated(self, tracked, clock):
        clock["t"] = datetime.datetime(2026, 10, 15, 23, 55)
        assert run_daily_job() == 0
        assert tracked["order"][-1] == "notify"

    def test_scheduled_refresh_finishing_past_deadline_reports_missed(
        self, tracked, clock, monkeypatch
    ):
        def slow_refresh():
            tracked["order"].append("refresh")
            clock["t"] = datetime.datetime(2026, 10, 16, 3, 57)
            return 0

        monkeypatch.setattr(daily_job, "refresh_data", slow_refresh)
        assert run_daily_job(scheduled=True) == 1
        assert tracked["order"] == ["refresh"]
        assert "MISSED" in tracked["notices"][0] and "2026-10-15" in tracked["notices"][0]

    def test_late_firing_after_successful_manual_run_is_quiet(self, tracked, clock):
        clock["t"] = datetime.datetime(2026, 10, 15, 17, 30)
        assert run_daily_job() == 0  # manual run delivers the slate
        tracked["order"].clear()
        for late in (datetime.datetime(2026, 10, 15, 23, 55), datetime.datetime(2026, 10, 16, 3, 0)):
            clock["t"] = late
            assert run_daily_job(scheduled=True) == 0
        assert tracked["order"] == []
        assert tracked["notices"] == []

    def test_manual_run_after_midnight_counts_toward_previous_slate(self, tracked, clock):
        clock["t"] = datetime.datetime(2026, 10, 16, 0, 20)
        assert run_daily_job() == 0
        assert daily_job.SUCCESS_MARKER.read_text() == "2026-10-15"
        assert tracked["persisted_dates"] == [datetime.date(2026, 10, 15)]

        clock["t"] = datetime.datetime(2026, 10, 16, 16, 0, 5)
        tracked["order"].clear()
        assert run_daily_job(scheduled=True) == 0
        assert tracked["order"][-1] == "notify"  # the 16th still runs

    def test_scheduled_stall_after_refresh_reports_missed_before_sending(
        self, tracked, clock, monkeypatch
    ):
        def stalled_capture():
            tracked["order"].append("capture")
            clock["t"] = datetime.datetime(2026, 10, 16, 3, 30)
            return Path("shot.png")

        monkeypatch.setattr(daily_job, "capture", stalled_capture)
        assert run_daily_job(scheduled=True) == 1
        assert "persist" not in tracked["order"] and "notify" not in tracked["order"]
        assert "MISSED" in tracked["notices"][0] and "2026-10-15" in tracked["notices"][0]
