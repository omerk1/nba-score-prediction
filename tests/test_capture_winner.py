import datetime
from contextlib import contextmanager

import pytest

from src.serving import capture_winner
from src.serving.capture_winner import (
    capture_nba_page,
    dated_capture_path,
    failure_path,
    prune_old_captures,
)

TODAY = datetime.date(2026, 10, 5)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(capture_winner.time, "sleep", lambda s: None)


class FakePage:
    """Page-like object: scripted failures via goto_errors/screenshot_errors
    (consumed per call), otherwise records the call sequence and writes a
    stub file on screenshot."""

    def __init__(self, goto_errors=None, screenshot_errors=None):
        self.goto_errors = list(goto_errors or [])
        self.screenshot_errors = list(screenshot_errors or [])
        self.calls = []

    def goto(self, url, timeout=None, wait_until=None):
        self.calls.append(("goto", url, wait_until))
        if self.goto_errors:
            raise self.goto_errors.pop(0)

    def evaluate(self, script):
        self.calls.append(("evaluate", script))

    def screenshot(self, path=None, full_page=None, timeout=None, animations=None):
        self.calls.append(("screenshot", path, full_page))
        if self.screenshot_errors:
            raise self.screenshot_errors.pop(0)
        with open(path, "wb") as f:
            f.write(b"png")


def _factory(pages):
    pages = list(pages)

    @contextmanager
    def factory():
        yield pages.pop(0)

    return factory


class TestPaths:
    def test_dated_capture_path(self, tmp_path):
        assert dated_capture_path(tmp_path, today=TODAY) == tmp_path / "winner_nba_2026-10-05.png"

    def test_failure_path_suffix(self, tmp_path):
        assert failure_path(tmp_path / "winner_nba_2026-10-05.png").name.endswith(".failed.png")


class TestPrune:
    def test_removes_only_expired_matching_files(self, tmp_path):
        (tmp_path / "winner_nba_2026-09-01.png").write_bytes(b"old")
        (tmp_path / "winner_nba_2026-10-01.png").write_bytes(b"recent")
        (tmp_path / "unrelated.png").write_bytes(b"keep")
        removed = prune_old_captures(tmp_path, retention_days=14, today=TODAY)
        assert removed == 1
        assert not (tmp_path / "winner_nba_2026-09-01.png").exists()
        assert (tmp_path / "winner_nba_2026-10-01.png").exists()
        assert (tmp_path / "unrelated.png").exists()

    def test_missing_dir_is_noop(self, tmp_path):
        assert prune_old_captures(tmp_path / "nope", today=TODAY) == 0


class TestCapture:
    def test_happy_path_custom_output_never_prunes_callers_dir(self, tmp_path):
        out = tmp_path / "winner_nba_2026-10-05.png"
        (tmp_path / "winner_nba_2020-01-01.png").write_bytes(b"archive")
        page = FakePage()
        result = capture_nba_page(output_path=out, url="http://x", page_factory=_factory([page]))
        assert result == out
        assert out.read_bytes() == b"png"
        assert page.calls[0] == ("goto", "http://x", "domcontentloaded")
        # a caller-owned directory must never see deletions
        assert (tmp_path / "winner_nba_2020-01-01.png").exists()

    def test_default_path_prunes_managed_dir(self, tmp_path, monkeypatch):
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "winner_nba_2020-01-01.png").write_bytes(b"ancient")
        (managed / "winner_nba_2020-01-02.failed.png").write_bytes(b"old diagnostic")
        monkeypatch.setattr(capture_winner, "DEFAULT_OUTPUT_DIR", managed)
        result = capture_nba_page(url="http://x", page_factory=_factory([FakePage()]))
        assert result.parent == managed
        assert not (managed / "winner_nba_2020-01-01.png").exists()
        assert not (managed / "winner_nba_2020-01-02.failed.png").exists()  # .failed pruned too

    def test_prune_failure_does_not_fail_a_successful_capture(self, tmp_path, monkeypatch):
        managed = tmp_path / "managed"
        monkeypatch.setattr(capture_winner, "DEFAULT_OUTPUT_DIR", managed)

        def broken_prune(*a, **kw):
            raise PermissionError("undeletable")

        monkeypatch.setattr(capture_winner, "prune_old_captures", broken_prune)
        pages = [FakePage()]
        result = capture_nba_page(url="http://x", page_factory=_factory(pages))
        assert result.exists()
        gotos = [c for c in pages[0].calls if c[0] == "goto"]
        assert len(gotos) == 1  # no retry happened

    def test_failed_goto_writes_best_effort_screenshot_then_retries(self, tmp_path):
        out = tmp_path / "winner_nba_2026-10-05.png"
        first = FakePage(goto_errors=[TimeoutError("nav timeout")])
        second = FakePage()
        result = capture_nba_page(
            output_path=out, url="http://x", page_factory=_factory([first, second])
        )
        assert result == out
        # best-effort partial render: viewport-only, written by the failed
        # attempt, then removed once the retry succeeded (stale diagnostic)
        assert ("screenshot", str(failure_path(out)), False) in first.calls
        assert not failure_path(out).exists()

    def test_two_failures_raise_runtime_error(self, tmp_path):
        out = tmp_path / "winner_nba_2026-10-05.png"
        pages = [
            FakePage(goto_errors=[TimeoutError("t1")], screenshot_errors=[OSError("no page")]),
            FakePage(goto_errors=[TimeoutError("t2")], screenshot_errors=[OSError("no page")]),
        ]
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            capture_nba_page(output_path=out, url="http://x", page_factory=_factory(pages))

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "winner_nba_2026-10-05.png"
        capture_nba_page(output_path=out, url="http://x", page_factory=_factory([FakePage()]))
        assert out.exists()
