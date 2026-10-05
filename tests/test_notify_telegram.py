from types import SimpleNamespace

import pytest
import requests

from src.serving import notify_telegram
from src.serving.notify_telegram import (
    TELEGRAM_MESSAGE_LIMIT,
    format_game_block,
    format_telegram_message,
    send_recommendations,
    send_telegram,
)
from src.utils.config_loader import load_config

TEAM_NAMES = {1: "Lakers", 2: "Celtics"}

FULL_REC = {
    "home_team_id": 1,
    "away_team_id": 2,
    "predicted_home_score": 114.2,
    "predicted_away_score": 108.9,
    "predicted_diff": 5.3,
    "predicted_total": 223.1,
    "home_win_probability": 0.682,
    "home_moneyline_market_probability": 0.600,
    "home_moneyline_edge": 0.082,
    "away_win_probability": 0.318,
    "away_moneyline_market_probability": 0.420,
    "away_moneyline_edge": -0.102,
    "home_spread": -4.5,
    "home_cover_probability": 0.553,
    "away_cover_probability": 0.447,
    "home_spread_market_probability": 0.526,
    "home_spread_edge": 0.027,
    "away_spread_market_probability": 0.512,
    "away_spread_edge": -0.065,
    "total_line": 224.5,
    "over_probability": 0.481,
    "under_probability": 0.519,
    "over_market_probability": 0.513,
    "over_edge": -0.032,
    "under_market_probability": 0.515,
    "under_edge": 0.004,
}

MINIMAL_REC = {
    "home_team_id": 1,
    "away_team_id": 2,
    "predicted_home_score": 110.0,
    "predicted_away_score": 105.0,
    "predicted_diff": 5.0,
    "predicted_total": 215.0,
    "home_win_probability": 0.64,
}


class FakeTransport:
    """Records posts; per-call behavior scripted via `outcomes` (a
    status code, or an exception instance to raise)."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(status_code=outcome, text=f"status {outcome}")


class TestFormatting:
    def test_full_rec_renders_every_market(self):
        block = format_game_block(FULL_REC, TEAM_NAMES)
        assert "<b>Lakers vs Celtics</b>" in block
        assert "diff +5.3, total 223.1" in block
        assert "Home win: 68.2%" in block
        assert "ML home: 68.2% | mkt 60.0% | edge +8.2%" in block
        assert "Spread -4.5 home cover: 55.3% | mkt 52.6% | edge +2.7%" in block
        assert "Total 224.5 over: 48.1% | mkt 51.3% | edge -3.2%" in block
        assert "Under: 51.9% | mkt 51.5% | edge +0.4%" in block

    def test_positive_edge_lines_are_bolded(self):
        block = format_game_block(FULL_REC, TEAM_NAMES)
        assert "<b>ML home: 68.2%" in block  # +8.2% edge
        assert "<b>ML away" not in block  # -10.2% edge

    def test_minimal_rec_has_no_market_lines(self):
        block = format_game_block(MINIMAL_REC, TEAM_NAMES)
        assert "Home win: 64.0%" in block
        assert "mkt" not in block
        assert "Spread" not in block
        assert "Total" not in block

    def test_team_names_html_escaped_and_unknown_id_falls_back(self):
        rec = dict(MINIMAL_REC, home_team_id=7)
        block = format_game_block(rec, {7: "A&M <Spurs>", 2: "Celtics"})
        assert "A&amp;M &lt;Spurs&gt; vs Celtics" in block

    def test_spread_probability_shown_even_without_odds(self):
        rec = dict(MINIMAL_REC, home_spread=-3.0, home_cover_probability=0.52,
                   away_cover_probability=0.48)
        block = format_game_block(rec, TEAM_NAMES)
        assert "Spread -3 home cover: 52.0%" in block
        assert "mkt" not in block


class TestChunking:
    @pytest.fixture(autouse=True)
    def static_team_names(self, monkeypatch):
        monkeypatch.setattr(notify_telegram, "build_team_id_to_name", lambda: TEAM_NAMES)

    def test_empty_recs_no_messages(self):
        assert format_telegram_message([]) == []

    def test_small_slate_fits_one_message(self):
        messages = format_telegram_message([FULL_REC, MINIMAL_REC])
        assert len(messages) == 1
        assert messages[0].count("<b>Lakers vs Celtics</b>") == 2

    def test_large_slate_chunks_whole_blocks(self):
        messages = format_telegram_message([FULL_REC] * 30)
        assert len(messages) > 1
        block = format_game_block(FULL_REC, TEAM_NAMES)
        total_blocks = 0
        for msg in messages:
            assert len(msg) <= TELEGRAM_MESSAGE_LIMIT
            # whole blocks only — every chunk splits back into intact copies
            parts = msg.split("\n\n")
            assert all(p == block for p in parts)
            total_blocks += len(parts)
        assert total_blocks == 30

    def test_oversize_single_block_truncated(self, monkeypatch):
        huge = dict(MINIMAL_REC)
        monkeypatch.setattr(
            notify_telegram, "format_game_block", lambda rec, names: "x" * 5000
        )
        messages = format_telegram_message([huge])
        assert len(messages) == 1
        assert len(messages[0]) <= TELEGRAM_MESSAGE_LIMIT
        assert messages[0].endswith("…truncated")


class TestSendTelegram:
    def test_posts_each_message_with_payload(self):
        transport = FakeTransport()
        ok = send_telegram(["msg1", "msg2"], token="T", chat_id="C", transport=transport)
        assert ok
        assert len(transport.posts) == 2
        assert transport.posts[0]["url"] == "https://api.telegram.org/botT/sendMessage"
        assert transport.posts[0]["json"] == {"chat_id": "C", "text": "msg1", "parse_mode": "HTML"}
        assert transport.posts[0]["timeout"] == 10

    def test_non_200_logged_and_continues(self, caplog):
        transport = FakeTransport(outcomes=[400, 200])
        ok = send_telegram(["bad", "good"], token="T", chat_id="C", transport=transport)
        assert not ok
        assert len(transport.posts) == 2  # second message still sent
        assert "400" in caplog.text

    def test_transport_exception_swallowed(self):
        transport = FakeTransport(outcomes=[ValueError("boom"), 200])
        ok = send_telegram(["a", "b"], token="T", chat_id="C", transport=transport)
        assert not ok
        assert len(transport.posts) == 2

    def test_timeout_retried_once_then_succeeds(self):
        transport = FakeTransport(outcomes=[requests.exceptions.Timeout(), 200])
        ok = send_telegram(["a"], token="T", chat_id="C", transport=transport)
        assert ok
        assert len(transport.posts) == 2

    def test_timeout_twice_gives_up(self):
        transport = FakeTransport(
            outcomes=[requests.exceptions.Timeout(), requests.exceptions.Timeout()]
        )
        ok = send_telegram(["a"], token="T", chat_id="C", transport=transport)
        assert not ok
        assert len(transport.posts) == 2


class TestSendRecommendations:
    def test_repo_config_default_is_disabled_noop(self, monkeypatch):
        # guards the ship-disabled convention: the checked-in config must
        # not send (and therefore needs no secrets)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        config = load_config()
        assert config.notifications.telegram.enabled is False
        assert send_recommendations([MINIMAL_REC], config=config) is False

    def _enabled_config(self):
        return SimpleNamespace(
            notifications=SimpleNamespace(
                telegram=SimpleNamespace(enabled=True, parse_mode="HTML", timeout_seconds=10)
            )
        )

    def test_missing_secrets_logs_and_returns_false(self, monkeypatch, caplog):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert send_recommendations([MINIMAL_REC], config=self._enabled_config()) is False
        assert "TELEGRAM_BOT_TOKEN" in caplog.text

    def test_enabled_with_secrets_sends(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
        monkeypatch.setattr(notify_telegram, "build_team_id_to_name", lambda: TEAM_NAMES)
        sent = []
        monkeypatch.setattr(
            notify_telegram,
            "send_telegram",
            lambda messages, **kw: sent.append((messages, kw)) or True,
        )
        assert send_recommendations([MINIMAL_REC], config=self._enabled_config()) is True
        (messages, kwargs) = sent[0]
        assert "Lakers vs Celtics" in messages[0]
        assert kwargs["token"] == "T" and kwargs["chat_id"] == "C"
