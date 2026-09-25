"""
extract_picks_from_screenshot's parsing/validation logic -- team-nickname
resolution, dropping unrecognized teams, malformed-JSON handling -- tested
with a fake Gemini client (same "swap a fake with matching interface" pattern
as src/availability/llm_estimator.py's GeminiClient), never a real network
call. This does NOT test whether Gemini can actually read a screenshot
correctly; see docs/features/serving/scope.md for that gap.

tests/fixtures/screenshot_spread_example.png and screenshot_total_example.png
are real bookmaker screenshots (Hebrew, real NBA games) saved for this and
for future live re-verification once GOOGLE_API_KEY has credits. The canned
JSON below is what a correct reading of those specific images should look
like (established by manually reading them), used here purely as example
model output for testing the parsing code -- not a live check that Gemini
actually produces it.
"""

import json

import pytest

from src.serving import extract_picks as EP
from src.serving.extract_picks import _swap_home_away, extract_picks_from_screenshot
from src.serving.team_lookup import build_nickname_to_team_id

FIXTURES = "tests/fixtures"


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_call = None

    def generate_content(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text: str):
        self.models = _FakeModels(response_text)


@pytest.fixture(autouse=True)
def no_schedule_lookup(monkeypatch):
    """extract_picks_from_screenshot always calls resolve_home_away (a real
    nba_api network call) -- patched to "no match found" by default so
    every test above stays a fast, offline, pure-parsing test unless a test
    explicitly overrides this to exercise the override/swap path below."""
    monkeypatch.setattr(EP, "resolve_home_away", lambda *a, **k: None)


class TestExtractPicksFromScreenshot:
    def test_spread_example_resolves_real_team_ids(self):
        nick2id = build_nickname_to_team_id()
        raw = [
            {
                "home_team_nickname": "Heat",
                "away_team_nickname": "Celtics",
                "home_spread": -1.5,
                "away_spread": 1.5,
                "home_spread_odds": 1.80,
                "away_spread_odds": 1.80,
            },
            {
                "home_team_nickname": "Mavericks",
                "away_team_nickname": "Warriors",
                "home_spread": -11.5,
                "away_spread": 11.5,
                "home_spread_odds": 1.80,
                "away_spread_odds": 1.80,
            },
        ]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        assert len(picks) == 2
        assert picks[0]["home_team_id"] == nick2id["Heat"]
        assert picks[0]["away_team_id"] == nick2id["Celtics"]
        assert picks[0]["home_spread"] == -1.5
        assert picks[0]["home_spread_odds"] == 1.80
        assert picks[1]["home_team_id"] == nick2id["Mavericks"]
        assert picks[1]["away_team_id"] == nick2id["Warriors"]
        # nickname fields are consumed, not passed through
        assert "home_team_nickname" not in picks[0]

    def test_total_example_passes_through_total_fields(self):
        nick2id = build_nickname_to_team_id()
        raw = [
            {
                "home_team_nickname": "Heat",
                "away_team_nickname": "Celtics",
                "total_line": 169.5,
                "over_odds": 1.70,
                "under_odds": 1.80,
            }
        ]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_total_example.png", client=client)

        assert len(picks) == 1
        assert picks[0]["home_team_id"] == nick2id["Heat"]
        assert picks[0]["total_line"] == 169.5
        assert picks[0]["over_odds"] == 1.70
        assert picks[0]["under_odds"] == 1.80

    def test_unrecognized_team_is_dropped_not_guessed(self):
        raw = [
            {"home_team_nickname": "Heat", "away_team_nickname": "NotARealTeam"},
            {"home_team_nickname": "Warriors", "away_team_nickname": "Celtics"},
        ]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        assert len(picks) == 1
        assert picks[0]["away_team_id"] == build_nickname_to_team_id()["Celtics"]

    def test_no_nba_games_returns_empty_list(self):
        client = _FakeClient("[]")
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)
        assert picks == []

    def test_malformed_json_raises(self):
        client = _FakeClient("not json")
        with pytest.raises(ValueError, match="did not return valid JSON"):
            extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

    def test_request_includes_full_real_nba_nickname_vocabulary(self):
        """The prompt must constrain the model to real nicknames -- a
        regression here (e.g. an empty or truncated list) would silently
        let the model guess team names instead of being bound to reality."""
        nick2id = build_nickname_to_team_id()
        client = _FakeClient("[]")
        extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        system_instruction = client.models.last_call["config"].system_instruction
        for nickname in nick2id:
            assert nickname in system_instruction


class TestScheduleOverride:
    """Vision-based home/away proved unreliable in practice (flip-flopped
    across repeated real calls on the same image) -- these test the
    schedule-lookup override that replaced trusting the model's guess."""

    def test_schedule_disagreement_swaps_every_paired_field(self, monkeypatch):
        nick2id = build_nickname_to_team_id()
        heat, celtics = nick2id["Heat"], nick2id["Celtics"]
        # Model guessed Heat home / Celtics away; the "real" schedule below
        # says the opposite -- every home_*/away_* field must flip together.
        monkeypatch.setattr(EP, "resolve_home_away", lambda a, b, d: (celtics, heat))
        raw = [
            {
                "home_team_nickname": "Heat",
                "away_team_nickname": "Celtics",
                "home_spread": -1.5,
                "away_spread": 1.5,
                "home_spread_odds": 1.80,
                "away_spread_odds": 1.90,
            }
        ]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        assert picks[0]["home_team_id"] == celtics
        assert picks[0]["away_team_id"] == heat
        assert picks[0]["home_spread"] == 1.5
        assert picks[0]["away_spread"] == -1.5
        assert picks[0]["home_spread_odds"] == 1.90
        assert picks[0]["away_spread_odds"] == 1.80

    def test_schedule_agreement_leaves_pick_unchanged(self, monkeypatch):
        nick2id = build_nickname_to_team_id()
        heat, celtics = nick2id["Heat"], nick2id["Celtics"]
        monkeypatch.setattr(EP, "resolve_home_away", lambda a, b, d: (heat, celtics))
        raw = [{"home_team_nickname": "Heat", "away_team_nickname": "Celtics", "home_spread": -1.5}]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        assert picks[0]["home_team_id"] == heat
        assert picks[0]["home_spread"] == -1.5

    def test_no_scheduled_match_keeps_models_guess(self, monkeypatch):
        nick2id = build_nickname_to_team_id()
        monkeypatch.setattr(EP, "resolve_home_away", lambda a, b, d: None)
        raw = [{"home_team_nickname": "Heat", "away_team_nickname": "Celtics"}]
        client = _FakeClient(json.dumps(raw))
        picks = extract_picks_from_screenshot(f"{FIXTURES}/screenshot_spread_example.png", client=client)

        assert picks[0]["home_team_id"] == nick2id["Heat"]


class TestSwapHomeAway:
    def test_swaps_paired_fields_present_on_both_sides(self):
        pick = {"home_team_id": 1, "away_team_id": 2, "home_spread": -1.5, "away_spread": 1.5}
        swapped = _swap_home_away(pick)
        assert swapped == {"home_team_id": 2, "away_team_id": 1, "home_spread": 1.5, "away_spread": -1.5}

    def test_field_present_on_only_one_side_moves_to_the_other_and_is_removed(self):
        # Only home_moneyline_odds was given (away side omitted by the model) --
        # after swapping, it must become away_moneyline_odds, and home_moneyline_odds
        # must be absent entirely, not present as None (that would falsely look
        # like "the model saw this field but it was null").
        pick = {"home_team_id": 1, "away_team_id": 2, "home_moneyline_odds": 1.55}
        swapped = _swap_home_away(pick)
        assert swapped["away_moneyline_odds"] == 1.55
        assert "home_moneyline_odds" not in swapped

    def test_unrelated_fields_untouched(self):
        pick = {"home_team_id": 1, "away_team_id": 2, "total_line": 220.5, "push_odds": 9.0}
        swapped = _swap_home_away(pick)
        assert swapped["total_line"] == 220.5
        assert swapped["push_odds"] == 9.0
