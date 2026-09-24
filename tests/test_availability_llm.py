"""
LLM estimator tests: the anonymized prompt carries no identity or date, no
label ever reaches a prompt, and the cache prevents repeat calls.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from src.availability import llm_estimator as LE
from src.availability.prompt import render_prompt
from src.availability.retrieval import CONTEXT_COLUMNS


def _row(**over) -> pd.Series:
    base = {c: 0.0 for c in CONTEXT_COLUMNS}
    base.update(
        {
            "report_date": "2024-01-11",
            "game_date": "2024-01-12",
            "season": "2023-24",
            "team_id": 1610612738,
            "game_id": "g2",
            "player_id": 7,
            "player_name": "Jayson Tatum",
            "status": "Questionable",
            "reason": "Injury/Illness - Left Ankle; Sprain",
            "minutes": 30.0,
            "played": 1,
            "own_n_prior_uncertain": 4,
            "own_prior_play_rate": 0.5,
            "own_n_same_reason": 1,
            "own_same_reason_play_rate": 1.0,
            "min_last10_mean": 33.2,
            "min_last_game": 35.0,
            "days_since_last_played": 2,
            "imp_minutes_per_game": 34.0,
            "imp_pts_per_game": 27.0,
            "imp_usage_rate": 0.3,
            "is_home": 1,
            "rest_days": 2,
            "team_win_pct_std": 0.7,
            "opp_win_pct_std": 0.4,
            "days_into_season": 80,
        }
    )
    base.update(over)
    return pd.Series(base)


class TestPrompt:
    def test_anonymized_has_no_identity_or_date_or_label(self):
        text = render_prompt(_row())
        assert "Tatum" not in text and "BOS" not in text
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
        # The label and the outcome minutes never influence the prompt.
        assert render_prompt(_row(played=0, minutes=0.0)) == text
        assert 'status=Questionable; reason="Injury/Illness - Left Ankle; Sprain"' in text
        assert "33.2 min avg" in text

    def test_named_variant_adds_identity(self):
        text = render_prompt(_row(), "named")
        assert "Jayson Tatum" in text and "BOS" in text and "2024-01-12" in text

    def test_missing_values_render_as_unknown(self):
        text = render_prompt(_row(min_last10_mean=np.nan, imp_usage_rate=np.nan))
        assert "no games played yet this season" in text
        assert "usage unknown" in text


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps({"p_play": 0.42, "expected_minutes_share": 0.8, "rationale": "test"})


class _BadClient:
    def complete(self, prompt: str) -> str:
        return "not json"


@pytest.fixture
def est(tmp_path, monkeypatch):
    monkeypatch.setattr(LE, "_RETRY_BASE_DELAY", 0)
    fake = _FakeClient()
    e = LE.LLMEstimator(client=fake, db_path=str(tmp_path / "a.sqlite"), parallel_workers=2, model="fake")
    return e, fake


class TestPromptKeyCollision:
    """Regression test for a real bug (caught 2026-09-23): prompt_key hashed
    only model + prompt text, so a zero-shot prompt with thinking enabled was
    byte-identical to the same prompt with thinking disabled, and a
    "reasoning-enabled" run silently replayed the reasoning-disabled cache
    instead of making new calls. thinking_budget must be part of the key."""

    def test_same_prompt_different_thinking_budget_are_not_the_same_key(self):
        k0 = LE.prompt_key("m", "same text", thinking_budget=0)
        k1 = LE.prompt_key("m", "same text", thinking_budget=2048)
        assert k0 != k1

    def test_estimator_makes_new_calls_when_only_thinking_budget_differs(self, tmp_path):
        db = str(tmp_path / "a.sqlite")
        fake = _FakeClient()
        df = pd.DataFrame([_row()])  # n_shots=0 on both -> identical prompt text
        LE.LLMEstimator(client=fake, db_path=db, parallel_workers=1, model="fake", thinking_budget=0).predict(
            df
        )
        assert fake.calls == 1
        # Same prompt text, different reasoning budget: must NOT be served from
        # the thinking_budget=0 cache entry above.
        LE.LLMEstimator(
            client=fake, db_path=db, parallel_workers=1, model="fake", thinking_budget=2048
        ).predict(df)
        assert fake.calls == 2
        # Genuinely identical config: this one SHOULD be cached.
        LE.LLMEstimator(
            client=fake, db_path=db, parallel_workers=1, model="fake", thinking_budget=2048
        ).predict(df)
        assert fake.calls == 2


class TestEstimator:
    def test_predict_and_cache(self, est, tmp_path):
        e, fake = est
        df = pd.DataFrame([_row(), _row(reason="Illness")])
        p = e.predict(df)
        assert list(p) == [0.42, 0.42]
        assert fake.calls == 2
        # second run: fully served from cache, even with a fresh estimator on the same db
        e2 = LE.LLMEstimator(
            client=fake, db_path=str(tmp_path / "a.sqlite"), parallel_workers=2, model="fake"
        )
        assert list(e2.predict(df)) == [0.42, 0.42]
        assert fake.calls == 2
        assert e2.last_details["expected_minutes_share"].tolist() == [0.8, 0.8]

    def test_failure_returns_nan(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LE, "_RETRY_BASE_DELAY", 0)
        e = LE.LLMEstimator(
            client=_BadClient(), db_path=str(tmp_path / "b.sqlite"), parallel_workers=1, model="fake"
        )
        p = e.predict(pd.DataFrame([_row()]))
        assert np.isnan(p[0]) and e.n_failed == 1

    def test_parse_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            LE._parse(json.dumps({"p_play": 1.7}))
