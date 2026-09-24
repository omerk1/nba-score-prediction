"""
Tests for semantic retrieval over reason text. The property that matters and
would break silently is the point-in-time bound: a neighbour may only
contribute if its report predates the query's report.
"""

import numpy as np
import pandas as pd
import pytest

from src.availability.reason_retrieval import (
    FEATURE_COLUMNS,
    SELF_FEATURE_COLUMNS,
    ReasonIndex,
    RetrievalParams,
    build_reason_features,
    build_self_reason_features,
)
from src.availability.retrieval import normalize_reason


@pytest.fixture
def index():
    """Three reasons: two near-identical ankle sprains and one unrelated."""
    v = {
        normalize_reason("Right Ankle; Sprain"): np.array([1.0, 0.0, 0.0]),
        normalize_reason("Left Ankle; Sprain"): np.array([0.99, 0.141, 0.0]),
        normalize_reason("Personal Reasons"): np.array([0.0, 0.0, 1.0]),
    }
    return ReasonIndex(v)


def _hist(rows):
    return pd.DataFrame(rows, columns=["report_date", "player_id", "reason", "status", "played"])


class TestIndex:
    def test_similar_reasons_score_high_and_unrelated_low(self, index):
        i = index.pos[normalize_reason("Right Ankle; Sprain")]
        j = index.pos[normalize_reason("Left Ankle; Sprain")]
        k = index.pos[normalize_reason("Personal Reasons")]
        assert index.sim[i, j] > 0.98
        assert index.sim[i, k] < 0.05

    def test_unknown_reason_maps_to_minus_one(self, index):
        assert index.ids(pd.Series(["Something Unseen"]))[0] == -1


class TestPointInTime:
    def test_only_earlier_reports_contribute(self, index):
        history = _hist(
            [
                ("2024-01-01", 1, "Left Ankle; Sprain", "Questionable", 1),
                ("2024-01-02", 2, "Left Ankle; Sprain", "Questionable", 1),
                # Same day as the query's report: must NOT contribute.
                ("2024-01-10", 3, "Left Ankle; Sprain", "Questionable", 0),
                # After the query: must NOT contribute.
                ("2024-02-01", 4, "Left Ankle; Sprain", "Questionable", 0),
            ]
        )
        q = pd.DataFrame([{"report_date": "2024-01-10", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        p = RetrievalParams(k=0, min_sim=0.5, power=1.0, prior_weight=0.0)
        f = build_reason_features(q, history, index, p)
        # Two prior listings, both played, so the rate is 1.0. Including either
        # later row would pull it below 1.
        assert f["reason_nbr_play_rate"].iat[0] == pytest.approx(1.0)
        assert f["reason_nbr_support"].iat[0] == pytest.approx(2.0, abs=0.1)

    def test_no_prior_history_yields_missing_not_zero(self, index):
        history = _hist([("2024-02-01", 1, "Left Ankle; Sprain", "Questionable", 1)])
        q = pd.DataFrame([{"report_date": "2024-01-01", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        f = build_reason_features(q, history, index, RetrievalParams(k=0, min_sim=0.5, prior_weight=0.0))
        assert np.isnan(f["reason_nbr_play_rate"].iat[0])

    def test_dissimilar_reasons_are_excluded_by_the_floor(self, index):
        history = _hist([("2024-01-01", 1, "Personal Reasons", "Questionable", 1)])
        q = pd.DataFrame([{"report_date": "2024-01-05", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        f = build_reason_features(q, history, index, RetrievalParams(k=0, min_sim=0.5, prior_weight=0.0))
        assert np.isnan(f["reason_nbr_play_rate"].iat[0])

    def test_out_share_uses_all_statuses(self, index):
        history = _hist(
            [
                ("2024-01-01", 1, "Left Ankle; Sprain", "Out", 0),
                ("2024-01-02", 2, "Left Ankle; Sprain", "Questionable", 1),
            ]
        )
        q = pd.DataFrame([{"report_date": "2024-01-10", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        f = build_reason_features(q, history, index, RetrievalParams(k=0, min_sim=0.5, prior_weight=0.0))
        assert f["reason_nbr_out_share"].iat[0] == pytest.approx(0.5, abs=0.02)
        # Only the uncertain row counts toward the play rate.
        assert f["reason_nbr_play_rate"].iat[0] == pytest.approx(1.0)


class TestSelfVariant:
    def test_only_the_same_player_contributes(self, index):
        history = _hist(
            [
                ("2024-01-01", 9, "Left Ankle; Sprain", "Questionable", 1),
                ("2024-01-02", 7, "Left Ankle; Sprain", "Questionable", 0),  # other player
            ]
        )
        q = pd.DataFrame([{"report_date": "2024-01-10", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        f = build_self_reason_features(q, history, index, RetrievalParams(k=0, min_sim=0.5, power=1.0))
        assert f["self_reason_nbr_play_rate"].iat[0] == pytest.approx(1.0)

    def test_point_in_time_holds_for_the_self_variant(self, index):
        history = _hist([("2024-02-01", 9, "Left Ankle; Sprain", "Questionable", 0)])
        q = pd.DataFrame([{"report_date": "2024-01-10", "player_id": 9, "reason": "Right Ankle; Sprain"}])
        f = build_self_reason_features(q, history, index, RetrievalParams(k=0, min_sim=0.5, power=1.0))
        assert np.isnan(f["self_reason_nbr_play_rate"].iat[0])

    def test_columns_are_distinct_between_variants(self):
        assert not set(FEATURE_COLUMNS) & set(SELF_FEATURE_COLUMNS)
