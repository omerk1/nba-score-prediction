"""
Regression test for the multi-archetype injury-delta accumulation bug: when
2+ different-archetype players are out for the same team on the same date and
their injury_impact deltas share an affected metric, build_injury_adjusted_fingerprints
used to recompute each archetype's delta from the ORIGINAL layer=1 value
instead of the running adjusted one -- so only the last-iterated archetype's
contribution to a shared metric survived, silently dropping the others,
instead of summing (the documented, intended behavior -- see
injury_layer.py's module docstring: "Different archetypes missing
simultaneously DO stack additively"). Found by session rs_20260808_1's D2
leakage audit (see EXPERIMENTS.md); not leakage itself (no future
information involved), a same-date accumulation bug.
"""

import pytest

from src.matchups.injury_layer import _apply_injury_deltas


class TestInjuryDeltaAccumulation:

    def test_single_archetype_applies_its_delta(self):
        layer1 = {"pace_score": 100.0, "defensive_rating": 110.0, "paint_activity": 20.0}
        injury_impact = {"rim_protector": {"defensive_rating": 0.4281, "paint_activity": -0.2729}}
        out = _apply_injury_deltas(layer1, {"rim_protector": 0.6}, injury_impact)
        assert out["defensive_rating"] == 110.0 + 0.4281 * 0.6
        assert out["paint_activity"] == 20.0 + (-0.2729) * 0.6
        assert out["pace_score"] == 100.0  # untouched metric passes through unchanged

    def test_two_archetypes_sharing_a_metric_sum_not_overwrite(self):
        """The exact failure mode this regression test locks in: before the
        fix, only `combo`'s contribution to defensive_rating/paint_activity
        would have survived (the last-iterated archetype in dict order),
        silently dropping rim_protector's."""
        layer1 = {
            "pace_score": 221.5046,
            "three_pt_reliance": 0.3397,
            "paint_activity": 24.3822,
            "defensive_rating": 115.4307,
            "assist_rate": 0.5523,
            "offensive_rating": 101.9681,
        }
        injury_impact = {
            "rim_protector": {"defensive_rating": 0.4281, "paint_activity": -0.2729},
            "combo": {
                "pace_score": -0.4059,
                "three_pt_reliance": -0.0026,
                "paint_activity": 0.0367,
                "defensive_rating": 0.5902,
                "assist_rate": -0.0048,
            },
        }
        archetypes_out = {"rim_protector": 0.6, "combo": 0.3}
        out = _apply_injury_deltas(layer1, archetypes_out, injury_impact)

        # Both archetypes' contributions to the shared metrics must be summed.
        expected_defensive_rating = 115.4307 + 0.4281 * 0.6 + 0.5902 * 0.3
        expected_paint_activity = 24.3822 + (-0.2729) * 0.6 + 0.0367 * 0.3
        assert out["defensive_rating"] == pytest.approx(expected_defensive_rating)
        assert out["paint_activity"] == pytest.approx(expected_paint_activity)

        # combo-only metrics: just combo's own delta.
        expected_pace_score = 221.5046 + (-0.4059) * 0.3
        assert out["pace_score"] == pytest.approx(expected_pace_score)

        # offensive_rating: no archetype's injury_impact touches it -- passes through.
        assert out["offensive_rating"] == 101.9681

    def test_no_archetypes_out_is_a_no_op(self):
        layer1 = {"pace_score": 100.0, "defensive_rating": 110.0}
        out = _apply_injury_deltas(layer1, {}, {"rim_protector": {"defensive_rating": 0.4281}})
        assert out == layer1
