"""
Regression test for the FINGERPRINT_METRICS drift bug: matchup_index.py used to
define its own local FINGERPRINT_METRICS (5 metrics) instead of importing
fingerprint.py's (6, since the raw-fingerprint feature redesign added
offensive_rating) -- every other consumer (injury_layer.py, calibration.py,
tuning.py) already imported the shared constant; matchup_index.py was the one
file that drifted out of sync, silently excluding offensive_rating from the
Layer-3 KNN/cosine similarity vector.
"""

from src.matchups.fingerprint import FINGERPRINT_METRICS as fingerprint_metrics
from src.matchups.matchup_index import FINGERPRINT_METRICS as matchup_index_metrics


class TestFingerprintMetricsInSync:

    def test_matchup_index_imports_the_shared_constant(self):
        """matchup_index.py must use fingerprint.py's FINGERPRINT_METRICS directly
        (same object), not a locally-redefined copy that can silently drift --
        the exact failure mode this regression test locks in."""
        assert matchup_index_metrics is fingerprint_metrics

    def test_six_metrics_including_offensive_rating(self):
        """Locks in the specific count/content that was wrong before the fix
        (5 metrics, missing offensive_rating) -- not just object identity."""
        assert len(matchup_index_metrics) == 6
        assert "offensive_rating" in matchup_index_metrics
