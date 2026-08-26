"""Tests for the Phase-1 Blue-team ML detector: leakage gate, causal matrix, determinism,
budget threshold, end-to-end train/score/summarise, and joblib round-trip."""

import tempfile
import unittest

import numpy as np

from red_team_agent.planner import RedTeamAgent

from sentinelloop.blue_agent import GenAIBlueAgent
from sentinelloop.blue_ml.detector import FraudDetector
from sentinelloop.blue_ml.feature_frame import build_feature_frame_from_seed
from sentinelloop.blue_ml.features import FEATURES, FORBIDDEN_MODEL_INPUTS, audit_leakage
from sentinelloop.blue_ml.metrics import summarise, threshold_for_budget
from sentinelloop.config import AgentLabConfig
from sentinelloop.simulation import simulate_attack


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.groups, cls.meta = build_feature_frame_from_seed(42, seeds_per_cell=1)

    def test_allowlist_clean(self):
        self.assertEqual(audit_leakage(), [])

    def test_matrix_matches_allowlist_and_no_forbidden(self):
        self.assertEqual(list(self.X.columns), FEATURES)
        self.assertFalse(set(self.X.columns) & FORBIDDEN_MODEL_INPUTS)
        self.assertEqual(len(self.X), len(self.y))
        self.assertGreater(int(self.y.sum()), 0)

    def test_threshold_for_budget_picks_kth_largest(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
        self.assertEqual(threshold_for_budget(scores, 2), 0.7)  # top-2 cutoff
        self.assertEqual(threshold_for_budget(scores, 10), 0.0)  # budget >= n

    def test_determinism(self):
        d1 = FraudDetector().fit(self.X, self.y)
        d2 = FraudDetector().fit(self.X, self.y)
        np.testing.assert_allclose(d1.score(self.X), d2.score(self.X))

    def test_train_score_summarise_and_roundtrip(self):
        detector = FraudDetector().fit(self.X, self.y)
        scores = detector.score(self.X)
        self.assertEqual(len(scores), len(self.X))
        thr = threshold_for_budget(scores, max(1, round(0.01 * len(scores))))
        report = summarise(self.meta, self.y, scores, thr)
        self.assertGreaterEqual(report["chain_recall"], 0.0)
        self.assertLessEqual(report["chain_recall"], 1.0)
        self.assertIn("hard_false_positive_rate", report)

        with tempfile.TemporaryDirectory() as tmp:
            detector.threshold = thr
            detector.save(tmp)
            reloaded = FraudDetector.load(tmp)
            np.testing.assert_allclose(reloaded.score(self.X), scores)
            self.assertEqual(reloaded.threshold, thr)

            blue = GenAIBlueAgent(
                gateway=object(),
                config=AgentLabConfig(ml_detector_enabled=True, ml_model_dir=tmp),
            )
            event = simulate_attack(
                RedTeamAgent().plan(attack_family="ATO-01", difficulty="medium", seed=13)
            ).events[0]
            packet, facts = blue._ml_evidence(event, [event], [])
            self.assertEqual(packet.tool_name, "ml_risk_score")
            self.assertIn("cumulative_session_risk", facts)
            self.assertNotIn("attack_family", packet.to_dict())


if __name__ == "__main__":
    unittest.main()
