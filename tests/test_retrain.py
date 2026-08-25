"""Phase-3 tests: training-log schema, champion/challenger gate logic, and an end-to-end
retrain that promotes with no incumbent and then runs the gate against the fresh champion."""

import tempfile
import unittest
from pathlib import Path

from red_team_agent.planner import RedTeamAgent

from sentinelloop.blue_ml.labeling import load_log, log_round
from sentinelloop.blue_ml.retrain import _should_promote, retrain
from sentinelloop.simulation import simulate_attack, simulate_legitimate_controls


class LabelingTests(unittest.TestCase):
    def test_log_round_writes_all_stages_both_classes(self):
        spec = RedTeamAgent().plan(attack_family="ATO-01", difficulty="medium", seed=7)
        attack = simulate_attack(spec)
        controls = simulate_legitimate_controls(spec, spec.legitimate_controls)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            n = log_round(attack, controls, path)
            df = load_log(path)
            self.assertEqual(len(df), n)
            self.assertEqual(len(df[df.is_attack]), len(attack.events))  # all attack stages logged
            self.assertIn("attack", set(df.source))
            self.assertIn("control", set(df.source))
            self.assertIn("label", df.columns)


class GateTests(unittest.TestCase):
    def test_gate_promotes_and_rejects(self):
        inc = {"chain_recall": 0.9, "hard_false_positive_rate": 0.05}
        self.assertTrue(_should_promote({"chain_recall": 0.92, "hard_false_positive_rate": 0.04}, inc, 0.01))
        # worse recall beyond tolerance -> reject
        self.assertFalse(_should_promote({"chain_recall": 0.80, "hard_false_positive_rate": 0.04}, inc, 0.01))
        # higher hard-FP beyond tolerance -> reject
        self.assertFalse(_should_promote({"chain_recall": 0.95, "hard_false_positive_rate": 0.10}, inc, 0.01))
        # no incumbent -> always promote
        self.assertTrue(_should_promote({"chain_recall": 0.1, "hard_false_positive_rate": 0.9}, None, 0.01))
        # STRICT fp gate: a small hard-FP creep beyond fp_tol (0.002) is rejected even with better recall
        self.assertFalse(_should_promote({"chain_recall": 0.95, "hard_false_positive_rate": 0.055}, inc, 0.01))
        self.assertTrue(_should_promote({"chain_recall": 0.95, "hard_false_positive_rate": 0.051}, inc, 0.01))


class RetrainEndToEndTests(unittest.TestCase):
    def test_first_retrain_promotes_then_gate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            champ = Path(tmp) / "champion"
            log = Path(tmp) / "log.jsonl"  # empty log -> baseline only
            d1 = retrain(log, champion_dir=champ, baseline_seeds_per_cell=1,
                         generation=1, now_iso="2026-01-01T00:00:00+00:00")
            self.assertTrue(d1["promoted"])          # no incumbent -> promoted
            self.assertIsNone(d1["incumbent"])
            self.assertTrue((champ / "model.joblib").exists())
            self.assertIn("chain_recall", d1["challenger"])

            d2 = retrain(log, champion_dir=champ, baseline_seeds_per_cell=1,
                         generation=2, now_iso="2026-01-02T00:00:00+00:00")
            self.assertIsNotNone(d2["incumbent"])    # now gates against the fresh champion
            self.assertIn("promoted", d2)


if __name__ == "__main__":
    unittest.main()
