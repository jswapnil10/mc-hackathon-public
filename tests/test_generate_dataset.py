import unittest

from src.simulator.generate_dataset import generate_dataset


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.metadata = generate_dataset(
            account_count=40, legitimate_count=200, ato_count=20, mule_count=20, seed=42
        )

    def test_expected_volume_and_required_fields(self):
        self.assertEqual(len(self.frame), 240)
        self.assertEqual(self.metadata["fraud_events"], 40)
        self.assertTrue({"event_id", "event_ts", "label_fraud", "attack_family"}.issubset(self.frame.columns))
        self.assertTrue(self.frame["event_id"].is_unique)

    def test_attack_labels_and_scenarios_are_consistent(self):
        fraud = self.frame[self.frame["label_fraud"]]
        legitimate = self.frame[~self.frame["label_fraud"]]
        self.assertEqual(set(fraud["attack_family"]), {"ATO-01", "MULE-01"})
        self.assertTrue(fraud["scenario_id"].notna().all())
        self.assertTrue(legitimate["attack_family"].isna().all())
        self.assertTrue(legitimate["scenario_id"].isna().all())

    def test_contains_multiple_campaigns_and_legitimate_controls(self):
        fraud = self.frame[self.frame["label_fraud"]]
        self.assertGreater(fraud["scenario_id"].nunique(), 2)
        self.assertGreater(self.frame["legitimate_control"].notna().sum(), 0)
        self.assertFalse(self.frame.loc[self.frame["label_fraud"], "legitimate_control"].notna().any())

    def test_deterministic_seed(self):
        second, _ = generate_dataset(account_count=40, legitimate_count=200, ato_count=20, mule_count=20, seed=42)
        self.assertEqual(self.frame.to_csv(index=False), second.to_csv(index=False))


if __name__ == "__main__":
    unittest.main()
