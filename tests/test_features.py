import unittest

import pandas as pd

from src.models.features import FORBIDDEN_MODEL_INPUTS, MODEL_FEATURES, build_point_in_time_features
from src.simulator.generate_dataset import generate_dataset


class FeatureEngineeringTests(unittest.TestCase):
    def test_model_features_exclude_ground_truth(self):
        self.assertFalse(FORBIDDEN_MODEL_INPUTS.intersection(MODEL_FEATURES))

    def test_first_event_has_no_history_based_novelty(self):
        events, _ = generate_dataset(
            account_count=20, legitimate_count=100, ato_count=10, mule_count=10, seed=17
        )
        featured = build_point_in_time_features(events)
        first = featured.groupby("sender_account_id", sort=False).head(1)
        self.assertTrue(first["sender_prior_event_count"].eq(0).all())
        self.assertTrue(first[["is_new_device", "is_new_ip", "is_new_beneficiary"]].eq(0).all().all())

    def test_history_only_uses_prior_rows(self):
        events = pd.DataFrame(
            [
                {
                    "event_ts": "2026-08-01T10:00:00+00:00", "sender_account_id": "a1",
                    "beneficiary_id": "b1", "device_id": "d1", "ip_prefix": "10.1.1.0/24",
                    "amount_inr": 100.0, "session_age_seconds": 300, "account_age_days": 500,
                    "beneficiary_age_days": 100, "payment_rail": "UPI", "channel": "MOBILE_APP",
                    "auth_method": "PIN", "merchant_category": "P2P_TRANSFER",
                },
                {
                    "event_ts": "2026-08-01T11:00:00+00:00", "sender_account_id": "a1",
                    "beneficiary_id": "b2", "device_id": "d2", "ip_prefix": "10.2.2.0/24",
                    "amount_inr": 200.0, "session_age_seconds": 200, "account_age_days": 500,
                    "beneficiary_age_days": 0, "payment_rail": "UPI", "channel": "MOBILE_APP",
                    "auth_method": "OTP", "merchant_category": "P2P_TRANSFER",
                },
            ]
        )
        featured = build_point_in_time_features(events)
        self.assertEqual(featured.loc[1, "sender_prior_event_count"], 1)
        self.assertEqual(featured.loc[1, "is_new_device"], 1)
        self.assertEqual(featured.loc[1, "is_new_ip"], 1)
        self.assertEqual(featured.loc[1, "is_new_beneficiary"], 1)


if __name__ == "__main__":
    unittest.main()
