"""Leakage tests for the Phase-4 canonical cross-family attributes (§3).

Each new canonical must (a) be present on BOTH attack and legit rows (no L2 presence leak)
and (b) have OVERLAPPING attack-vs-legit value ranges (no L1 disjoint-bounds leak). The
feature allowlist must stay clean (these were NOT wired into build_features this pass).
"""

import unittest

from sentinelloop.blue_ml.features import audit_leakage
from sentinelloop.dataset import build_dataset

# Canonicals added across both passes. beneficiary_kyc_tier is categorical (overlap == shared value).
NUMERIC_CANONICALS = [
    # pass 1
    "receiver_inbound_sender_count",
    "funds_dwell_time_seconds",
    "balance_drain_ratio",
    "add_to_payment_seconds",
    # pass 2
    "device_binding_age_days",
    "dormancy_days_before_activity",
    "beneficiary_name_match_score",
    "round_amount_ratio",
]
CATEGORICAL_CANONICALS = ["beneficiary_kyc_tier"]


class CanonicalAttributeLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small but full sweep (6 families x 3 difficulties x 2 seeds) + controls + ambient.
        cls.df = build_dataset(seed=42, seeds_per_cell=2)

    def _split(self, col):
        present = self.df[self.df[col].notna()]
        attack = present[present["is_attack"]][col]
        legit = present[~present["is_attack"]][col]
        return attack, legit

    def test_feature_allowlist_still_clean(self):
        # The canonicals are now wired into FEATURES; the allowlist must remain leakage-free.
        self.assertEqual(audit_leakage(), [])

    def test_canonicals_wired_into_features(self):
        from sentinelloop.blue_ml.features import FEATURES
        for col in NUMERIC_CANONICALS + CATEGORICAL_CANONICALS:
            with self.subTest(attribute=col):
                self.assertIn(col, FEATURES, f"{col} not wired into the feature allowlist")

    def test_present_in_both_classes(self):
        for col in NUMERIC_CANONICALS + CATEGORICAL_CANONICALS:
            with self.subTest(attribute=col):
                self.assertIn(col, self.df.columns, f"{col} missing from dataset")
                attack, legit = self._split(col)
                self.assertGreater(len(attack), 0, f"{col} never present on attack rows")
                self.assertGreater(len(legit), 0, f"{col} never present on legit rows")

    def test_numeric_ranges_overlap(self):
        for col in NUMERIC_CANONICALS:
            with self.subTest(attribute=col):
                attack, legit = self._split(col)
                a_lo, a_hi = attack.min(), attack.max()
                l_lo, l_hi = legit.min(), legit.max()
                # Overlap iff neither interval sits entirely on one side of the other.
                self.assertTrue(
                    a_lo <= l_hi and l_lo <= a_hi,
                    f"{col} ranges disjoint: attack[{a_lo},{a_hi}] vs legit[{l_lo},{l_hi}]",
                )

    def test_categorical_values_overlap(self):
        for col in CATEGORICAL_CANONICALS:
            with self.subTest(attribute=col):
                attack, legit = self._split(col)
                shared = set(attack.unique()) & set(legit.unique())
                self.assertTrue(shared, f"{col} shares no value between classes")


if __name__ == "__main__":
    unittest.main()
