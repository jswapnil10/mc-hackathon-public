from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelloop.external_validation import (
    PCA_FEATURES,
    ExternalValidationConfig,
    assess_external_data_quality,
    chronological_splits,
    run_external_validation,
)


def _external_fixture(row_count: int = 2400, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(row_count, dtype=float) * 60.0
    labels = np.zeros(row_count, dtype=int)
    fraud_indexes = np.concatenate(
        [
            np.arange(30, int(row_count * 0.6), 90),
            np.arange(int(row_count * 0.6) + 10, int(row_count * 0.8), 60),
            np.arange(int(row_count * 0.8) + 10, row_count, 55),
        ]
    )
    labels[fraud_indexes] = 1
    payload = {
        feature: rng.normal(loc=labels * (1.4 if index < 4 else 0.1), size=row_count)
        for index, feature in enumerate(PCA_FEATURES)
    }
    payload.update(
        {
            "Time": time,
            "Amount": rng.lognormal(mean=3.5 + labels * 0.4, sigma=1.0),
            "Class": labels,
        }
    )
    return pd.DataFrame(payload)


class ExternalValidationTests(unittest.TestCase):
    def test_quality_gate_requires_published_schema(self) -> None:
        frame = _external_fixture()
        report = assess_external_data_quality(frame)
        self.assertEqual(report["critical_failure_count"], 0)
        self.assertEqual(report["row_count"], len(frame))
        self.assertEqual(report["dataset_grain"], "one anonymized card transaction per row")

    def test_chronological_split_does_not_shuffle_future_rows(self) -> None:
        frame = _external_fixture()
        splits = chronological_splits(
            frame, validation_fraction=0.20, test_fraction=0.20
        )
        self.assertLessEqual(splits["train"]["Time"].max(), splits["validation"]["Time"].min())
        self.assertLessEqual(splits["validation"]["Time"].max(), splits["test"]["Time"].min())
        self.assertEqual(sum(len(split) for split in splits.values()), len(frame))

    def test_external_benchmark_selects_on_validation_and_scores_future_test(self) -> None:
        frame = _external_fixture()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "external.csv"
            frame.to_csv(input_path, index=False)
            report, predictions = run_external_validation(
                ExternalValidationConfig(
                    input_path=input_path,
                    output_dir=Path(directory) / "outputs",
                    seed=19,
                ),
                synthetic_population_path=None,
            )
        self.assertTrue(report["methodology"]["test_opened_after_selection"])
        self.assertIn(report["defense"]["selected_model"], report["defense"]["candidate_models"])
        self.assertEqual(len(predictions), report["defense"]["test_metrics"]["event_count"])
        self.assertGreater(report["defense"]["test_metrics"]["pr_auc"], 0.05)
        self.assertIn("brier_skill_score", report["defense"]["test_metrics"])
        self.assertIn("expected_calibration_error", report["defense"]["test_metrics"])
        self.assertEqual(report["synthetic_calibration"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
