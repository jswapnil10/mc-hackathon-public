from __future__ import annotations

import unittest

from sentinelloop.benchmark import (
    FORBIDDEN_FEATURES,
    MODEL_FEATURES,
    assess_data_quality,
    run_population_benchmark,
)
from sentinelloop.population import PopulationConfig, generate_population_dataset
from sentinelloop.threat_atlas import ThreatAtlas


class ThreatAtlasTests(unittest.TestCase):
    def test_atlas_is_source_backed_and_simulation_ready(self) -> None:
        atlas = ThreatAtlas()
        summary = atlas.summary()
        self.assertGreaterEqual(summary["vector_count"], 30)
        self.assertEqual(summary["simulation_ready_vector_count"], summary["vector_count"])
        self.assertEqual(summary["attack_family_count"], 9)
        self.assertGreaterEqual(summary["rail_count"], 7)
        self.assertGreaterEqual(summary["source_count"], 10)
        self.assertTrue(all(source.url.startswith("https://") for source in atlas.sources))
        self.assertTrue(all(vector.source_ids for vector in atlas.vectors))


class PopulationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PopulationConfig(
            variants_per_vector=4,
            legitimate_event_count=400,
            seed=741,
        )
        cls.atlas = ThreatAtlas()
        cls.benchmark, cls.events, cls.predictions = run_population_benchmark(cls.config)

    def test_generation_covers_every_vector_family_and_split(self) -> None:
        metadata = self.benchmark["dataset"]
        self.assertEqual(metadata["attack_vector_count"], len(self.atlas.vectors))
        self.assertEqual(metadata["attack_family_count"], 9)
        self.assertEqual(set(metadata["split_counts"]), {"train", "validation", "test_known", "test_novel"})
        self.assertTrue(all(count > 0 for count in metadata["split_counts"].values()))

    def test_sealed_novel_vectors_and_entities_do_not_leak(self) -> None:
        novel_rows = self.events[self.events["attack_vector_id"].isin(self.atlas.novel_holdout_vector_ids)]
        self.assertEqual(set(novel_rows["split"]), {"test_novel"})
        scenario_split_counts = (
            self.events.dropna(subset=["scenario_id"])
            .groupby("scenario_id")["split"]
            .nunique()
        )
        entity_split_counts = self.events.groupby("sender_account_id")["split"].nunique()
        self.assertTrue(scenario_split_counts.le(1).all())
        self.assertTrue(entity_split_counts.le(1).all())

    def test_defense_features_exclude_answer_keys(self) -> None:
        self.assertFalse(FORBIDDEN_FEATURES.intersection(MODEL_FEATURES))

    def test_quality_fidelity_and_hidden_metrics_are_inspectable(self) -> None:
        quality = assess_data_quality(self.events, self.atlas)
        self.assertEqual(quality["status"], "passed")
        self.assertEqual(quality["critical_failure_count"], 0)
        self.assertEqual(
            self.benchmark["fidelity"]["reference_type"],
            "declared_behavioral_priors_not_real_customer_data",
        )
        metrics = self.benchmark["defense"]["metrics"]["combined_hidden_test"]
        for name in ("pr_auc", "precision", "recall", "f1", "false_positive_rate"):
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)
        self.assertGreater(metrics["confusion_matrix"]["tp"], 0)
        self.assertGreater(metrics["confusion_matrix"]["tn"], 0)

    def test_generation_is_reproducible_for_fixed_seed(self) -> None:
        replay, replay_metadata = generate_population_dataset(self.config, atlas=self.atlas)
        columns = ["event_id", "case_id", "split", "label_fraud", "amount_inr"]
        self.assertTrue(self.events[columns].equals(replay[columns]))
        self.assertEqual(
            self.benchmark["dataset"]["split_counts"], replay_metadata["split_counts"]
        )


if __name__ == "__main__":
    unittest.main()
