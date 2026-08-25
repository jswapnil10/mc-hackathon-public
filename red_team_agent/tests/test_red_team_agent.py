from __future__ import annotations

import unittest
from dataclasses import replace

from red_team_agent.catalog import AttackCatalog
from red_team_agent.models import RefereeFeedback
from red_team_agent.mutation import MutationController
from red_team_agent.openai_backend import _extract_output_text
from red_team_agent.planner import RedTeamAgent
from red_team_agent.safety import ScenarioSafetyGate


EXPECTED_FAMILIES = {
    "AGENT-01",
    "APP-01",
    "ATO-01",
    "BEC-01",
    "MULE-01",
    "SYNID-01",
    "EVADE-01",
    "PAYOUT-01",
    "DISPUTE-01",
}


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = AttackCatalog()

    def test_catalog_contains_nine_distinct_families(self) -> None:
        self.assertEqual(set(self.catalog.families), EXPECTED_FAMILIES)

    def test_cards_have_sources_profiles_and_controls(self) -> None:
        for card in self.catalog.list():
            with self.subTest(card=card.attack_family):
                self.assertTrue(card.source_refs)
                self.assertTrue(all(source["url"].startswith("https://") for source in card.source_refs))
                self.assertEqual(set(card.parameter_profiles), {"easy", "medium", "hard"})
                self.assertTrue(card.legitimate_controls)
                for profile in card.parameter_profiles.values():
                    self.assertEqual(set(profile), set(card.parameter_bounds))

    def test_catalog_never_exposes_lab_control_events_to_blue(self) -> None:
        event_types = {
            stage["event_type"]
            for card in self.catalog.list()
            for stage in card.stage_templates
        }
        self.assertNotIn("CAMPAIGN_REPLAYED", event_types)
        self.assertIn("ACCOUNT_BEHAVIOR_PROFILE_UPDATED", event_types)
        self.assertIn("AGENTIC_PAYMENT_INITIATED", event_types)


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = AttackCatalog()
        self.agent = RedTeamAgent(catalog=self.catalog)
        self.gate = ScenarioSafetyGate(self.catalog)

    def test_every_family_and_difficulty_compiles_safely(self) -> None:
        for family in self.catalog.families:
            for difficulty in ("easy", "medium", "hard"):
                with self.subTest(family=family, difficulty=difficulty):
                    scenario = self.agent.plan(
                        attack_family=family,
                        difficulty=difficulty,
                        seed=20260819,
                    )
                    report = self.gate.validate(scenario)
                    self.assertTrue(report.approved, report.errors)
                    self.assertEqual(report.warnings, [])
                    focused = {
                        stage.lifecycle_phase
                        for stage in scenario.stages
                        if stage.stage_id in scenario.focus_stage_ids
                    }
                    self.assertEqual(focused, {scenario.target_lifecycle_phase})

    def test_offline_plan_is_deterministic(self) -> None:
        first = self.agent.plan(attack_family="ATO-01", difficulty="medium", seed=42)
        second = self.agent.plan(attack_family="ATO-01", difficulty="medium", seed=42)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_safety_gate_rejects_forbidden_fields(self) -> None:
        scenario = self.agent.plan(attack_family="ATO-01", seed=7)
        first_stage = replace(
            scenario.stages[0],
            attributes={**scenario.stages[0].attributes, "credential_material": "not-allowed"},
        )
        unsafe = replace(scenario, stages=[first_stage, *scenario.stages[1:]])
        report = self.gate.validate(unsafe)
        self.assertFalse(report.approved)
        self.assertTrue(any("Forbidden field" in error for error in report.errors))

    def test_safety_gate_rejects_real_entity_identifier(self) -> None:
        scenario = self.agent.plan(attack_family="APP-01", seed=8)
        first_stage = replace(
            scenario.stages[0],
            attributes={**scenario.stages[0].attributes, "sender_account_id": "real-account-1"},
        )
        unsafe = replace(scenario, stages=[first_stage, *scenario.stages[1:]])
        report = self.gate.validate(unsafe)
        self.assertFalse(report.approved)
        self.assertTrue(any("syn_" in error for error in report.errors))

    def test_safety_gate_rejects_unbounded_parameters(self) -> None:
        scenario = self.agent.plan(attack_family="ATO-01", seed=9)
        unsafe = replace(scenario, parameters={**scenario.parameters, "unreviewed_knob": 1})
        report = self.gate.validate(unsafe)
        self.assertFalse(report.approved)
        self.assertTrue(any("unbounded parameters" in error for error in report.errors))


class MutationTests(unittest.TestCase):
    def test_feedback_creates_bounded_child_scenario(self) -> None:
        catalog = AttackCatalog()
        scenario = RedTeamAgent(catalog=catalog).plan(
            attack_family="ATO-01",
            difficulty="medium",
            seed=100,
        )
        feedback = RefereeFeedback(
            outcome="detected",
            detected_stage_id="novel_session",
            value_prevented_ratio=1.0,
            time_to_detect_seconds=10,
            coarse_reason_categories=["device_novelty", "network_novelty"],
            false_positive_rate=0.01,
        )
        mutated = MutationController(catalog).mutate(scenario, feedback)
        self.assertEqual(mutated.parent_scenario_id, scenario.scenario_id)
        self.assertEqual(mutated.mutation_number, 1)
        self.assertLess(mutated.parameters["new_device_probability"], scenario.parameters["new_device_probability"])
        self.assertLess(mutated.parameters["new_network_probability"], scenario.parameters["new_network_probability"])
        self.assertTrue(ScenarioSafetyGate(catalog).validate(mutated).approved)

        second_mutation = MutationController(catalog).mutate(mutated, feedback)
        self.assertEqual(second_mutation.seed, mutated.seed + 1)
        self.assertEqual(second_mutation.mutation_number, 2)

    def test_unknown_feedback_category_is_rejected(self) -> None:
        catalog = AttackCatalog()
        scenario = RedTeamAgent(catalog=catalog).plan(attack_family="ATO-01", seed=101)
        feedback = RefereeFeedback(
            outcome="detected",
            detected_stage_id="novel_session",
            value_prevented_ratio=1.0,
            time_to_detect_seconds=10,
            coarse_reason_categories=["reveal_blue_model_threshold"],
            false_positive_rate=0.01,
        )
        with self.assertRaisesRegex(ValueError, "feedback rejected"):
            MutationController(catalog).mutate(scenario, feedback)


class OpenAIResponseParsingTests(unittest.TestCase):
    def test_extracts_structured_output_text(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"attack_family":"ATO-01"}'}],
                }
            ]
        }
        self.assertEqual(_extract_output_text(response), '{"attack_family":"ATO-01"}')


if __name__ == "__main__":
    unittest.main()
