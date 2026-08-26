"""Regression tests for the Red/Blue hybrid integration boundary."""

from __future__ import annotations

import random
import unittest

from red_team_agent.planner import RedTeamAgent

from sentinelloop.simulation import (
    _materialize_attributes,
    event_delivery_profile,
    simulate_attack,
    simulate_legitimate_controls,
)


class HybridSimulationAlignmentTests(unittest.TestCase):
    def test_probability_fields_keep_blue_evidence_contract_names(self) -> None:
        attributes = _materialize_attributes(
            {
                "bot_behavior_probability": 1.0,
                "identity_mismatch_probability": 1.0,
                "evidence_conflict_probability": 1.0,
                "agent_signature_valid_probability": 0.0,
                "consumer_consent_valid_probability": 0.0,
                "intent_scope_match_probability": 0.0,
                "payment_container_match_probability": 0.0,
                "merchant_scope_match_probability": 0.0,
            },
            rng=random.Random(7),
            event_type="SESSION_STARTED",
        )
        self.assertTrue(attributes["behavior_automation_suspected"])
        self.assertTrue(attributes["identity_consistency_mismatch"])
        self.assertTrue(attributes["evidence_conflict_present"])
        for key in (
            "agent_signature_valid",
            "consumer_consent_valid",
            "intent_scope_match",
            "payment_container_match",
            "merchant_scope_match",
        ):
            self.assertIn(key, attributes)
            self.assertFalse(attributes[key])
            self.assertNotIn(f"{key}_flag", attributes)

    def test_every_simulated_event_uses_its_production_style_delivery_lane(self) -> None:
        scenario = RedTeamAgent().plan(attack_family="AGENT-01", difficulty="medium", seed=11)
        cases = [
            simulate_attack(scenario),
            *simulate_legitimate_controls(scenario, scenario.legitimate_controls),
        ]
        for case in cases:
            for event in case.events:
                expected = event_delivery_profile(event.event_type)
                self.assertEqual(
                    (event.source_system, event.decision_lane, event.latency_budget_ms),
                    expected,
                )

    def test_value_at_risk_covers_agentic_payout_and_dispute_value(self) -> None:
        for family in ("AGENT-01", "PAYOUT-01", "DISPUTE-01"):
            with self.subTest(family=family):
                scenario = RedTeamAgent().plan(attack_family=family, difficulty="medium", seed=17)
                case = simulate_attack(scenario)
                self.assertGreater(sum(record.value_at_risk_inr for record in case.truth), 0.0)
        dispute = simulate_attack(
            RedTeamAgent().plan(attack_family="DISPUTE-01", difficulty="medium", seed=19)
        )
        original = next(record for record in dispute.truth if record.stage_id == "original_purchase")
        refund = next(
            record
            for record, event in zip(dispute.truth, dispute.events)
            if event.event_type == "DISPUTE_REFUND_ISSUED"
        )
        self.assertEqual(original.value_at_risk_inr, 0.0)
        self.assertGreater(refund.value_at_risk_inr, 0.0)


if __name__ == "__main__":
    unittest.main()
