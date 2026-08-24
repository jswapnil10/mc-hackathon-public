from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

from sentinelloop.blue_agent import GenAIBlueAgent
from sentinelloop.config import AgentLabConfig
from sentinelloop.contracts import BlueDecision, ObservedEvent
from sentinelloop.model_gateway import ModelCall
from sentinelloop.orchestrator import SentinelLoopOrchestrator
from sentinelloop.simulation import simulate_attack
from red_team_agent.planner import RedTeamAgent


class TestGateway:
    """Model-shaped test double; production never uses this gateway."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[dict[str, Any], ModelCall]:
        self.calls.append(
            {
                "agent_name": agent_name,
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "schema": schema,
            }
        )
        trace = ModelCall(agent_name, model, schema_name, 1)
        if agent_name == "red_planner":
            card = user_payload["attack_cards"][0]
            changes = []
            if user_payload["referee_feedback"]:
                changes = [
                    {
                        "parameter": "new_device_probability",
                        "value": 0.55,
                        "rationale": "Reduce the strongest declassified novelty signal.",
                    }
                ]
            return (
                {
                    "attack_family": card["attack_family"],
                    "difficulty": user_payload["requested_difficulty"],
                    "objective": "Test whether defense joins session, beneficiary, and payment evidence early.",
                    "stage_emphasis": [stage["stage_id"] for stage in card["stages"][:2]],
                    "adaptation_hypothesis": "A less novel session may force defense to use the full sequence.",
                    "parameter_changes": changes,
                    "reasoning_summary": "Use a bounded account-takeover campaign with staged behavioral evidence.",
                },
                trace,
            )
        if agent_name == "blue_investigator":
            return (
                {
                    "preliminary_risk": "Inspect the sequence and test both risky and legitimate explanations.",
                    "requested_tools": [
                        "timeline_summary",
                        "payment_context",
                        "legitimate_alternatives",
                    ],
                    "investigation_focus": ["sequence", "payment size", "verified context"],
                },
                trace,
            )
        if agent_name in {"blue_decider", "blue_decider_repair"}:
            event = user_payload["current_event"]
            attributes = event["attributes"]
            evidence_refs = [item["evidence_id"] for item in user_payload["tool_evidence"]]
            supporting = any(
                key in attributes
                for key in (
                    "customer_notified_change",
                    "customer_confirmation_complete",
                    "strong_authentication_passed",
                    "out_of_band_verification_complete",
                    "registered_entity_age_days",
                    "independent_checks_passed",
                    "recurring_mandate_age_days",
                )
            )
            ratio = float(attributes.get("amount_inr", 0.0)) / max(
                float(attributes.get("sender_baseline_amount_inr", 1.0)), 1.0
            )
            if supporting:
                action, risk, reasons = "allow", "low", ["legitimate_context"]
            elif (
                any(item["action"] == "step_up" for item in user_payload["prior_blue_decisions"])
                and event["event_type"]
                in {"PAYMENT_INITIATED", "PAYMENT_REPEATED", "FUNDS_RECEIVED", "FUNDS_DISPERSED"}
            ):
                action, risk, reasons = "hold", "high", ["behavior_sequence", "velocity"]
            elif event["event_type"] == "BENEFICIARY_ADDED":
                action, risk, reasons = "step_up", "high", ["beneficiary_novelty", "behavior_sequence"]
            elif event["event_type"] == "PAYMENT_INITIATED" and ratio >= 2.0:
                action, risk, reasons = "block", "critical", ["amount_anomaly", "velocity"]
            else:
                action, risk, reasons = "monitor", "medium", ["insufficient_evidence"]
            return (
                {
                    "event_id": event["event_id"],
                    "action": action,
                    "risk_level": risk,
                    "confidence": 0.88,
                    "reason_codes": reasons,
                    "evidence_refs": evidence_refs,
                    "decision_summary": "The action follows the visible evidence and proportionality policy.",
                    "mitigation": "Apply the selected action and retain the evidence trail.",
                },
                trace,
            )
        raise AssertionError(f"Unexpected agent {agent_name}")


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


class InformationBoundaryTests(unittest.TestCase):
    def test_simulator_keeps_truth_out_of_observed_events(self) -> None:
        scenario = RedTeamAgent().plan(attack_family="ATO-01", difficulty="medium", seed=11)
        case = simulate_attack(scenario)
        observed_keys = _all_keys([event.to_dict() for event in case.events])
        self.assertFalse({"attack_family", "scenario_id", "stage_id", "is_attack"} & observed_keys)
        self.assertTrue(all(record.attack_family == "ATO-01" for record in case.truth))

    def test_blue_model_calls_never_receive_sealed_truth(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        result = lab.run(attack_family="ATO-01", rounds=1, seed=21)
        self.assertEqual(len(result.rounds), 1)
        forbidden = {"attack_family", "scenario_id", "stage_id", "is_attack", "label_fraud"}
        for call in gateway.calls:
            if call["agent_name"].startswith("blue_"):
                self.assertFalse(forbidden & _all_keys(call["user_payload"]))

    def test_red_receives_only_declassified_feedback_on_next_round(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        result = lab.run(attack_family="ATO-01", rounds=2, seed=31)
        red_calls = [call for call in gateway.calls if call["agent_name"] == "red_planner"]
        feedback = red_calls[1]["user_payload"]["referee_feedback"]
        self.assertEqual(
            set(feedback),
            {
                "outcome",
                "detected_stage_id",
                "time_to_detect_seconds",
                "value_prevented_ratio",
                "false_positive_rate",
                "coarse_reason_categories",
            },
        )
        self.assertEqual(result.rounds[1].red_turn.scenario.parent_scenario_id, result.rounds[0].red_turn.scenario.scenario_id)


class ClosedLoopTests(unittest.TestCase):
    def test_policy_guard_promotes_hold_medium_to_high(self) -> None:
        decision = BlueDecision(
            event_id="syn_evt_1",
            action="hold",
            risk_level="medium",
            confidence=0.8,
            reason_codes=["velocity"],
            evidence_refs=[],
            decision_summary="Hold the payment while the case is investigated.",
            mitigation="Place a temporary hold.",
        )
        normalized, adjustments = GenAIBlueAgent._normalize_policy_labels(
            decision,
            prior_turns=[SimpleNamespace(decision=SimpleNamespace(action="step_up"))],
        )
        self.assertEqual(normalized.action, "hold")
        self.assertEqual(normalized.risk_level, "high")
        self.assertTrue(adjustments)

    def test_two_genai_roles_and_referee_complete_a_round(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        result = lab.run(attack_family="ATO-01", rounds=1, seed=41)
        round_result = result.rounds[0]
        self.assertTrue(round_result.attack_blue_turns)
        self.assertEqual(len(round_result.control_results), 3)
        self.assertGreater(round_result.referee_report.attack_detection_rate, 0)
        self.assertEqual(round_result.referee_report.hard_false_positive_rate, 0)
        self.assertTrue(any(call["agent_name"] == "red_planner" for call in gateway.calls))
        self.assertTrue(any(call["agent_name"] == "blue_investigator" for call in gateway.calls))
        self.assertTrue(any(call["agent_name"] == "blue_decider" for call in gateway.calls))
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
