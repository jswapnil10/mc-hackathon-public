from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from sentinelloop.blue_agent import GenAIBlueAgent
from sentinelloop.config import AgentLabConfig
from sentinelloop.contracts import BlueDecision, DefensePlaybook, ObservedEvent
from sentinelloop.model_gateway import ModelCall
from sentinelloop.orchestrator import SentinelLoopOrchestrator
from sentinelloop.red_agent import GenAIRedAgent
from sentinelloop.evidence import _entity_linkage, synthesize_case_risk
from sentinelloop.simulation import simulate_attack
from red_team_agent.planner import RedTeamAgent, lifecycle_phase_for_template


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
            focus_stage = next(stage for stage in card["stages"] if stage["mutable_parameters"])
            parameter = focus_stage["mutable_parameters"][0]
            value = card["parameter_profiles"][user_payload["requested_difficulty"]][parameter]
            return (
                {
                    "attack_family": card["attack_family"],
                    "difficulty": user_payload["requested_difficulty"],
                    "objective": "Test whether defense joins session, beneficiary, and payment evidence early.",
                    "target_lifecycle_phase": focus_stage["lifecycle_phase"],
                    "focus_stage_ids": [focus_stage["stage_id"]],
                    "adaptation_goal": "Stress the selected lifecycle control using one bounded behavioral change.",
                    "adaptation_hypothesis": "A less novel session may force defense to use the full sequence.",
                    "parameter_changes": [
                        {
                            "parameter": parameter,
                            "value": value,
                            "rationale": "Keep the change bounded and tied to the selected focus stage.",
                        }
                    ],
                    "reasoning_summary": "Use a bounded account-takeover campaign with staged behavioral evidence.",
                },
                trace,
            )
        if agent_name == "blue_strategist":
            return (
                {
                    "preferred_tools": ["communication_risk", "behavioral_biometrics"],
                    "focus_reason_codes": ["identity_linkage", "behavior_sequence"],
                    "investigation_guidance": [
                        "Join entity relationships with behavioral continuity before value moves."
                    ],
                    "change_hypothesis": (
                        "Adding relationship and behavioral evidence should improve coverage without "
                        "increasing legitimate customer friction."
                    ),
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
        if agent_name in {"blue_event_agent", "blue_event_repair"}:
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
            response = {
                    "event_id": event["event_id"],
                    "action": action,
                    "risk_level": risk,
                    "confidence": 0.88,
                    "reason_codes": reasons,
                    "evidence_refs": evidence_refs,
                    "decision_summary": "The action follows the visible evidence and proportionality policy.",
                    "mitigation": "Apply the selected action and retain the evidence trail.",
                }
            if agent_name == "blue_event_agent":
                response.update(
                    {
                        "preliminary_risk": (
                            "Inspect the sequence and test both risky and legitimate explanations."
                        ),
                        "requested_tools": user_payload["available_evidence_tools"][:4],
                        "investigation_focus": [
                            "sequence",
                            "payment size",
                            "verified context",
                        ],
                    }
                )
            return response, trace
        raise AssertionError(f"Unexpected agent {agent_name}")


class PhaseMismatchGateway(TestGateway):
    def generate_json(self, **kwargs: Any) -> tuple[dict[str, Any], ModelCall]:
        result, model_call = super().generate_json(**kwargs)
        if kwargs["agent_name"] == "red_planner":
            focus_id = result["focus_stage_ids"][0]
            card = kwargs["user_payload"]["attack_cards"][0]
            actual_phase = next(
                stage["lifecycle_phase"]
                for stage in card["stages"]
                if stage["stage_id"] == focus_id
            )
            result["target_lifecycle_phase"] = next(
                phase
                for phase in ("pre_transaction", "transaction", "post_transaction")
                if phase != actual_phase
            )
        return result, model_call


class BlueTimeoutGateway(TestGateway):
    def generate_json(self, **kwargs: Any) -> tuple[dict[str, Any], ModelCall]:
        if kwargs["agent_name"] in {"blue_event_agent", "blue_event_repair"}:
            raise RuntimeError("The local model missed its operational window.")
        return super().generate_json(**kwargs)


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
    def test_blue_model_timeout_keeps_deterministic_control_lane_available(self) -> None:
        case = simulate_attack(
            RedTeamAgent().plan(attack_family="ATO-01", difficulty="medium", seed=2026)
        )
        blue = GenAIBlueAgent(
            gateway=BlueTimeoutGateway(),
            config=AgentLabConfig(ml_detector_enabled=False),
        )
        turn = blue.investigate_event(
            event=case.events[0],
            visible_history=[case.events[0]],
            prior_turns=[],
            playbook=DefensePlaybook.baseline(),
            seed=2026,
        )
        self.assertIn(turn.decision.action, {"monitor", "step_up", "hold"})
        self.assertEqual(turn.model_calls, [])
        self.assertIn("Qwen did not complete", turn.decision.decision_summary)
        self.assertTrue(
            any("Availability fallback" in item for item in turn.policy_adjustments)
        )

    def test_red_normalizes_phase_label_from_bounded_focus_stage(self) -> None:
        agent = GenAIRedAgent(
            gateway=PhaseMismatchGateway(),
            config=AgentLabConfig(),
        )
        turn = agent.plan(
            attack_family="ATO-01",
            difficulty="medium",
            seed=2026,
        )
        selected_stage = next(
            stage
            for stage in agent.catalog.get("ATO-01").stage_templates
            if stage["stage_id"] == turn.plan.focus_stage_ids[0]
        )
        self.assertEqual(
            turn.plan.target_lifecycle_phase,
            lifecycle_phase_for_template(selected_stage),
        )

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
    def test_one_round_progress_is_truthful_and_does_not_claim_adaptation(self) -> None:
        updates: list[dict[str, Any]] = []
        gateway = TestGateway()
        SentinelLoopOrchestrator(
            config=AgentLabConfig(),
            gateway=gateway,
            progress_callback=updates.append,
        ).run(attack_family="ATO-01", rounds=1, seed=40)

        stages = [update["stage"] for update in updates]
        self.assertEqual(stages.count("red_planning"), 1)
        self.assertEqual(stages.count("simulation"), 1)
        self.assertEqual(stages.count("referee_scoring"), 1)
        self.assertEqual(stages.count("completed"), 1)
        self.assertNotIn("blue_adaptation", stages)
        self.assertNotIn("blue_replay", stages)
        self.assertGreaterEqual(stages.count("blue_investigation"), 2)
        blue_updates = [
            update for update in updates if update["stage"] == "blue_investigation"
        ]
        self.assertGreater(blue_updates[-1]["completed_events"], 0)

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
            prior_turns=[
                SimpleNamespace(
                    decision=SimpleNamespace(action="step_up", reason_codes=["velocity"])
                )
            ],
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
        self.assertEqual(
            set(round_result.referee_report.lifecycle_metrics),
            {"pre_transaction", "transaction", "post_transaction"},
        )
        self.assertGreaterEqual(
            round_result.referee_report.balanced_lifecycle_defense_score, 0
        )
        self.assertLessEqual(
            round_result.referee_report.balanced_lifecycle_defense_score, 100
        )
        self.assertGreaterEqual(round_result.referee_report.red_capability_score, 0)
        self.assertLessEqual(round_result.referee_report.red_capability_score, 100)
        self.assertAlmostEqual(
            round_result.referee_report.realized_impact_inr
            + round_result.referee_report.value_prevented_inr,
            round_result.referee_report.total_value_at_risk_inr,
        )
        self.assertTrue(any(call["agent_name"] == "red_planner" for call in gateway.calls))
        self.assertTrue(any(call["agent_name"] == "blue_event_agent" for call in gateway.calls))
        blue_calls = [call for call in gateway.calls if call["agent_name"] == "blue_event_agent"]
        model_evaluated_events = sum(
            bool(turn.model_calls)
            for turn in round_result.attack_blue_turns
        ) + sum(
            bool(turn.model_calls)
            for _, turns in round_result.control_results
            for turn in turns
        )
        self.assertEqual(len(blue_calls), model_evaluated_events)
        # NOTE (merge: ML-branch mirror controls): controls now MIRROR the attack chain (same
        # stages/keys, benign overlapping values) as hard ML negatives, so they no longer carry
        # deterministic verified-legitimacy markers and are scrutinized by the model. A hard
        # look-alike may incur step_up/monitor friction, but a legitimate customer must never be
        # money-blocked. We assert the no-hard-block property (hold/block are disallowed on controls).
        self.assertTrue(
            all(
                turn.decision.action not in {"hold", "block"}
                for _, turns in round_result.control_results
                for turn in turns
            )
        )
        self.assertTrue(
            all(
                isinstance(packet.facts, dict)
                for turn in round_result.attack_blue_turns
                for packet in turn.evidence
            )
        )
        json.dumps(result.to_dict())

    def test_blue_candidate_is_not_promoted_for_tool_coverage_alone(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        result = lab.run(attack_family="ATO-01", rounds=2, seed=51)
        adaptation = result.rounds[0].blue_adaptation
        self.assertIsNotNone(adaptation)
        self.assertFalse(adaptation.promoted)
        self.assertGreater(adaptation.replay_report.evidence_tool_coverage, result.rounds[0].referee_report.evidence_tool_coverage)
        self.assertEqual(result.rounds[1].active_blue_playbook.version, 1)
        self.assertEqual(result.final_defense_playbook.version, 1)

    def test_fast_sequence_guard_catches_hard_low_and_slow_repetition(self) -> None:
        scenario = RedTeamAgent().plan(
            attack_family="EVADE-01", difficulty="hard", seed=20260824
        )
        case = simulate_attack(scenario)
        repeated_index = next(
            index for index, event in enumerate(case.events) if event.event_type == "PAYMENT_REPEATED"
        )
        synthesis = synthesize_case_risk(case.events[: repeated_index + 1])
        self.assertEqual(synthesis["minimum_action"], "hold")
        self.assertFalse(synthesis["uses_sealed_truth"])

    def test_same_case_entity_repetition_is_not_trusted_history(self) -> None:
        scenario = RedTeamAgent().plan(
            attack_family="EVADE-01", difficulty="hard", seed=20260824
        )
        facts = _entity_linkage(simulate_attack(scenario).events)
        self.assertTrue(facts["same_case_repeated_entity_paths"])
        self.assertFalse(facts["historical_relationship_verified"])

    def test_round_exports_judge_facing_fidelity_and_feasibility(self) -> None:
        gateway = TestGateway()
        result = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway).run(
            attack_family="AGENT-01", rounds=1, seed=91
        )
        evaluation = result.rounds[0].submission_evaluation
        self.assertEqual(evaluation["fidelity"]["lab_only_event_types_exposed"], [])
        self.assertTrue(evaluation["fidelity"]["truth_boundary_clean"])
        self.assertEqual(evaluation["live_feasibility"]["pre_model_fast_path_coverage"], 1.0)
        self.assertIn(
            "fast_guard_actionable_event_count", evaluation["detection_efficacy"]
        )
        self.assertEqual(result.submission_profile["diversity"]["attack_family_count"], 9)

    def test_hold_continues_evaluation_until_resolution_or_block(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        result = lab.run(attack_family="ATO-01", rounds=1, seed=61)
        turns = result.rounds[0].attack_blue_turns
        hold_index = next(
            index for index, turn in enumerate(turns) if turn.decision.action == "hold"
        )
        self.assertGreater(len(turns), hold_index + 1)

    def test_blue_candidate_with_more_customer_harm_is_rejected(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        baseline = lab.run(attack_family="ATO-01", rounds=1, seed=71).rounds[0].referee_report
        unsafe_candidate = replace(
            baseline,
            blue_score=min(100.0, baseline.blue_score + 1.0),
            hard_false_positive_rate=1.0,
        )
        promoted, reason = lab._promotion_decision(baseline, unsafe_candidate)
        self.assertFalse(promoted)
        self.assertIn("legitimate-case safety", reason)

    def test_resolved_hold_does_not_claim_prevented_value(self) -> None:
        gateway = TestGateway()
        lab = SentinelLoopOrchestrator(config=AgentLabConfig(), gateway=gateway)
        round_result = lab.run(attack_family="ATO-01", rounds=1, seed=81).rounds[0]
        last_turn = round_result.attack_blue_turns[-1]
        resolved_decision = replace(
            last_turn.decision,
            action="allow",
            risk_level="low",
            reason_codes=["legitimate_context"],
        )
        resolved_turns = [
            *round_result.attack_blue_turns[:-1],
            replace(last_turn, decision=resolved_decision),
        ]
        report = lab.referee.score(
            attack_case=round_result.attack_case,
            attack_turns=resolved_turns,
            control_results=round_result.control_results,
        )
        self.assertEqual(report.outcome, "detected")
        self.assertEqual(report.value_prevented_ratio, 0.0)
        self.assertEqual(report.realized_impact_inr, report.total_value_at_risk_inr)


if __name__ == "__main__":
    unittest.main()
