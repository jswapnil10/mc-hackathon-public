"""Judge-facing evidence for the five published hackathon evaluation criteria."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

from red_team_agent.catalog import AttackCatalog
from red_team_agent.planner import lifecycle_phase_for_template

from .contracts import BlueTurn, RefereeReport, SimulationCase


LAB_ONLY_EVENT_TYPES = {"CAMPAIGN_REPLAYED"}
SEALED_KEYS = {"attack_family", "scenario_id", "stage_id", "is_attack", "label_fraud"}


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def catalog_submission_profile(catalog: AttackCatalog | None = None) -> dict[str, Any]:
    active_catalog = catalog or AttackCatalog()
    cards = active_catalog.list()
    surfaces = sorted({surface for card in cards for surface in card.payment_surface})
    event_types = sorted(
        {stage["event_type"] for card in cards for stage in card.stage_templates}
    )
    phases = sorted(
        {
            lifecycle_phase_for_template(stage)
            for card in cards
            for stage in card.stage_templates
        }
    )
    source_urls = {
        source["url"] for card in cards for source in card.source_refs if source.get("url")
    }
    source_domains = sorted({urlparse(url).netloc for url in source_urls})
    matrix = [
        {
            "attack_family": card.attack_family,
            "name": card.name,
            "payment_surfaces": card.payment_surface,
            "lifecycle_phases": sorted(
                {lifecycle_phase_for_template(stage) for stage in card.stage_templates}
            ),
            "stage_count": len(card.stage_templates),
            "bounded_mutation_count": len(card.allowed_mutations),
            "legitimate_control_count": len(card.legitimate_controls),
            "source_count": len(card.source_refs),
        }
        for card in cards
    ]
    return {
        "diversity": {
            "attack_family_count": len(cards),
            "observable_event_type_count": len(event_types),
            "payment_surface_count": len(surfaces),
            "lifecycle_phase_count": len(phases),
            "legitimate_control_count": sum(len(card.legitimate_controls) for card in cards),
            "bounded_adaptation_parameter_count": sum(
                len(card.allowed_mutations) for card in cards
            ),
            "authoritative_source_count": len(source_urls),
            "source_domains": source_domains,
            "event_types": event_types,
            "payment_surfaces": surfaces,
            "coverage_matrix": matrix,
        },
        "fidelity": {
            "simulation_contract": "seeded_deterministic_synthetic_event_stream",
            "truth_boundary": "sealed_truth_is_never_in_blue_observables",
            "look_alike_testing": "three_family_specific_legitimate_controls_per_attack",
            "production_metadata": ["source_system", "decision_lane", "latency_budget_ms"],
            "lab_only_event_types_allowed": [],
        },
        "detection": {
            "layers": [
                "pre_model_observable_sequence_guard",
                "verified_legitimate_context_fast_exit",
                "qwen_evidence_grounded_investigator_and_decider",
                "deterministic_policy_and_continuity_gate",
                "legitimate_look_alike_referee_evaluation",
            ],
            "efficacy_metrics": [
                "value_prevented_ratio",
                "hard_false_positive_rate",
                "legitimate_friction_rate",
                "time_to_detect_seconds",
                "balanced_lifecycle_defense_score",
                "worst_phase_score",
            ],
        },
        "novelty": {
            "features": [
                "two_speed_genai_plus_deterministic_data_plane",
                "red_and_blue_genai_agents_with_sealed_referee",
                "guarded_red_and_blue_feedback_loops",
                "agentic_commerce_intent_attack_family",
                "weakest_phase_aware_promotion_gate",
            ]
        },
        "live_feasibility": {
            "deployment_pattern": "fast_guardrail_inline_qwen_investigation_async",
            "transaction_latency_budget_ms": 300,
            "pre_transaction_latency_budget_ms": 2000,
            "post_transaction_latency_budget_ms": 5000,
            "integration_contract": "sanitized_events_in_actions_evidence_and_audit_out",
            "model_endpoint": "openai_compatible_open_model_endpoint",
        },
    }


def round_submission_evaluation(
    *,
    attack_case: SimulationCase,
    attack_turns: list[BlueTurn],
    control_results: list[tuple[SimulationCase, list[BlueTurn]]],
    report: RefereeReport,
    round_duration_ms: int | None = None,
    case_parallelism: int = 1,
) -> dict[str, Any]:
    events = attack_case.events
    serialized_events = [event.to_dict() for event in events]
    event_count = len(events)
    production_event_rate = (
        sum(event.event_type not in LAB_ONLY_EVENT_TYPES for event in events) / event_count
        if event_count
        else 0.0
    )
    delivery_metadata_rate = (
        sum(
            bool(event.source_system and event.decision_lane and event.latency_budget_ms)
            for event in events
        )
        / event_count
        if event_count
        else 0.0
    )
    observable_signal_rate = (
        sum(bool(event.observable_signals) for event in events) / event_count
        if event_count
        else 0.0
    )
    truth_boundary_clean = not bool(SEALED_KEYS.intersection(_nested_keys(serialized_events)))
    control_coverage = min(1.0, len(control_results) / 3.0)
    fidelity_score = 100.0 * (
        0.25 * production_event_rate
        + 0.20 * delivery_metadata_rate
        + 0.20 * observable_signal_rate
        + 0.20 * float(truth_boundary_clean)
        + 0.15 * control_coverage
    )

    all_blue_turns = [
        *attack_turns,
        *(turn for _, turns in control_results for turn in turns),
    ]
    model_latencies = sorted(
        float(call.get("latency_ms", 0) or 0)
        for turn in all_blue_turns
        for call in turn.model_calls
    )
    p95_index = max(0, math.ceil(0.95 * len(model_latencies)) - 1)
    model_p95 = model_latencies[p95_index] if model_latencies else None
    fast_path_coverage = (
        sum(bool(turn.risk_synthesis) for turn in attack_turns) / len(attack_turns)
        if attack_turns
        else 0.0
    )
    guard_interventions = sum(
        any("pre-model sequence-risk floor" in adjustment for adjustment in turn.policy_adjustments)
        for turn in attack_turns
    )
    guard_actionable_events = sum(
        turn.risk_synthesis.get(
            "operational_minimum_action", turn.risk_synthesis.get("minimum_action")
        )
        in {"step_up", "hold", "block"}
        for turn in attack_turns
    )
    evidence_citation_count = sum(
        len(turn.decision.evidence_refs) for turn in attack_turns
    )
    evidence_packet_count = sum(len(turn.evidence) for turn in attack_turns)
    blue_model_call_count = sum(len(turn.model_calls) for turn in all_blue_turns)
    repair_call_count = sum(
        call.get("agent_name") == "blue_event_repair"
        for turn in all_blue_turns
        for call in turn.model_calls
    )
    verified_context_fast_exits = sum(
        not turn.model_calls and "legitimate_context" in turn.decision.reason_codes
        for turn in all_blue_turns
    )

    return {
        "fidelity": {
            "score": round(fidelity_score, 2),
            "production_event_rate": round(production_event_rate, 4),
            "delivery_metadata_rate": round(delivery_metadata_rate, 4),
            "observable_signal_rate": round(observable_signal_rate, 4),
            "truth_boundary_clean": truth_boundary_clean,
            "legitimate_look_alike_cases": len(control_results),
            "lab_only_event_types_exposed": sorted(
                {event.event_type for event in events}.intersection(LAB_ONLY_EVENT_TYPES)
            ),
        },
        "detection_efficacy": {
            "outcome": report.outcome,
            "value_prevented_ratio": report.value_prevented_ratio,
            "realized_impact_ratio": report.realized_impact_ratio,
            "hard_false_positive_rate": report.hard_false_positive_rate,
            "legitimate_friction_rate": report.legitimate_friction_rate,
            "time_to_detect_seconds": report.time_to_detect_seconds,
            "balanced_lifecycle_defense_score": report.balanced_lifecycle_defense_score,
            "worst_phase_score": report.worst_phase_score,
            "evidence_citation_rate": round(
                min(1.0, evidence_citation_count / evidence_packet_count)
                if evidence_packet_count
                else 0.0,
                4,
            ),
            "fast_guard_intervention_count": guard_interventions,
            "fast_guard_actionable_event_count": guard_actionable_events,
        },
        "live_feasibility": {
            "event_contract_coverage": round(delivery_metadata_rate, 4),
            "pre_model_fast_path_coverage": round(fast_path_coverage, 4),
            "model_call_p95_ms": round(model_p95, 2) if model_p95 is not None else None,
            "blue_model_call_count": blue_model_call_count,
            "blue_event_count": len(all_blue_turns),
            "model_calls_per_blue_event": round(
                blue_model_call_count / len(all_blue_turns), 3
            )
            if all_blue_turns
            else 0.0,
            "policy_repair_call_count": repair_call_count,
            "verified_context_fast_exit_count": verified_context_fast_exits,
            "round_duration_ms": round_duration_ms,
            "case_parallelism": case_parallelism,
            "inline_transaction_budget_ms": 300,
            "recommended_runtime": "fast_guardrail_inline_qwen_investigation_async",
            "qwen_is_not_claimed_as_a_300ms_inline_dependency": True,
        },
    }
