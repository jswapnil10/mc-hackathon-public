"""Compile safe scenario stages into label-separated, deterministic event streams."""

from __future__ import annotations

import copy
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from red_team_agent.models import ScenarioSpec

from .contracts import ObservedEvent, SimulationCase, TruthRecord


BASELINE_AMOUNT_INR = 2_500.0
PAYMENT_EVENT_TYPES = {
    "PAYMENT_INITIATED",
    "AGENTIC_PAYMENT_INITIATED",
    "PAYMENT_REPEATED",
    "PAYOUT_REQUESTED",
    "PAYOUT_SETTLED",
    "DISPUTE_REFUND_ISSUED",
}
PRE_TRANSACTION_EVENT_TYPES = {
    "COMMUNICATION_RISK_CONTEXT",
    "SESSION_STARTED",
    "AUTHENTICATION_CONTEXT_CHANGED",
    "IDENTITY_APPLICATION_SUBMITTED",
    "IDENTITY_VERIFICATION_ATTEMPTED",
    "ACCOUNT_OPENED",
    "ACCOUNT_WARMUP_ACTIVITY",
    "ACCOUNT_BEHAVIOR_PROFILE_UPDATED",
    "BENEFICIARY_ADDED",
    "SUPPLIER_PROFILE_CHANGED",
    "PAYOUT_DESTINATION_CHANGED",
    "AGENT_COMMERCE_SESSION_STARTED",
    "AGENT_PAYMENT_INTENT_PRESENTED",
}
POST_TRANSACTION_EVENT_TYPES = {
    "FUNDS_RECEIVED",
    "FUNDS_DISPERSED",
    "PAYOUT_SETTLED",
    "DISPUTE_OPENED",
    "DISPUTE_EVIDENCE_REVIEWED",
    "DISPUTE_REFUND_ISSUED",
}


def event_lifecycle_phase(event_type: str) -> str:
    if event_type in PRE_TRANSACTION_EVENT_TYPES:
        return "pre_transaction"
    if event_type in POST_TRANSACTION_EVENT_TYPES:
        return "post_transaction"
    return "transaction"


def event_delivery_profile(event_type: str) -> tuple[str, str, int]:
    """Map a synthetic event to the production-style stream and decision budget it represents."""
    if event_type in {"COMMUNICATION_RISK_CONTEXT"}:
        source = "channel_risk_gateway"
    elif event_type in {
        "IDENTITY_APPLICATION_SUBMITTED",
        "IDENTITY_VERIFICATION_ATTEMPTED",
        "ACCOUNT_OPENED",
    }:
        source = "identity_verification_stream"
    elif event_type in {
        "SESSION_STARTED",
        "AUTHENTICATION_CONTEXT_CHANGED",
        "ACCOUNT_BEHAVIOR_PROFILE_UPDATED",
    }:
        source = "authentication_behavior_stream"
    elif event_type in {"AGENT_COMMERCE_SESSION_STARTED", "AGENT_PAYMENT_INTENT_PRESENTED"}:
        source = "agentic_commerce_trust_gateway"
    elif event_type.startswith("DISPUTE_"):
        source = "dispute_case_stream"
    elif event_type in {"FUNDS_RECEIVED", "FUNDS_DISPERSED", "PAYOUT_SETTLED"}:
        source = "payment_network_stream"
    elif event_type in {
        "BENEFICIARY_ADDED",
        "SUPPLIER_PROFILE_CHANGED",
        "PAYOUT_DESTINATION_CHANGED",
        "SALES_VELOCITY_CHANGED",
    }:
        source = "account_profile_stream"
    else:
        source = "payment_orchestration_stream"

    phase = event_lifecycle_phase(event_type)
    if phase == "pre_transaction":
        return source, "asynchronous_pre_transaction", 2000
    if phase == "post_transaction":
        return source, "streaming_post_transaction", 5000
    return source, "inline_payment_decision", 300


def _stable_id(*parts: object) -> str:
    return f"syn_evt_{uuid.uuid5(uuid.NAMESPACE_URL, '|'.join(map(str, parts))).hex[:20]}"


def _materialize_attributes(
    attributes: dict[str, Any], *, rng: random.Random, event_type: str
) -> dict[str, Any]:
    materialized = copy.deepcopy(attributes)
    materialized.pop("planner_emphasis", None)
    probability_fields = {
        "new_device_probability": "device_is_new",
        "new_network_probability": "network_is_new",
        "shared_device_probability": "device_shared_across_accounts",
        "shared_network_probability": "network_shared_across_accounts",
        "device_reuse_probability": "device_reused_across_profiles",
        "bot_behavior_probability": "behavior_automation_suspected",
        "identity_mismatch_probability": "identity_consistency_mismatch",
        "evidence_conflict_probability": "evidence_conflict_present",
        "agent_signature_valid_probability": "agent_signature_valid",
        "consumer_consent_valid_probability": "consumer_consent_valid",
        "intent_scope_match_probability": "intent_scope_match",
        "payment_container_match_probability": "payment_container_match",
        "merchant_scope_match_probability": "merchant_scope_match",
    }
    for source, target in probability_fields.items():
        if source in materialized:
            materialized[target] = rng.random() < float(materialized.pop(source))
    multiplier = float(materialized.pop("amount_multiplier", 1.0))
    if event_type in PAYMENT_EVENT_TYPES or event_type in {"FUNDS_RECEIVED", "FUNDS_DISPERSED"}:
        materialized["sender_baseline_amount_inr"] = BASELINE_AMOUNT_INR
        materialized["amount_inr"] = round(BASELINE_AMOUNT_INR * multiplier, 2)
    return materialized


def _value_at_risk(scenario: ScenarioSpec, stage_id: str, event_type: str, attributes: dict[str, Any]) -> float:
    amount = float(attributes.get("amount_inr", 0.0))
    if scenario.attack_family == "DISPUTE-01" and stage_id == "original_purchase":
        return 0.0
    if event_type in {"PAYMENT_INITIATED", "AGENTIC_PAYMENT_INITIATED"}:
        return amount
    if event_type == "PAYMENT_REPEATED":
        count = max(1, int(attributes.get("payment_count", scenario.parameters.get("payment_count", 2))))
        return amount * count
    if scenario.attack_family == "MULE-01" and stage_id == "fan_in":
        return amount * int(scenario.parameters.get("sender_count", 1))
    if event_type in {"PAYOUT_REQUESTED", "DISPUTE_REFUND_ISSUED"}:
        return amount
    return 0.0


def simulate_attack(scenario: ScenarioSpec) -> SimulationCase:
    rng = random.Random(scenario.seed)
    start = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc) + timedelta(
        minutes=scenario.seed % 1440
    )
    events: list[ObservedEvent] = []
    truth: list[TruthRecord] = []
    for stage in scenario.stages:
        event_id = _stable_id(scenario.scenario_id, stage.stage_id, stage.sequence)
        attributes = _materialize_attributes(stage.attributes, rng=rng, event_type=stage.event_type)
        source_system, decision_lane, latency_budget_ms = event_delivery_profile(stage.event_type)
        events.append(
            ObservedEvent(
                event_id=event_id,
                sequence=stage.sequence,
                occurred_at=(start + timedelta(seconds=stage.offset_seconds)).isoformat(),
                lifecycle_phase=stage.lifecycle_phase,
                event_type=stage.event_type,
                observable_signals=list(stage.observable_signals),
                attributes=attributes,
                source_system=source_system,
                decision_lane=decision_lane,
                latency_budget_ms=latency_budget_ms,
            )
        )
        truth.append(
            TruthRecord(
                event_id=event_id,
                stage_id=stage.stage_id,
                attack_family=scenario.attack_family,
                scenario_id=scenario.scenario_id,
                intervention_point=stage.blue_intervention_point,
                is_attack=True,
                value_at_risk_inr=round(
                    _value_at_risk(scenario, stage.stage_id, stage.event_type, attributes), 2
                ),
                offset_seconds=stage.offset_seconds,
            )
        )
    return SimulationCase(case_id=scenario.scenario_id, events=events, truth=truth)


def _control_shape(name: str, index: int) -> list[tuple[str, list[str], dict[str, Any]]]:
    sender = f"syn_control_sender_{index:02d}"
    beneficiary = f"syn_control_beneficiary_{index:02d}"
    common = {"sender_account_id": sender, "beneficiary_id": beneficiary}
    if "agent" in name:
        agent = f"syn_control_agent_{index:02d}"
        merchant = f"syn_control_merchant_{index:02d}"
        return [
            (
                "AGENT_COMMERCE_SESSION_STARTED",
                ["agent_registration_status", "message_signature_validity", "request_freshness"],
                {
                    "agent_id": agent,
                    "merchant_id": merchant,
                    "agent_signature_valid": True,
                    "request_age_seconds": 12,
                    "nonce_reuse_count": 0,
                    "registered_agent_age_days": 420,
                },
            ),
            (
                "AGENT_PAYMENT_INTENT_PRESENTED",
                ["consumer_consent_validity", "intent_scope_match", "basket_deviation"],
                {
                    "agent_id": agent,
                    "merchant_id": merchant,
                    "consumer_consent_valid": True,
                    "intent_scope_match": True,
                    "basket_value_multiplier": 1.02,
                    "customer_confirmation_complete": True,
                },
            ),
            (
                "AGENTIC_PAYMENT_INITIATED",
                ["payment_container_match", "merchant_scope_match", "amount_vs_verified_intent"],
                {
                    "agent_id": agent,
                    "merchant_id": merchant,
                    "beneficiary_id": beneficiary,
                    "payment_container_match": True,
                    "merchant_scope_match": True,
                    "sender_baseline_amount_inr": 2500.0,
                    "amount_inr": 2550.0,
                    "strong_authentication_passed": True,
                },
            ),
        ]
    if any(token in name for token in ("bank_account_change", "sales_spike", "merchant_payout")):
        merchant = f"syn_control_merchant_{index:02d}"
        return [
            (
                "PAYOUT_DESTINATION_CHANGED",
                ["destination_novelty", "profile_change_recency"],
                {
                    "merchant_id": merchant,
                    "beneficiary_id": beneficiary,
                    "bank_change_recency_days": 30,
                    "out_of_band_verification_complete": True,
                    "registered_entity_age_days": 1250,
                },
            ),
            (
                "PAYOUT_REQUESTED",
                ["payout_amount_vs_baseline", "new_destination"],
                {
                    "merchant_id": merchant,
                    "beneficiary_id": beneficiary,
                    "sender_baseline_amount_inr": 55000.0,
                    "amount_inr": 57500.0,
                    "customer_confirmation_complete": True,
                    "scheduled_payout": True,
                },
            ),
        ]
    if any(token in name for token in ("goods_not_received", "duplicate_charge", "service_failure")):
        merchant = f"syn_control_merchant_{index:02d}"
        return [
            (
                "DISPUTE_OPENED",
                ["dispute_age", "linked_dispute_volume"],
                {
                    "sender_account_id": sender,
                    "merchant_id": merchant,
                    "dispute_age_hours": 168,
                    "linked_dispute_count": 1,
                    "customer_confirmation_complete": True,
                },
            ),
            (
                "DISPUTE_EVIDENCE_REVIEWED",
                ["evidence_consistency", "independent_case_checks"],
                {
                    "sender_account_id": sender,
                    "merchant_id": merchant,
                    "evidence_conflict_present": False,
                    "independent_checks_passed": 4,
                },
            ),
        ]
    if any(token in name for token in ("supplier", "invoice", "payroll", "tax_run")):
        return [
            (
                "SUPPLIER_PROFILE_CHANGED",
                ["supplier_profile_change", "beneficiary_novelty"],
                {
                    **common,
                    "supplier_relationship_days": 920,
                    "out_of_band_verification_complete": True,
                    "approval_step_count": 3,
                },
            ),
            (
                "PAYMENT_INITIATED",
                ["amount_vs_sender_baseline", "recent_supplier_change"],
                {
                    **common,
                    "sender_baseline_amount_inr": 42000.0,
                    "amount_inr": 44500.0,
                    "beneficiary_age_days": 480,
                    "dual_approval_complete": True,
                },
            ),
        ]
    if any(token in name for token in ("household", "crowdfunding", "merchant", "aggregation")):
        return [
            (
                "FUNDS_RECEIVED",
                ["receiver_fan_in", "shared_infrastructure"],
                {
                    **common,
                    "registered_entity_age_days": 1180,
                    "declared_business_purpose": "registered_aggregation",
                    "sender_count": 18,
                    "unique_device_count": 17,
                },
            ),
            (
                "PAYMENT_REPEATED",
                ["receiver_value_velocity", "fan_in_count"],
                {
                    **common,
                    "sender_baseline_amount_inr": 1500.0,
                    "amount_inr": 1450.0,
                    "payment_count": 8,
                    "settlement_pattern_seen_days": 210,
                },
            ),
        ]
    if any(token in name for token in ("thin_file", "student", "applicant", "verification_retry")):
        return [
            (
                "IDENTITY_VERIFICATION_ATTEMPTED",
                ["verification_retry_count", "identity_linkage"],
                {
                    "identity_id": f"syn_control_identity_{index:02d}",
                    "verification_retry_count": 2,
                    "independent_checks_passed": 4,
                    "linked_attribute_count": 0,
                    "accessibility_path_used": "accessibility" in name,
                },
            ),
            (
                "ACCOUNT_OPENED",
                ["thin_file", "device_reuse"],
                {
                    "sender_account_id": sender,
                    "device_reused_across_profiles": False,
                    "verified_funding_source": True,
                    "account_age_days": 0,
                },
            ),
        ]
    if any(token in name for token in ("device", "migration", "travelling", "phone")):
        return [
            (
                "SESSION_STARTED",
                ["device_novelty", "network_novelty"],
                {
                    "sender_account_id": sender,
                    "device_id": f"syn_control_device_{index:02d}",
                    "device_is_new": True,
                    "network_is_new": "travelling" in name,
                    "customer_notified_change": True,
                    "strong_authentication_passed": True,
                },
            ),
            (
                "PAYMENT_INITIATED",
                ["amount_vs_sender_baseline", "beneficiary_novelty"],
                {
                    **common,
                    "sender_baseline_amount_inr": 2500.0,
                    "amount_inr": 2650.0,
                    "beneficiary_age_days": 320,
                    "customer_confirmation_complete": True,
                },
            ),
        ]
    if "new_beneficiary" in name:
        return [
            (
                "BENEFICIARY_ADDED",
                ["beneficiary_novelty", "same_session_add_and_pay"],
                {
                    **common,
                    "beneficiary_age_days": 0,
                    "customer_confirmation_complete": True,
                    "known_relationship_days": 540,
                },
            ),
            (
                "PAYMENT_INITIATED",
                ["amount_vs_sender_baseline", "beneficiary_novelty"],
                {
                    **common,
                    "sender_baseline_amount_inr": 2500.0,
                    "amount_inr": 2600.0,
                    "beneficiary_age_days": 0,
                    "strong_authentication_passed": True,
                    "customer_confirmation_complete": True,
                },
            ),
        ]
    if any(token in name for token in ("recurring", "installment", "gradual")):
        return [
            (
                "PAYMENT_REPEATED",
                ["repeat_payment_count", "payment_interval"],
                {
                    **common,
                    "sender_baseline_amount_inr": 4000.0,
                    "amount_inr": 3950.0,
                    "payment_count": 3,
                    "recurring_mandate_age_days": 240,
                    "beneficiary_age_days": 640,
                },
            )
        ]
    return [
        (
            "COMMUNICATION_RISK_CONTEXT",
            ["declared_urgency_band", "first_contact_recency"],
            {
                "sender_account_id": sender,
                "urgency_level": 4,
                "known_contact_relationship_days": 720,
                "customer_confirmation_complete": True,
                "content_available": False,
            },
        ),
        (
            "PAYMENT_INITIATED",
            ["amount_vs_sender_baseline", "beneficiary_novelty"],
            {
                **common,
                "sender_baseline_amount_inr": 2500.0,
                "amount_inr": 2750.0,
                "beneficiary_age_days": 365,
                "strong_authentication_passed": True,
            },
        ),
    ]


def simulate_legitimate_controls(scenario: ScenarioSpec, control_names: list[str]) -> list[SimulationCase]:
    base = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
    cases: list[SimulationCase] = []
    for index, name in enumerate(control_names, start=1):
        case_id = f"CTRL-{scenario.attack_family}-{index:02d}-{scenario.seed}"
        events: list[ObservedEvent] = []
        truth: list[TruthRecord] = []
        for sequence, (event_type, signals, attributes) in enumerate(_control_shape(name, index), start=1):
            event_id = _stable_id(case_id, sequence)
            offset = (sequence - 1) * 120
            source_system, decision_lane, latency_budget_ms = event_delivery_profile(event_type)
            events.append(
                ObservedEvent(
                    event_id=event_id,
                    sequence=sequence,
                    occurred_at=(base + timedelta(minutes=index * 10, seconds=offset)).isoformat(),
                    lifecycle_phase=event_lifecycle_phase(event_type),
                    event_type=event_type,
                    observable_signals=signals,
                    attributes=attributes,
                    source_system=source_system,
                    decision_lane=decision_lane,
                    latency_budget_ms=latency_budget_ms,
                )
            )
            truth.append(
                TruthRecord(
                    event_id=event_id,
                    stage_id=f"control_stage_{sequence}",
                    attack_family=None,
                    scenario_id=case_id,
                    intervention_point="DECIDE",
                    is_attack=False,
                    value_at_risk_inr=0.0,
                    offset_seconds=offset,
                )
            )
        cases.append(SimulationCase(case_id=case_id, events=events, truth=truth, control_name=name))
    return cases
