"""Deterministic, read-only tools that Blue may choose to call."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable

from .contracts import EVIDENCE_TOOLS, EvidencePacket, ObservedEvent


VALUE_MOVING_EVENT_TYPES = {
    "PAYMENT_INITIATED",
    "AGENTIC_PAYMENT_INITIATED",
    "PAYMENT_REPEATED",
    "FUNDS_RECEIVED",
    "FUNDS_DISPERSED",
    "PAYOUT_REQUESTED",
    "PAYOUT_SETTLED",
    "DISPUTE_REFUND_ISSUED",
}
STRONG_BOOLEAN_CONTEXT = {
    "out_of_band_verification_complete",
    "dual_approval_complete",
    "customer_notified_change",
    "strong_authentication_passed",
    "customer_confirmation_complete",
    "verified_funding_source",
}
HISTORICAL_CONTEXT_DAYS = {
    "registered_entity_age_days": 30,
    "settlement_pattern_seen_days": 30,
    "recurring_mandate_age_days": 30,
    "known_contact_relationship_days": 30,
    "supplier_relationship_days": 30,
    "registered_agent_age_days": 30,
}
AGENT_TRUST_FACTORS = {
    "agent_signature_valid",
    "consumer_consent_valid",
    "intent_scope_match",
    "payment_container_match",
    "merchant_scope_match",
}


def _legitimate_context_summary(events: list[ObservedEvent]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    verified_markers: set[str] = set()
    agent_factors: dict[str, bool] = {}
    for event in events:
        context: dict[str, Any] = {}
        for key in STRONG_BOOLEAN_CONTEXT:
            if event.attributes.get(key) is True:
                context[key] = True
                verified_markers.add(key)
        if float(event.attributes.get("independent_checks_passed", 0) or 0) > 0:
            context["independent_checks_passed"] = event.attributes["independent_checks_passed"]
            verified_markers.add("independent_checks_passed")
        for key, minimum_days in HISTORICAL_CONTEXT_DAYS.items():
            value = event.attributes.get(key)
            if isinstance(value, (int, float)) and value >= minimum_days:
                context[key] = value
                verified_markers.add(key)
        for key in AGENT_TRUST_FACTORS:
            if key in event.attributes:
                agent_factors[key] = event.attributes[key] is True
                context[key] = event.attributes[key]
        if context:
            observations.append({"event_id": event.event_id, "context": context})

    complete_agent_trust = bool(agent_factors) and AGENT_TRUST_FACTORS.issubset(agent_factors) and all(
        agent_factors[key] for key in AGENT_TRUST_FACTORS
    )
    independently_verified = bool(verified_markers) or complete_agent_trust
    if complete_agent_trust and verified_markers:
        strength = "corroborated"
    elif independently_verified:
        strength = "verified"
    else:
        strength = "unverified"
    return {
        "supporting_context_found": independently_verified,
        "independent_verification_found": independently_verified,
        "verification_strength": strength,
        "verified_markers": sorted(verified_markers),
        "agent_trust_factors": agent_factors,
        "complete_agent_trust_chain": complete_agent_trust,
        "supporting_observations": observations,
        "warning": (
            "Same-case repetition and internally consistent identifiers do not prove an established "
            "relationship. Only explicit verification or pre-existing history can resolve risk."
        ),
    }


def synthesize_case_risk(events: list[ObservedEvent]) -> dict[str, Any]:
    """Fast, observable-only sequence guard used before the GenAI investigation."""
    indicators: list[dict[str, Any]] = []

    def add(code: str, points: int, detail: str) -> None:
        if not any(item["code"] == code for item in indicators):
            indicators.append({"code": code, "points": points, "detail": detail})

    event_types = [event.event_type for event in events]
    phases = list(dict.fromkeys(event.lifecycle_phase for event in events))
    current = events[-1]
    current_attributes = current.attributes
    value_events = [event for event in events if event.event_type in VALUE_MOVING_EVENT_TYPES]

    if len(phases) >= 2:
        add("cross_phase_progression", 1, f"The case has progressed across {len(phases)} lifecycle phases.")
    if len(value_events) >= 2:
        add("cumulative_value_sequence", 2, "Multiple value-moving events are linked in the visible case.")
    if current.event_type == "PAYMENT_REPEATED" or int(current_attributes.get("payment_count", 1) or 1) > 1:
        add("repeated_value_movement", 2, "The current sequence repeats or batches value movement.")
    authorized_counts = [
        int(event.attributes["authorized_payment_count"])
        for event in events
        if isinstance(event.attributes.get("authorized_payment_count"), (int, float))
    ]
    actual_counts = [
        int(event.attributes["payment_count"])
        for event in events
        if isinstance(event.attributes.get("payment_count"), (int, float))
    ]
    if authorized_counts and actual_counts and max(actual_counts) > max(authorized_counts):
        add("agent_intent_count_mismatch", 4, "Observed payment count exceeds the signed intent scope.")
    if current.event_type in {"FUNDS_DISPERSED", "PAYOUT_SETTLED", "DISPUTE_REFUND_ISSUED"}:
        add("downstream_value_release", 2, "The case has reached a downstream release or dispersal point.")

    context_changes = {
        "AUTHENTICATION_CONTEXT_CHANGED",
        "BENEFICIARY_ADDED",
        "SUPPLIER_PROFILE_CHANGED",
        "PAYOUT_DESTINATION_CHANGED",
    }
    if context_changes.intersection(event_types):
        add("sensitive_context_change", 1, "A security, beneficiary, supplier, or payout context changed.")
    if context_changes.intersection(event_types) and value_events:
        add("context_change_before_value", 2, "A sensitive context change preceded value movement.")

    if any(event.attributes.get("device_is_new") is True for event in events):
        add("new_device", 1, "A newly observed device is present.")
    if any(event.attributes.get("network_is_new") is True for event in events):
        add("new_network", 1, "A newly observed network is present.")
    if any(
        event.attributes.get(key) is True
        for event in events
        for key in ("device_shared_across_accounts", "network_shared_across_accounts", "device_reused_across_profiles")
    ):
        add("shared_infrastructure", 2, "Infrastructure is shared across accounts or profiles.")
    if any(float(event.attributes.get("urgency_level", 0) or 0) >= 7 for event in events):
        add("communication_pressure", 2, "High declared urgency is present in channel metadata.")
    if any(int(event.attributes.get("channel_switch_count", 0) or 0) >= 2 for event in events):
        add("channel_switching", 1, "The interaction moved across multiple channels.")
    if any(
        isinstance(event.attributes.get("beneficiary_age_days"), (int, float))
        and event.attributes["beneficiary_age_days"] <= 7
        for event in events
    ):
        add("new_beneficiary", 2, "A newly introduced beneficiary is linked to the sequence.")

    amount_ratios: list[float] = []
    for event in value_events:
        amount = float(event.attributes.get("amount_inr", 0) or 0)
        baseline = float(event.attributes.get("sender_baseline_amount_inr", amount or 1) or 1)
        amount_ratios.append(amount / max(1.0, baseline))
    max_ratio = max(amount_ratios, default=1.0)
    if max_ratio >= 3:
        add("extreme_amount_deviation", 3, f"Value reached {max_ratio:.2f}× the available baseline.")
    elif max_ratio >= 1.5:
        add("amount_deviation", 2, f"Value reached {max_ratio:.2f}× the available baseline.")
    elif max_ratio >= 1.15:
        add("subtle_amount_deviation", 1, f"Value reached {max_ratio:.2f}× the available baseline.")

    if any(int(event.attributes.get("linked_attribute_count", 0) or 0) >= 3 for event in events):
        add("dense_identity_linkage", 2, "Several identity attributes link across profiles.")
    if any(int(event.attributes.get("verification_retry_count", 0) or 0) >= 3 for event in events):
        add("verification_retries", 1, "Repeated identity verification attempts are present.")
    for score_key in ("profile_consistency_score", "document_consistency_score", "liveness_confidence"):
        if any(
            isinstance(event.attributes.get(score_key), (int, float))
            and event.attributes[score_key] < 0.6
            for event in events
        ):
            add("identity_consistency_gap", 2, f"At least one {score_key} observation is weak.")
    if any(event.attributes.get("behavior_automation_suspected") is True for event in events):
        add("automation_signal", 1, "Behavioral telemetry indicates possible automation.")
    if any(event.attributes.get("evidence_conflict_present") is True for event in events):
        add("evidence_conflict", 2, "Case evidence contains an explicit inconsistency.")
    if any(int(event.attributes.get("sender_count", 0) or 0) >= 4 for event in events):
        add("fan_in_concentration", 3, "Many senders converge on the observed receiving entity.")
    if any(float(event.attributes.get("outflow_ratio", 0) or 0) >= 0.5 for event in events):
        add("rapid_outflow", 2, "A material portion of received value moves onward.")
    if any(float(event.attributes.get("destination_novelty_score", 0) or 0) >= 0.6 for event in events):
        add("novel_payout_destination", 2, "The payout destination has high novelty.")
    if any(float(event.attributes.get("sales_velocity_multiplier", 0) or 0) >= 2 for event in events):
        add("sales_velocity_shift", 2, "Sales velocity materially exceeds the observed baseline.")
    if current.event_type == "PAYOUT_REQUESTED" and current.attributes.get("accelerated_settlement") is True:
        add("accelerated_payout_after_change", 2, "An accelerated payout follows the visible profile sequence.")
    if any(int(event.attributes.get("linked_dispute_count", 0) or 0) >= 5 for event in events):
        add("linked_dispute_burst", 2, "Multiple linked disputes are present.")
    if {"DISPUTE_OPENED", "DISPUTE_EVIDENCE_REVIEWED"}.issubset(event_types):
        add("dispute_evidence_sequence", 2, "A dispute progressed through evidence review in this case.")
    if current.event_type == "DISPUTE_REFUND_ISSUED" and "DISPUTE_OPENED" in event_types:
        add("refund_after_dispute_sequence", 3, "Refund value is about to release after the observed dispute sequence.")
    if (
        {"IDENTITY_APPLICATION_SUBMITTED", "ACCOUNT_OPENED"}.issubset(event_types)
        and any(event.event_type in {"PAYMENT_INITIATED", "AGENTIC_PAYMENT_INITIATED"} for event in events)
    ):
        add("identity_to_payment_activation", 2, "An observed identity-onboarding sequence has reached payment activation.")

    agent_checks = {
        "agent_signature_valid": 3,
        "consumer_consent_valid": 3,
        "intent_scope_match": 3,
        "payment_container_match": 3,
        "merchant_scope_match": 2,
    }
    for key, points in agent_checks.items():
        if any(event.attributes.get(key) is False for event in events):
            add(f"{key}_failed", points, f"The agentic-commerce trust check {key} failed.")
    if any(float(event.attributes.get("request_age_seconds", 0) or 0) > 300 for event in events):
        add("stale_agent_request", 1, "The signed agent request is outside the fresh-interaction window.")
    if any(int(event.attributes.get("nonce_reuse_count", 0) or 0) > 0 for event in events):
        add("agent_request_replay", 2, "The agent request contains nonce reuse evidence.")
    if any(float(event.attributes.get("basket_value_multiplier", 1) or 1) > 1.25 for event in events):
        add("agent_intent_value_mismatch", 2, "Basket value materially exceeds the presented intent.")

    legitimacy = _legitimate_context_summary(events)
    raw_score = sum(item["points"] for item in indicators)
    verification_discount = 0
    if legitimacy["independent_verification_found"]:
        verification_discount = 4 + 2 * max(0, len(legitimacy["verified_markers"]) - 1)
        if legitimacy["complete_agent_trust_chain"]:
            verification_discount += 2
    guarded_score = max(0, raw_score - verification_discount)
    if legitimacy["independent_verification_found"] and guarded_score < 3:
        minimum_action = "allow"
    elif (
        current.lifecycle_phase == "transaction"
        and any(item["code"] == "agent_intent_count_mismatch" for item in indicators)
        and guarded_score >= 4
    ):
        minimum_action = "hold"
    elif current.lifecycle_phase == "transaction" and guarded_score >= 6:
        minimum_action = "hold"
    elif current.lifecycle_phase == "post_transaction" and guarded_score >= 8:
        minimum_action = "hold"
    elif guarded_score >= 3:
        minimum_action = "step_up"
    elif guarded_score >= 1:
        minimum_action = "monitor"
    else:
        minimum_action = "allow"
    risk_band = {
        "allow": "low",
        "monitor": "medium",
        "step_up": "high" if guarded_score >= 5 else "medium",
        "hold": "high",
    }[minimum_action]
    return {
        "observable_risk_score": guarded_score,
        "raw_risk_score": raw_score,
        "risk_band": risk_band,
        "minimum_action": minimum_action,
        "indicator_count": len(indicators),
        "indicators": indicators,
        "lifecycle_phases_seen": phases,
        "value_event_count": len(value_events),
        "independent_legitimate_context": legitimacy["independent_verification_found"],
        "verification_discount": verification_discount,
        "fast_path_role": "pre_model_minimum_action_guard",
        "uses_sealed_truth": False,
    }


def _timestamps(events: list[ObservedEvent]) -> list[datetime]:
    return [datetime.fromisoformat(event.occurred_at) for event in events]


def _timeline_summary(events: list[ObservedEvent]) -> dict[str, Any]:
    timestamps = _timestamps(events)
    return {
        "event_count": len(events),
        "event_type_counts": dict(Counter(event.event_type for event in events)),
        "ordered_event_types": [event.event_type for event in events],
        "ordered_lifecycle_phases": [event.lifecycle_phase for event in events],
        "lifecycle_phases_reached": list(dict.fromkeys(event.lifecycle_phase for event in events)),
        "value_event_count": sum(event.event_type in VALUE_MOVING_EVENT_TYPES for event in events),
        "elapsed_seconds": round((max(timestamps) - min(timestamps)).total_seconds()) if len(timestamps) > 1 else 0,
    }


def _entity_linkage(events: list[ObservedEvent]) -> dict[str, Any]:
    value_to_paths: dict[str, list[str]] = defaultdict(list)
    relationship_edges: list[dict[str, str]] = []
    for event in events:
        entities: list[tuple[str, str]] = []
        for key, value in event.attributes.items():
            if key.endswith("_id") and isinstance(value, str):
                value_to_paths[value].append(f"event_{event.sequence}.{key}")
                entities.append((key.removesuffix("_id"), value))
        for left_index, (left_kind, left_value) in enumerate(entities):
            for right_kind, right_value in entities[left_index + 1 :]:
                relationship_edges.append(
                    {
                        "source": left_value,
                        "source_kind": left_kind,
                        "target": right_value,
                        "target_kind": right_kind,
                        "event_id": event.event_id,
                    }
                )
    repeated = {value: paths for value, paths in value_to_paths.items() if len(paths) > 1}
    history = _legitimate_context_summary(events)
    explicit_linkage = [
        {
            "event_id": event.event_id,
            "linked_attribute_count": event.attributes.get("linked_attribute_count"),
            "device_reused_across_profiles": event.attributes.get("device_reused_across_profiles"),
            "device_shared_across_accounts": event.attributes.get("device_shared_across_accounts"),
            "network_shared_across_accounts": event.attributes.get("network_shared_across_accounts"),
        }
        for event in events
        if any(
            key in event.attributes
            for key in (
                "linked_attribute_count",
                "device_reused_across_profiles",
                "device_shared_across_accounts",
                "network_shared_across_accounts",
            )
        )
    ]
    return {
        "unique_entity_count": len(value_to_paths),
        "relationship_edge_count": len(relationship_edges),
        "same_case_relationship_edges": relationship_edges,
        "same_case_repeated_entity_paths": repeated,
        "historical_relationship_verified": history["independent_verification_found"],
        "historical_relationship_markers": history["verified_markers"],
        "explicit_linkage_observations": explicit_linkage,
        "interpretation_warning": (
            "These edges show continuity inside this case only. Repetition is not evidence of a "
            "pre-existing trusted relationship unless historical markers are present."
        ),
    }


def _velocity_profile(events: list[ObservedEvent]) -> dict[str, Any]:
    timestamps = _timestamps(events)
    elapsed = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
    value_events = [
        event
        for event in events
        if event.event_type
        in {
            "PAYMENT_INITIATED",
            "AGENTIC_PAYMENT_INITIATED",
            "PAYMENT_REPEATED",
            "FUNDS_RECEIVED",
            "FUNDS_DISPERSED",
            "PAYOUT_REQUESTED",
            "PAYOUT_SETTLED",
            "DISPUTE_REFUND_ISSUED",
        }
    ]
    amounts = [float(event.attributes.get("amount_inr", 0.0)) for event in value_events]
    return {
        "elapsed_seconds": round(elapsed),
        "value_event_count": len(value_events),
        "declared_payment_count": sum(int(event.attributes.get("payment_count", 1)) for event in value_events),
        "observed_value_inr": round(sum(amounts), 2),
        "events_per_minute": round(len(events) / max(elapsed / 60.0, 1.0), 3),
    }


def _payment_context(events: list[ObservedEvent]) -> dict[str, Any]:
    payments: list[dict[str, Any]] = []
    for event in events:
        if "amount_inr" not in event.attributes:
            continue
        amount = float(event.attributes["amount_inr"])
        baseline = float(event.attributes.get("sender_baseline_amount_inr", amount or 1.0))
        payments.append(
            {
                "event_id": event.event_id,
                "amount_inr": amount,
                "baseline_amount_inr": baseline,
                "amount_to_baseline_ratio": round(amount / max(baseline, 1.0), 3),
                "beneficiary_age_days": event.attributes.get("beneficiary_age_days"),
                "payment_count": event.attributes.get("payment_count", 1),
            }
        )
    return {"payments": payments, "payment_count": len(payments)}


def _legitimate_alternatives(events: list[ObservedEvent]) -> dict[str, Any]:
    return _legitimate_context_summary(events)


def _behavioral_biometrics(events: list[ObservedEvent]) -> dict[str, Any]:
    behavioral_keys = {
        "behavior_automation_suspected",
        "typing_cadence_similarity",
        "mouse_movement_humanness",
        "liveness_consistency_score",
        "liveness_confidence",
        "identity_consistency_mismatch",
        "voice_session_anomaly",
        "verification_retry_count",
    }
    observations = [
        {
            "event_id": event.event_id,
            "signals": {
                key: value for key, value in event.attributes.items() if key in behavioral_keys
            },
        }
        for event in events
        if behavioral_keys.intersection(event.attributes)
    ]
    return {
        "behavioral_observation_count": len(observations),
        "observations": observations,
        "warning": "Behavioral indicators are probabilistic evidence, not identity truth.",
    }


def _communication_risk(events: list[ObservedEvent]) -> dict[str, Any]:
    communication_keys = {
        "urgency_level",
        "channel_switch_count",
        "first_contact_recency",
        "known_contact_relationship_days",
        "content_available",
    }
    observations = [
        {
            "event_id": event.event_id,
            "signals": {
                key: value for key, value in event.attributes.items() if key in communication_keys
            },
        }
        for event in events
        if event.event_type == "COMMUNICATION_RISK_CONTEXT"
        or communication_keys.intersection(event.attributes)
    ]
    return {
        "communication_observation_count": len(observations),
        "observations": observations,
        "content_inspected": False,
    }


def _evidence_quality(events: list[ObservedEvent]) -> dict[str, Any]:
    quality_keys = {
        "signal_confidence",
        "source_reliability",
        "evidence_delay_seconds",
        "evidence_conflict_present",
        "document_consistency_score",
        "profile_consistency_score",
    }
    observations = [
        {
            "event_id": event.event_id,
            "quality": {key: value for key, value in event.attributes.items() if key in quality_keys},
        }
        for event in events
        if quality_keys.intersection(event.attributes)
    ]
    conflict_count = sum(
        bool(item["quality"].get("evidence_conflict_present")) for item in observations
    )
    return {
        "quality_metadata_event_count": len(observations),
        "conflicting_evidence_count": conflict_count,
        "observations": observations,
        "default_assumption": "Missing quality metadata is treated as unknown, not trustworthy.",
    }


TOOL_IMPLEMENTATIONS: dict[str, Callable[[list[ObservedEvent]], dict[str, Any]]] = {
    "timeline_summary": _timeline_summary,
    "entity_linkage": _entity_linkage,
    "velocity_profile": _velocity_profile,
    "payment_context": _payment_context,
    "legitimate_alternatives": _legitimate_alternatives,
    "behavioral_biometrics": _behavioral_biometrics,
    "communication_risk": _communication_risk,
    "evidence_quality": _evidence_quality,
    "case_risk_synthesis": synthesize_case_risk,
}


class EvidenceWorkbench:
    def run(self, tool_names: list[str], events: list[ObservedEvent]) -> list[EvidencePacket]:
        if not events:
            raise ValueError("Evidence tools require at least one visible event.")
        unknown = set(tool_names).difference(EVIDENCE_TOOLS)
        if unknown:
            raise ValueError(f"Blue requested unknown evidence tools: {sorted(unknown)}")
        packets: list[EvidencePacket] = []
        for index, tool_name in enumerate(dict.fromkeys(tool_names), start=1):
            packets.append(
                EvidencePacket(
                    evidence_id=f"evidence_{events[-1].event_id[-8:]}_{index}",
                    tool_name=tool_name,
                    facts=TOOL_IMPLEMENTATIONS[tool_name](events),
                    source="synthetic_observable_history",
                    as_of_event_id=events[-1].event_id,
                    confidence=0.9,
                )
            )
        return packets
