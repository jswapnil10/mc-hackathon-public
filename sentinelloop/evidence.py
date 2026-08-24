"""Deterministic, read-only tools that Blue may choose to call."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable

from .contracts import EVIDENCE_TOOLS, EvidencePacket, ObservedEvent


def _timestamps(events: list[ObservedEvent]) -> list[datetime]:
    return [datetime.fromisoformat(event.occurred_at) for event in events]


def _timeline_summary(events: list[ObservedEvent]) -> dict[str, Any]:
    timestamps = _timestamps(events)
    return {
        "event_count": len(events),
        "event_type_counts": dict(Counter(event.event_type for event in events)),
        "ordered_event_types": [event.event_type for event in events],
        "elapsed_seconds": round((max(timestamps) - min(timestamps)).total_seconds()) if len(timestamps) > 1 else 0,
    }


def _entity_linkage(events: list[ObservedEvent]) -> dict[str, Any]:
    value_to_paths: dict[str, list[str]] = defaultdict(list)
    for event in events:
        for key, value in event.attributes.items():
            if key.endswith("_id") and isinstance(value, str):
                value_to_paths[value].append(f"event_{event.sequence}.{key}")
    repeated = {value: paths for value, paths in value_to_paths.items() if len(paths) > 1}
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
        "repeated_entity_paths": repeated,
        "explicit_linkage_observations": explicit_linkage,
    }


def _velocity_profile(events: list[ObservedEvent]) -> dict[str, Any]:
    timestamps = _timestamps(events)
    elapsed = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
    value_events = [
        event for event in events if event.event_type in {"PAYMENT_INITIATED", "PAYMENT_REPEATED", "FUNDS_RECEIVED", "FUNDS_DISPERSED"}
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
    supporting_markers = {
        "out_of_band_verification_complete",
        "dual_approval_complete",
        "customer_notified_change",
        "strong_authentication_passed",
        "customer_confirmation_complete",
        "independent_checks_passed",
        "verified_funding_source",
        "registered_entity_age_days",
        "settlement_pattern_seen_days",
        "recurring_mandate_age_days",
        "known_contact_relationship_days",
        "supplier_relationship_days",
    }
    observations: list[dict[str, Any]] = []
    for event in events:
        present = {
            key: value
            for key, value in event.attributes.items()
            if key in supporting_markers and value not in (False, None, 0, "")
        }
        if present:
            observations.append({"event_id": event.event_id, "context": present})
    return {
        "supporting_context_found": bool(observations),
        "supporting_observations": observations,
        "warning": "Supporting context is evidence, not a truth label; corroborate before allowing.",
    }


TOOL_IMPLEMENTATIONS: dict[str, Callable[[list[ObservedEvent]], dict[str, Any]]] = {
    "timeline_summary": _timeline_summary,
    "entity_linkage": _entity_linkage,
    "velocity_profile": _velocity_profile,
    "payment_context": _payment_context,
    "legitimate_alternatives": _legitimate_alternatives,
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
                )
            )
        return packets
