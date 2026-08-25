"""Leakage-controlled, point-in-time feature engineering for the Blue-team ML pipeline.

ONE causal function, `build_features(current_event, visible_history)`, used identically at
dataset-build time and (later) at inference time, so legit and fraud rows get identical
arithmetic and no label path can open. `visible_history` is the strictly-prior events for
the same chain/account (the streaming cursor blue_agent.py already maintains).

No model here — this is data prep only. The classifier is deferred (see package docstring).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# ---- allowlist: the ONLY columns a future model may consume ----------------------------
NUMERIC_FEATURES = [
    "amount_to_baseline_ratio",
    "beneficiary_age_days",
    "session_age_seconds",
    "urgency_level",
    "approval_step_count",
    "prior_event_count",            # causal: number of strictly-prior events in this chain
    "seconds_since_prev_event",
    "events_last_hour",
    "distinct_beneficiaries_prior",
    "distinct_devices_prior",
    "hour_of_day",
    "day_of_week",
    # ---- Tier 1: signals the LLM already leverages; present on both attack AND legit rows
    # (controls mirror attack stages, hard-negative traps carry them too) so no presence leak.
    # Booleans are left as True/False/None — a HistGBM-style model treats them natively.
    "device_is_new",
    "network_is_new",
    "strong_authentication_passed",
    "customer_confirmation_complete",
    "payment_count",
    "device_shared_across_accounts",
    "network_shared_across_accounts",
    "device_reused_across_profiles",
    # ---- Tier 2: family-narrow, sparse (mostly None); safe because controls carry them too.
    "sender_count",                 # MULE fan-in width
    "change_risk_band",             # ATO authentication-change severity
    "linked_attribute_count",       # SYNID/MULE entity linkage
    # ---- Derived causal aggregates computed from strictly-prior events (not read raw).
    "distinct_networks_prior",
    "value_sum_last_hour_inr",
    # ---- Phase-4 canonical cross-family attributes (§3). Simulated in cards + controls + ambient
    # with overlapping legit/fraud distributions (leakage tests in tests/test_canonical_attributes),
    # so each is present on all three sources with no L1 disjoint-range or L2 presence leak.
    "receiver_inbound_sender_count",
    "funds_dwell_time_seconds",
    "balance_drain_ratio",
    "add_to_payment_seconds",
    "device_binding_age_days",
    "dormancy_days_before_activity",
    "beneficiary_name_match_score",
    "round_amount_ratio",
]
CATEGORICAL_FEATURES = ["event_type", "beneficiary_kyc_tier"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- forbidden inputs: never allowed to reach FEATURES ---------------------------------
ANSWER_KEY = {
    "is_attack", "fraud_contributing", "stage_id", "attack_family",
    "scenario_id", "intervention_point", "value_at_risk_inr", "label",
}
GROUPING_KEYS = {
    "sender_account_id", "beneficiary_id", "device_id", "network_id",
    "merchant_id", "identity_id", "supplier_id", "event_id", "case_id", "chain_id", "group_key",
}
FINGERPRINTS = {"occurred_at", "sequence", "offset_seconds", "observable_signals"}
FORBIDDEN_MODEL_INPUTS = ANSWER_KEY | GROUPING_KEYS | FINGERPRINTS


def audit_leakage() -> list[str]:
    """Return the list of forbidden columns that leaked into FEATURES (empty == clean)."""
    return sorted(set(FEATURES) & FORBIDDEN_MODEL_INPUTS)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def build_features(current_event: dict[str, Any], visible_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Causal feature vector for one event. Uses ONLY the current event + strictly-prior events.

    `current_event` / `visible_history` items are ObservedEvent.to_dict() shaped:
      {event_id, sequence, occurred_at, event_type, observable_signals, attributes}
    Missing attributes are left as None (a HistGBM-style model treats NaN as signal).
    """
    attrs = current_event.get("attributes", {}) or {}
    ts = _parse_ts(current_event.get("occurred_at"))

    prior = list(visible_history)
    seconds_since_prev = None
    events_last_hour = 0
    value_sum_last_hour = 0.0
    if prior:
        prev_ts = _parse_ts(prior[-1].get("occurred_at"))
        if ts and prev_ts:
            seconds_since_prev = max(0.0, (ts - prev_ts).total_seconds())
        for ev in prior:
            ev_ts = _parse_ts(ev.get("occurred_at"))
            if ts and ev_ts and 0 <= (ts - ev_ts).total_seconds() <= 3600:
                events_last_hour += 1
                try:
                    value_sum_last_hour += float((ev.get("attributes", {}) or {}).get("amount_inr") or 0.0)
                except (TypeError, ValueError):
                    pass
    # Include the current event's own amount in the trailing-hour value (causal: known at decision time).
    try:
        value_sum_last_hour += float(attrs.get("amount_inr") or 0.0)
    except (TypeError, ValueError):
        pass

    def _prior_distinct(field: str) -> int:
        seen = {(e.get("attributes", {}) or {}).get(field) for e in prior}
        seen.discard(None)
        return len(seen)

    feats: dict[str, Any] = {
        "amount_to_baseline_ratio": attrs.get("amount_to_baseline_ratio"),
        "beneficiary_age_days": attrs.get("beneficiary_age_days"),
        "session_age_seconds": attrs.get("session_age_seconds"),
        "urgency_level": attrs.get("urgency_level"),
        "approval_step_count": attrs.get("approval_step_count"),
        "prior_event_count": len(prior),
        "seconds_since_prev_event": seconds_since_prev,
        "events_last_hour": events_last_hour,
        "distinct_beneficiaries_prior": _prior_distinct("beneficiary_id"),
        "distinct_devices_prior": _prior_distinct("device_id"),
        "hour_of_day": ts.hour if ts else None,
        "day_of_week": ts.weekday() if ts else None,
        # Tier 1 (from current-event attributes; None when absent — treated as NaN by the model).
        "device_is_new": attrs.get("device_is_new"),
        "network_is_new": attrs.get("network_is_new"),
        "strong_authentication_passed": attrs.get("strong_authentication_passed"),
        "customer_confirmation_complete": attrs.get("customer_confirmation_complete"),
        "payment_count": attrs.get("payment_count"),
        "device_shared_across_accounts": attrs.get("device_shared_across_accounts"),
        "network_shared_across_accounts": attrs.get("network_shared_across_accounts"),
        "device_reused_across_profiles": attrs.get("device_reused_across_profiles"),
        # Tier 2 (family-narrow, sparse).
        "sender_count": attrs.get("sender_count"),
        "change_risk_band": attrs.get("change_risk_band"),
        "linked_attribute_count": attrs.get("linked_attribute_count"),
        # Derived causal aggregates.
        "distinct_networks_prior": _prior_distinct("network_id"),
        "value_sum_last_hour_inr": round(value_sum_last_hour, 2),
        # Phase-4 canonicals (from current-event attributes; None when absent -> NaN).
        "receiver_inbound_sender_count": attrs.get("receiver_inbound_sender_count"),
        "funds_dwell_time_seconds": attrs.get("funds_dwell_time_seconds"),
        "balance_drain_ratio": attrs.get("balance_drain_ratio"),
        "add_to_payment_seconds": attrs.get("add_to_payment_seconds"),
        "device_binding_age_days": attrs.get("device_binding_age_days"),
        "dormancy_days_before_activity": attrs.get("dormancy_days_before_activity"),
        "beneficiary_name_match_score": attrs.get("beneficiary_name_match_score"),
        "round_amount_ratio": attrs.get("round_amount_ratio"),
        "event_type": current_event.get("event_type"),
        "beneficiary_kyc_tier": attrs.get("beneficiary_kyc_tier"),
    }
    # Contract guarantee: build_features never emits a forbidden column.
    assert not (set(feats) & FORBIDDEN_MODEL_INPUTS), "build_features leaked a forbidden column"
    assert set(feats) == set(FEATURES), "build_features output must match the FEATURES allowlist"
    return feats
