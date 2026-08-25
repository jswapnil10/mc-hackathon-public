"""Compile safe scenario stages into label-separated, deterministic event streams.

Phase-0 leakage fixes (see docs/blue-team-ml-evolution-research.md):
  L4 amount baseline    -> per-account baseline + amount_to_baseline_ratio (no hard-coded 2500)
  materializer gap      -> any *_probability attribute is rolled to a boolean, not just the 5 named ones
  fraud_contributing    -> benign-in-isolation attack stages are labelled non-contributing
  L2/L3 controls        -> legitimate controls mirror the attack's full stage length + attribute keys
"""

from __future__ import annotations

import copy
import hashlib
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from red_team_agent.models import ScenarioSpec

from .contracts import ObservedEvent, SimulationCase, TruthRecord


BASELINE_AMOUNT_INR = 2_500.0  # fallback only, when no sender account is present
PAYMENT_EVENT_TYPES = {
    "PAYMENT_INITIATED", "PAYMENT_REPEATED", "AGENTIC_PAYMENT_INITIATED",
    "PAYOUT_REQUESTED", "PAYOUT_SETTLED", "DISPUTE_REFUND_ISSUED",
}
VALUE_EVENT_TYPES = PAYMENT_EVENT_TYPES | {"FUNDS_RECEIVED", "FUNDS_DISPERSED"}

# MasterGuard lifecycle taxonomy (pre_transaction / transaction / post_transaction). Every
# ObservedEvent must carry a lifecycle_phase (required by the merged contracts + referee metrics).
PRE_TRANSACTION_EVENT_TYPES = {
    "COMMUNICATION_RISK_CONTEXT", "SESSION_STARTED", "AUTHENTICATION_CONTEXT_CHANGED",
    "IDENTITY_APPLICATION_SUBMITTED", "IDENTITY_VERIFICATION_ATTEMPTED", "ACCOUNT_OPENED",
    "ACCOUNT_WARMUP_ACTIVITY", "ACCOUNT_BEHAVIOR_PROFILE_UPDATED", "BENEFICIARY_ADDED",
    "SUPPLIER_PROFILE_CHANGED", "PAYOUT_DESTINATION_CHANGED", "AGENT_COMMERCE_SESSION_STARTED",
    "AGENT_PAYMENT_INTENT_PRESENTED",
}
POST_TRANSACTION_EVENT_TYPES = {
    "FUNDS_RECEIVED", "FUNDS_DISPERSED", "PAYOUT_SETTLED", "DISPUTE_OPENED",
    "DISPUTE_EVIDENCE_REVIEWED", "DISPUTE_REFUND_ISSUED",
}


def event_lifecycle_phase(event_type: str) -> str:
    if event_type in PRE_TRANSACTION_EVENT_TYPES:
        return "pre_transaction"
    if event_type in POST_TRANSACTION_EVENT_TYPES:
        return "post_transaction"
    return "transaction"

# Attack stages that are benign in isolation: part of the kill chain but carrying no
# malicious action on their own, so they are labelled non-fraud-contributing.
BENIGN_STAGE_IDS = {
    "novel_session",
    "familiarized_session",
    "persuasion_context",
    "supplier_contact_context",
    "account_warmup",
    "history_building",
    "bounded_replay",
}


def _stable_id(*parts: object) -> str:
    return f"syn_evt_{uuid.uuid5(uuid.NAMESPACE_URL, '|'.join(map(str, parts))).hex[:20]}"


def account_baseline_inr(account_id: object) -> float:
    """Deterministic per-account baseline spend (INR), stable across events for one account.

    Replaces the hard-coded 2500 so a legit account and an attacked account draw amounts
    from the SAME per-account distribution (kills the amount-vs-baseline label copy, L4).
    """
    if not account_id:
        return BASELINE_AMOUNT_INR
    digest = hashlib.md5(str(account_id).encode("utf-8")).hexdigest()
    r = int(digest[:8], 16) / 0xFFFFFFFF  # 0..1
    return round(800.0 + r * (12_000.0 - 800.0), 2)  # ~800..12000 INR


def _materialize_attributes(
    attributes: dict[str, Any], *, rng: random.Random, event_type: str
) -> dict[str, Any]:
    materialized = copy.deepcopy(attributes)
    materialized.pop("planner_emphasis", None)

    # Named probability -> boolean mappings (kept for readable, well-known signal names).
    named = {
        "new_device_probability": "device_is_new",
        "new_network_probability": "network_is_new",
        "shared_device_probability": "device_shared_across_accounts",
        "shared_network_probability": "network_shared_across_accounts",
        "device_reuse_probability": "device_reused_across_profiles",
    }
    for source, target in named.items():
        if source in materialized:
            materialized[target] = rng.random() < float(materialized.pop(source))
    # Generic fallback: any remaining "<x>_probability" attribute is rolled to "<x>_flag".
    for source in [k for k in materialized if isinstance(k, str) and k.endswith("_probability")]:
        try:
            p = float(materialized.pop(source))
        except (TypeError, ValueError):
            continue
        materialized[source[: -len("_probability")] + "_flag"] = rng.random() < p

    multiplier = float(materialized.pop("amount_multiplier", 1.0))
    if event_type in VALUE_EVENT_TYPES:
        base = account_baseline_inr(materialized.get("sender_account_id"))
        materialized["sender_baseline_amount_inr"] = base
        materialized["amount_inr"] = round(base * multiplier, 2)
        materialized["amount_to_baseline_ratio"] = round(multiplier, 4)

    # Fix A (reverse-leak): auth/confirmation fields must be PRESENT on attack events too, so a
    # model cannot learn "field present => legit". But they are stamped strictly False here —
    # attackers do not pass strong auth / genuine confirmation, and MasterGuard's evidence layer
    # only treats these as verified-legitimacy markers when they are `is True`. So attacks get the
    # key present-but-False (ML sees it on both classes; their fast-path is never falsely triggered).
    # Controls/ambient set them True via _benignize, giving the value its genuine legit meaning.
    if event_type in VALUE_EVENT_TYPES or event_type == "SESSION_STARTED":
        materialized.setdefault("strong_authentication_passed", False)
        materialized.setdefault("customer_confirmation_complete", False)
    return materialized


def _value_at_risk(scenario: ScenarioSpec, stage_id: str, event_type: str, attributes: dict[str, Any]) -> float:
    amount = float(attributes.get("amount_inr", 0.0))
    if event_type == "PAYMENT_INITIATED":
        return amount
    if event_type == "PAYMENT_REPEATED":
        count = max(1, int(attributes.get("payment_count", scenario.parameters.get("payment_count", 2))))
        return amount * count
    if scenario.attack_family == "MULE-01" and stage_id == "fan_in":
        return amount * int(scenario.parameters.get("sender_count", 1))
    return 0.0


def simulate_attack(scenario: ScenarioSpec) -> SimulationCase:
    rng = random.Random(scenario.seed)
    # Spread attack start times across the full 90-day window (and thus across weekdays/hours), the
    # same span ambient traffic uses, so hour_of_day/day_of_week carry no attack signal (was clustered
    # on ~one day because the offset was seed % 1440 minutes).
    start = datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(minutes=scenario.seed % (90 * 24 * 60))
    events: list[ObservedEvent] = []
    truth: list[TruthRecord] = []
    for stage in scenario.stages:
        event_id = _stable_id(scenario.scenario_id, stage.stage_id, stage.sequence)
        attributes = _materialize_attributes(stage.attributes, rng=rng, event_type=stage.event_type)
        events.append(
            ObservedEvent(
                event_id=event_id,
                sequence=stage.sequence,
                occurred_at=(start + timedelta(seconds=stage.offset_seconds)).isoformat(),
                lifecycle_phase=event_lifecycle_phase(stage.event_type),
                event_type=stage.event_type,
                observable_signals=list(stage.observable_signals),
                attributes=attributes,
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
                value_at_risk_inr=round(_value_at_risk(scenario, stage.stage_id, stage.event_type, attributes), 2),
                offset_seconds=stage.offset_seconds,
                fraud_contributing=stage.stage_id not in BENIGN_STAGE_IDS,
            )
        )
    return SimulationCase(case_id=scenario.scenario_id, events=events, truth=truth)


# ---------------------------------------------------------------------------- controls

# Benign markers, some carrying overlapping "suspicious-looking but genuine" values, so a
# control is not separable from an attack by attribute PRESENCE (L2) or chain LENGTH (L3):
# a legit control now mirrors the attack's exact stage count, event types and attribute keys,
# differing only in VALUES drawn from an overlapping legit distribution.
def _benignize(attrs: dict[str, Any], *, event_type: str, rng: random.Random) -> dict[str, Any]:
    benign = copy.deepcopy(attrs)
    account_id = benign.get("sender_account_id")

    if event_type in VALUE_EVENT_TYPES:
        base = account_baseline_inr(account_id)
        ratio = round(rng.uniform(0.5, 1.8), 4)  # overlaps the low end of attack multipliers
        benign["sender_baseline_amount_inr"] = base
        benign["amount_inr"] = round(base * ratio, 2)
        benign["amount_to_baseline_ratio"] = ratio

    # Novelty / sharing flags: mostly benign but occasionally true (genuine new phone / shared
    # household device) so a True value is not a fraud marker.
    for key in list(benign):
        if isinstance(key, str) and (
            key.endswith("_is_new")
            or key.endswith("_flag")
            or key.endswith("_across_accounts")
            or key.endswith("_across_profiles")
        ):
            benign[key] = rng.random() < 0.15

    if "beneficiary_age_days" in benign:
        # Genuine customers DO sometimes pay a brand-new payee; if legit payees were always aged
        # (>=90d) while attacks top out ~180d, "young payee" becomes a near-perfect fraud shortcut
        # (fix B). ~20% fresh (0-180d, overlapping the attack range) so age alone can't separate.
        benign["beneficiary_age_days"] = rng.randint(0, 180) if rng.random() < 0.2 else rng.randint(180, 1500)
    if "urgency_level" in benign:
        benign["urgency_level"] = rng.randint(0, 3)

    # Phase-4 canonical attributes (§3): benign draws MUST overlap the hard-tier attack values so
    # no single canonical separates the classes (L1). Each hard attack value sits inside the
    # legit range: fan-in 30<=[5,150], dwell 2400<=fast-tail[30,3600], drain 0.55<=[0.05,0.9],
    # add-latency 3600<=quick-tail[30,1800], kyc full_kyc/min_kyc both appear in legit.
    if "receiver_inbound_sender_count" in benign:
        benign["receiver_inbound_sender_count"] = rng.randint(5, 150)  # popular legit payees also high
    if "funds_dwell_time_seconds" in benign:
        benign["funds_dwell_time_seconds"] = (
            rng.randint(30, 3600) if rng.random() < 0.15 else rng.randint(600, 172800)
        )  # ~15% legit sweep-accounts move fast
    if "balance_drain_ratio" in benign:
        benign["balance_drain_ratio"] = round(rng.uniform(0.05, 0.9), 4)  # many legit payments are large
    if "add_to_payment_seconds" in benign:
        benign["add_to_payment_seconds"] = (
            rng.randint(30, 1800) if rng.random() < 0.15 else rng.randint(120, 86400)
        )  # ~15% genuine urgent payments are quick
    if "beneficiary_kyc_tier" in benign:
        benign["beneficiary_kyc_tier"] = rng.choices(
            ["full_kyc", "business_kyc", "min_kyc"], weights=[55, 25, 20]
        )[0]  # legit thin-file users are also min_kyc

    # Phase-4 canonical attributes (2nd pass): same L1 overlap discipline. Hard-tier attack
    # values sit inside the legit range: binding-age 30<=fresh-tail[0,60], dormancy 45<=[0,120]
    # and 400<=long-tail[120,900], name-match 0.85<=near-miss-tail[0.4,0.85], round 0.4<=[0,0.8].
    if "device_binding_age_days" in benign:
        benign["device_binding_age_days"] = (
            rng.randint(0, 60) if rng.random() < 0.15 else rng.randint(30, 1500)
        )  # ~15% genuine new-phone/travel show fresh binding
    if "dormancy_days_before_activity" in benign:
        benign["dormancy_days_before_activity"] = (
            rng.randint(120, 900) if rng.random() < 0.2 else rng.randint(0, 120)
        )  # seasonal legit accounts also reactivate after dormancy
    if "beneficiary_name_match_score" in benign:
        benign["beneficiary_name_match_score"] = (
            round(rng.uniform(0.4, 0.85), 4) if rng.random() < 0.15 else round(rng.uniform(0.7, 1.0), 4)
        )  # ~15% legit CoP near-misses (abbreviations/initials)
    if "round_amount_ratio" in benign:
        benign["round_amount_ratio"] = round(rng.uniform(0.0, 0.8), 4)  # many legit payments are round too

    benign["strong_authentication_passed"] = True
    benign["customer_confirmation_complete"] = True
    return benign


def _standalone_case(case_id: str, event_type: str, attrs: dict[str, Any], ts: datetime) -> SimulationCase:
    """Wrap one standalone legit event as a length-1 SimulationCase Blue + Referee can consume."""
    event_id = _stable_id(case_id, event_type)
    event = ObservedEvent(
        event_id=event_id,
        sequence=1,
        occurred_at=ts.isoformat(),
        lifecycle_phase=event_lifecycle_phase(event_type),
        event_type=event_type,
        observable_signals=[],
        attributes=attrs,
    )
    truth = TruthRecord(
        event_id=event_id,
        stage_id="standalone",
        attack_family=None,
        scenario_id=case_id,
        intervention_point="DECIDE",
        is_attack=False,
        value_at_risk_inr=0.0,
        offset_seconds=0,
        fraud_contributing=False,
    )
    return SimulationCase(case_id=case_id, events=[event], truth=[truth], control_name=None)


def simulate_ambient_cases(*, seed: int, count: int) -> list[SimulationCase]:
    """Ordinary standalone legit transactions (no kill chain), each a length-1 case. Mirrors the
    attribute space of dataset._ambient_events so the live false-positive denominator matches
    training-time ambient traffic. Deterministic in `seed`."""
    rng = random.Random(f"ambient-{seed}")
    base_ts = datetime(2026, 8, 3, 11, 0, tzinfo=timezone.utc)
    event_types = ["PAYMENT_INITIATED", "SESSION_STARTED", "BENEFICIARY_ADDED", "FUNDS_RECEIVED", "PAYMENT_REPEATED"]
    cases: list[SimulationCase] = []
    for i in range(max(0, count)):
        acct = f"syn_acct_{rng.randint(100000, 999999)}"
        et = rng.choices(event_types, weights=[50, 20, 12, 10, 8])[0]
        base = account_baseline_inr(acct)
        ratio = round(rng.uniform(0.3, 2.2), 4)
        attrs: dict[str, Any] = {
            "sender_account_id": acct,
            "beneficiary_id": f"syn_beneficiary_{rng.randint(100000, 999999)}",
            "device_id": f"syn_device_{rng.randint(100000, 999999)}",
            "beneficiary_age_days": rng.randint(0, 1800),
            "session_age_seconds": rng.randint(30, 4000),
        }
        if et in {"PAYMENT_INITIATED", "PAYMENT_REPEATED", "FUNDS_RECEIVED"}:
            attrs["sender_baseline_amount_inr"] = base
            attrs["amount_inr"] = round(base * ratio, 2)
            attrs["amount_to_baseline_ratio"] = ratio
        if et == "PAYMENT_REPEATED":
            attrs["payment_count"] = rng.randint(2, 6)
        ts = base_ts + timedelta(minutes=rng.randint(0, 90 * 24 * 60))
        cases.append(_standalone_case(f"AMB-{seed}-{i:04d}", et, attrs, ts))
    return cases


def simulate_trap_cases(*, seed: int, count_each: int) -> list[SimulationCase]:
    """Standalone hard-negatives: each mimics ONE scary-looking attack signal in isolation
    (big-ticket / new-payee / new-device / reactivation), so no single value flags fraud alone.
    Mirrors dataset._hard_negative_traps. Deterministic in `seed`."""
    rng = random.Random(f"traps-{seed}")
    base_ts = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    cases: list[SimulationCase] = []
    for kind in ["big_ticket", "new_payee", "new_device", "reactivation"]:
        for i in range(max(0, count_each)):
            acct = f"syn_acct_{rng.randint(100000, 999999)}"
            base = account_baseline_inr(acct)
            attrs: dict[str, Any] = {"sender_account_id": acct, "session_age_seconds": rng.randint(30, 4000)}
            et = "PAYMENT_INITIATED"
            if kind == "big_ticket":
                ratio = round(rng.uniform(6.0, 14.0), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, beneficiary_age_days=rng.randint(200, 1500))
            elif kind == "new_payee":
                ratio = round(rng.uniform(0.6, 1.8), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, beneficiary_age_days=0,
                             beneficiary_id=f"syn_beneficiary_{rng.randint(100000, 999999)}")
            elif kind == "new_device":
                et = "SESSION_STARTED"
                attrs.update(device_id=f"syn_device_{rng.randint(100000, 999999)}", device_is_new=True,
                             strong_authentication_passed=True)
            else:  # reactivation
                ratio = round(rng.uniform(0.5, 1.5), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, dormancy_days_before_activity=rng.randint(180, 900))
            ts = base_ts + timedelta(minutes=rng.randint(0, 90 * 24 * 60))
            cases.append(_standalone_case(f"TRAP-{kind}-{seed}-{i:04d}", et, attrs, ts))
    return cases


def simulate_legitimate_controls(scenario: ScenarioSpec, control_names: list[str]) -> list[SimulationCase]:
    """Full-length benign twins of the attack: same stage count/event-types/attribute keys,
    benign overlapping values, is_attack=False. Fixes the chain-length (L3) and
    presence/absence (L2) leaks that 1-2 event hand-authored controls created."""
    base_ts = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
    cases: list[SimulationCase] = []
    for index, name in enumerate(control_names, start=1):
        rng = random.Random(f"{scenario.seed}-{name}-{index}")
        case_id = f"CTRL-{scenario.attack_family}-{index:02d}-{scenario.seed}"
        events: list[ObservedEvent] = []
        truth: list[TruthRecord] = []
        for stage in scenario.stages:
            attack_attrs = _materialize_attributes(stage.attributes, rng=rng, event_type=stage.event_type)
            attrs = _benignize(attack_attrs, event_type=stage.event_type, rng=rng)
            event_id = _stable_id(case_id, stage.sequence)
            events.append(
                ObservedEvent(
                    event_id=event_id,
                    sequence=stage.sequence,
                    occurred_at=(base_ts + timedelta(minutes=index * 10, seconds=stage.offset_seconds)).isoformat(),
                    lifecycle_phase=event_lifecycle_phase(stage.event_type),
                    event_type=stage.event_type,
                    observable_signals=list(stage.observable_signals),
                    attributes=attrs,
                )
            )
            truth.append(
                TruthRecord(
                    event_id=event_id,
                    stage_id=f"control_{stage.stage_id}",
                    attack_family=None,
                    scenario_id=case_id,
                    intervention_point=stage.blue_intervention_point,
                    is_attack=False,
                    value_at_risk_inr=0.0,
                    offset_seconds=stage.offset_seconds,
                    fraud_contributing=False,
                )
            )
        cases.append(SimulationCase(case_id=case_id, events=events, truth=truth, control_name=name))
    return cases
