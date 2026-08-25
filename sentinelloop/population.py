"""Population-scale, sealed-label synthetic data generation for the Generate pillar."""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from red_team_agent.models import PlannerDecision, ScenarioSpec
from red_team_agent.planner import RedTeamAgent

from .contracts import ObservedEvent, SimulationCase, TruthRecord
from .simulation import simulate_attack, simulate_legitimate_controls
from .threat_atlas import ThreatAtlas, ThreatVector


SPLITS = ("train", "validation", "test_known", "test_novel")
RAIL_PRIORS = {
    "UPI": 0.45,
    "CARD": 0.25,
    "BANK_TRANSFER": 0.18,
    "WALLET": 0.12,
}
CHANNEL_PRIORS = {
    "MOBILE_APP": 0.56,
    "WEB": 0.20,
    "POS": 0.14,
    "API": 0.10,
}
RAIL_CHANNELS = {
    "UPI": (("MOBILE_APP", 0.88), ("QR", 0.12)),
    "CARD": (("ECOMMERCE", 0.55), ("POS", 0.35), ("MOBILE_APP", 0.10)),
    "BANK_TRANSFER": (("WEB", 0.42), ("MOBILE_APP", 0.45), ("ERP", 0.13)),
    "WALLET": (("MOBILE_APP", 0.82), ("WEB", 0.18)),
}
RAIL_AMOUNT_MEDIANS = {
    "UPI": 1_250.0,
    "CARD": 2_400.0,
    "BANK_TRANSFER": 12_500.0,
    "WALLET": 700.0,
    "BNPL": 3_500.0,
    "PAYOUT": 18_000.0,
    "AGENTIC_PAYMENT": 2_800.0,
}
AUTH_BY_RAIL = {
    "UPI": ("PIN", "BIOMETRIC"),
    "CARD": ("3DS", "OTP", "TOKEN_CRYPTOGRAM"),
    "BANK_TRANSFER": ("PASSWORD_OTP", "BIOMETRIC", "DUAL_APPROVAL"),
    "WALLET": ("PIN", "BIOMETRIC"),
    "BNPL": ("OTP", "DEVICE_BINDING"),
    "PAYOUT": ("DUAL_APPROVAL", "API_SIGNATURE"),
    "AGENTIC_PAYMENT": ("AGENT_SIGNATURE", "TOKEN_CRYPTOGRAM"),
}
REGIONS = ("MH", "DL", "KA", "TN", "GJ", "TS", "WB", "UP", "RJ", "KL")
MERCHANT_CATEGORIES = (
    "GROCERY",
    "FUEL",
    "DINING",
    "TRAVEL",
    "ECOMMERCE",
    "UTILITIES",
    "P2P_TRANSFER",
)
LEGITIMATE_EVENT_PRIORS = (
    ("PAYMENT_INITIATED", "transaction", "payment_orchestration_stream", "inline_payment_decision", 300, 0.58),
    ("SESSION_STARTED", "pre_transaction", "identity_session_stream", "pre_payment_intervention", 1500, 0.07),
    ("AUTHENTICATION_CONTEXT_CHANGED", "pre_transaction", "identity_risk_stream", "pre_payment_intervention", 1500, 0.05),
    ("BENEFICIARY_ADDED", "pre_transaction", "beneficiary_management_stream", "pre_payment_intervention", 1500, 0.06),
    ("SUPPLIER_PROFILE_CHANGED", "pre_transaction", "supplier_management_stream", "pre_payment_intervention", 1500, 0.03),
    ("AGENT_COMMERCE_SESSION_STARTED", "pre_transaction", "agent_trust_gateway", "pre_payment_intervention", 1500, 0.03),
    ("FUNDS_RECEIVED", "transaction", "ledger_event_stream", "inline_payment_decision", 300, 0.05),
    ("PAYOUT_REQUESTED", "transaction", "merchant_payout_stream", "inline_payment_decision", 300, 0.04),
    ("DISPUTE_OPENED", "post_transaction", "dispute_case_stream", "post_payment_recovery", 5000, 0.04),
    ("DISPUTE_EVIDENCE_REVIEWED", "post_transaction", "dispute_case_stream", "post_payment_recovery", 5000, 0.025),
    ("PAYOUT_SETTLED", "post_transaction", "merchant_payout_stream", "post_payment_recovery", 5000, 0.025),
    ("FUNDS_DISPERSED", "post_transaction", "ledger_event_stream", "post_payment_recovery", 5000, 0.02),
)
VERIFICATION_KEYS = {
    "out_of_band_verification_complete",
    "dual_approval_complete",
    "customer_notified_change",
    "strong_authentication_passed",
    "customer_confirmation_complete",
    "verified_funding_source",
}
AGENT_TRUST_KEYS = {
    "agent_signature_valid",
    "consumer_consent_valid",
    "intent_scope_match",
    "payment_container_match",
    "merchant_scope_match",
}


@dataclass(frozen=True)
class PopulationConfig:
    variants_per_vector: int = 6
    legitimate_event_count: int = 2400
    seed: int = 20260824

    def validate(self) -> None:
        if not 4 <= self.variants_per_vector <= 50:
            raise ValueError("variants_per_vector must be between 4 and 50.")
        if not 400 <= self.legitimate_event_count <= 100_000:
            raise ValueError("legitimate_event_count must be between 400 and 100000.")


def _stable_id(kind: str, *parts: object) -> str:
    value = uuid.uuid5(uuid.NAMESPACE_URL, "|".join(map(str, parts))).hex[:20]
    return f"syn_{kind}_{value}"


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _weighted_pairs(rng: random.Random, choices: tuple[tuple[str, float], ...]) -> str:
    return rng.choices(
        [item[0] for item in choices], weights=[item[1] for item in choices], k=1
    )[0]


def _auth_for_rail(rail: str, rng: random.Random) -> str:
    return rng.choice(AUTH_BY_RAIL.get(rail, ("RISK_BASED_AUTH",)))


def _split_for_ordinary(index: int, count: int) -> str:
    fraction = (index + 0.5) / count
    if fraction <= 0.60:
        return "train"
    if fraction <= 0.78:
        return "validation"
    if fraction <= 0.92:
        return "test_known"
    return "test_novel"


def _split_for_variant(vector: ThreatVector, index: int, count: int, atlas: ThreatAtlas) -> str:
    if vector.id in atlas.novel_holdout_vector_ids:
        return "test_novel"
    fraction = (index + 0.5) / count
    if fraction <= 0.60:
        return "train"
    if fraction <= 0.80:
        return "validation"
    return "test_known"


def _ordinary_rows(config: PopulationConfig, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    split_account_counts = {
        split: max(30, math.ceil(config.legitimate_event_count / 20)) for split in SPLITS
    }
    profiles: dict[tuple[str, int], dict[str, Any]] = {}
    for split, account_count in split_account_counts.items():
        for index in range(account_count):
            rail = _weighted_choice(rng, RAIL_PRIORS)
            baseline = RAIL_AMOUNT_MEDIANS[rail] * rng.lognormvariate(0.0, 0.55)
            profiles[(split, index)] = {
                "sender_account_id": _stable_id("account", split, index),
                "device_id": _stable_id("device", split, index),
                "ip_prefix": f"10.{SPLITS.index(split) + 10}.{index % 240}.0/24",
                "region": rng.choice(REGIONS),
                "account_age_days": rng.randint(60, 3650),
                "baseline": round(baseline, 2),
                "primary_rail": rail,
                "known_beneficiaries": [
                    _stable_id("beneficiary", split, index, beneficiary_index)
                    for beneficiary_index in range(rng.randint(2, 7))
                ],
            }

    for index in range(config.legitimate_event_count):
        split = _split_for_ordinary(index, config.legitimate_event_count)
        account_count = split_account_counts[split]
        profile = profiles[(split, rng.randrange(account_count))]
        rail = profile["primary_rail"] if rng.random() < 0.72 else _weighted_choice(rng, RAIL_PRIORS)
        channel = _weighted_pairs(rng, RAIL_CHANNELS[rail])
        edge_case = rng.random() < 0.12
        new_device = edge_case and rng.random() < 0.55
        new_beneficiary = edge_case and rng.random() < 0.70
        beneficiary = (
            _stable_id("beneficiary_new", split, index)
            if new_beneficiary
            else rng.choice(profile["known_beneficiaries"])
        )
        baseline = float(profile["baseline"])
        amount = min(250_000.0, baseline * rng.lognormvariate(0.0, 0.58))
        event_type, lifecycle_phase, source_system, decision_lane, latency_budget_ms, _ = rng.choices(
            LEGITIMATE_EVENT_PRIORS,
            weights=[item[-1] for item in LEGITIMATE_EVENT_PRIORS],
            k=1,
        )[0]
        carries_value = event_type in {
            "PAYMENT_INITIATED",
            "FUNDS_RECEIVED",
            "PAYOUT_REQUESTED",
            "PAYOUT_SETTLED",
            "FUNDS_DISPERSED",
        }
        observed_amount = amount if carries_value else 0.0
        event_ts = start + timedelta(
            days=SPLITS.index(split) * 10 + rng.randint(0, 8),
            minutes=rng.randint(0, 1439),
            seconds=rng.randint(0, 59),
        )
        verified = edge_case and rng.random() < 0.92
        rows.append(
            {
                "event_id": _stable_id("event", "ordinary", config.seed, index),
                "event_ts": event_ts.isoformat(),
                "case_id": _stable_id("case", "ordinary", config.seed, index),
                "scenario_id": None,
                "sequence": 1,
                "split": split,
                "case_role": "ordinary_legitimate",
                "label_fraud": False,
                "attack_family": None,
                "attack_vector_id": None,
                "near_neighbor_family": None,
                "difficulty": None,
                "legitimate_control": "VERIFIED_EDGE_CASE" if edge_case else None,
                "lifecycle_phase": lifecycle_phase,
                "event_type": event_type,
                "source_system": source_system,
                "decision_lane": decision_lane,
                "latency_budget_ms": latency_budget_ms,
                "payment_rail": rail,
                "channel": channel,
                "auth_method": _auth_for_rail(rail, rng),
                "merchant_category": "P2P_TRANSFER" if rail == "UPI" else rng.choice(MERCHANT_CATEGORIES),
                "currency": "INR",
                "transaction_status": "APPROVED",
                "sender_account_id": profile["sender_account_id"],
                "beneficiary_id": beneficiary,
                "device_id": (
                    _stable_id("device_new", split, index)
                    if new_device
                    else profile["device_id"]
                ),
                "ip_prefix": (
                    f"10.230.{index % 240}.0/24" if new_device else profile["ip_prefix"]
                ),
                "geo_region": rng.choice(REGIONS) if edge_case else profile["region"],
                "amount_inr": round(observed_amount, 2),
                "sender_baseline_amount_inr": round(baseline, 2),
                "amount_to_baseline": round(observed_amount / max(baseline, 1.0), 4),
                "session_age_seconds": rng.randint(45, 3600),
                "account_age_days": profile["account_age_days"],
                "beneficiary_age_days": rng.randint(0, 5) if new_beneficiary else rng.randint(60, 1200),
                "payment_count": 1,
                "urgency_level": 1,
                "channel_switch_count": 0,
                "observable_signal_count": rng.randint(2, 5),
                "device_is_new": int(new_device),
                "network_is_new": int(new_device),
                "behavior_automation_suspected": 0,
                "identity_consistency_mismatch": 0,
                "evidence_conflict_present": 0,
                "trust_failure_count": 0,
                "independent_verification_count": int(verified),
                "risk_signal_density": round((int(new_device) + int(new_beneficiary)) / 5.0, 4),
                "value_at_risk_inr": 0.0,
                "source_reference_count": 0,
            }
        )
    return rows


def _bounded_overrides(
    scenario: ScenarioSpec, *, agent: RedTeamAgent, rng: random.Random
) -> dict[str, Any]:
    card = agent.catalog.get(scenario.attack_family)
    overrides: dict[str, Any] = {}
    for name in card.allowed_mutations:
        value = scenario.parameters.get(name)
        bound = card.parameter_bounds.get(name, {})
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and "min" in bound
            and "max" in bound
        ):
            minimum, maximum = float(bound["min"]), float(bound["max"])
            radius = (maximum - minimum) * 0.08
            sampled = min(maximum, max(minimum, float(value) + rng.uniform(-radius, radius)))
            overrides[name] = int(round(sampled)) if isinstance(value, int) else round(sampled, 4)
    return overrides


def _compile_variant(
    *, agent: RedTeamAgent, family: str, difficulty: str, seed: int, rng: random.Random
) -> ScenarioSpec:
    base = agent.plan(attack_family=family, difficulty=difficulty, seed=seed)
    decision = PlannerDecision(
        attack_family=base.attack_family,
        difficulty=base.difficulty,
        objective=base.objective,
        target_lifecycle_phase=base.target_lifecycle_phase,
        focus_stage_ids=list(base.focus_stage_ids),
        adaptation_goal=base.adaptation_goal,
        reasoning_summary=(
            "Population Red Agent sampled one bounded behavioral variant from the source-backed card."
        ),
        backend="population-red-agent",
    )
    scenario = agent.compiler.compile(
        decision,
        seed=seed,
        parameter_overrides=_bounded_overrides(base, agent=agent, rng=rng),
    )
    report = agent.safety_gate.validate(scenario)
    if not report.approved:
        raise ValueError(f"Population scenario rejected by safety gate: {report.errors}")
    return scenario


def _numeric(attributes: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = attributes.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _flag(attributes: dict[str, Any], name: str) -> int:
    return int(attributes.get(name) is True)


def _normalize_case_rows(
    *,
    case: SimulationCase,
    split: str,
    vector: ThreatVector,
    difficulty: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    truth_by_event = {record.event_id: record for record in case.truth}
    is_attack = bool(case.truth and case.truth[0].is_attack)
    rail = rng.choice(vector.rails)
    channel = rng.choice(vector.channels)
    baseline = RAIL_AMOUNT_MEDIANS.get(rail, 2_500.0) * rng.lognormvariate(0.0, 0.45)
    account_age = rng.randint(1, 45) if vector.simulation_family == "SYNID-01" else rng.randint(90, 3200)
    rows: list[dict[str, Any]] = []
    for event in case.events:
        attributes = event.attributes
        truth: TruthRecord = truth_by_event[event.event_id]
        raw_amount = _numeric(attributes, "amount_inr")
        amount = raw_amount * baseline / 2_500.0 if raw_amount else 0.0
        # Simulator fixtures deliberately reuse readable identifiers. Namespace them
        # per case so an entity cannot leak from one sealed split into another while
        # preserving linkage between events inside the same synthetic case.
        sender = _stable_id(
            "account", case.case_id, attributes.get("sender_account_id", "sender")
        )
        beneficiary = _stable_id(
            "beneficiary", case.case_id, attributes.get("beneficiary_id", "beneficiary")
        )
        device = _stable_id(
            "device", case.case_id, attributes.get("device_id", "device")
        )
        network = _stable_id(
            "network", case.case_id, attributes.get("network_id", "network")
        )
        verification_count = sum(attributes.get(key) is True for key in VERIFICATION_KEYS)
        if _numeric(attributes, "independent_checks_passed") > 0:
            verification_count += 1
        trust_values = [attributes[key] for key in AGENT_TRUST_KEYS if key in attributes]
        trust_failures = sum(value is False for value in trust_values)
        signal_retention = {"easy": 0.96, "medium": 0.78, "hard": 0.56}[difficulty]
        sensor_flags = {
            key: _flag(attributes, key)
            for key in (
                "device_is_new",
                "network_is_new",
                "behavior_automation_suspected",
                "identity_consistency_mismatch",
                "evidence_conflict_present",
                "device_shared_across_accounts",
                "network_shared_across_accounts",
                "device_reused_across_profiles",
            )
        }
        vector_observables = " ".join(vector.defensive_observables).lower()
        if is_attack:
            sensor_flags = {
                key: int(value and rng.random() <= signal_retention)
                for key, value in sensor_flags.items()
            }
            # Compile the researched vector's defensive observables into the
            # existing production-shaped telemetry contract. This makes two
            # vectors in the same family behaviorally distinct without exposing
            # the vector identifier to Blue.
            if "device" in vector_observables and rng.random() <= signal_retention:
                sensor_flags["device_is_new"] = 1
            if "network" in vector_observables and rng.random() <= signal_retention:
                sensor_flags["network_is_new"] = 1
            if any(
                token in vector_observables
                for token in ("automation", "velocity", "burst", "replay")
            ) and rng.random() <= signal_retention:
                sensor_flags["behavior_automation_suspected"] = 1
            if any(
                token in vector_observables
                for token in ("identity", "liveness", "document", "evidence")
            ) and rng.random() <= signal_retention:
                sensor_flags["identity_consistency_mismatch"] = 1
            if any(
                token in vector_observables
                for token in ("consent", "signature", "scope", "approval", "mismatch")
            ) and rng.random() <= signal_retention:
                trust_failures += 1
            if difficulty == "hard" and rng.random() < 0.38:
                verification_count += 1
        elif rng.random() < 0.20:
            sensor_flags[rng.choice(tuple(sensor_flags))] = 1
            verification_count += 1
        risky_flags = sum(sensor_flags.values())
        beneficiary_age = _numeric(
            attributes,
            "beneficiary_age_days",
            "bank_change_recency_days",
            default=365.0,
        )
        payment_count = _numeric(attributes, "payment_count", "sender_count", default=1.0)
        if is_attack and any(
            token in vector_observables
            for token in ("beneficiary", "payee", "destination", "supplier")
        ):
            beneficiary_age = min(beneficiary_age, rng.randint(0, 5))
        if is_attack and any(
            token in vector_observables
            for token in ("velocity", "burst", "repeated", "fan_in", "fan_out")
        ):
            payment_count = max(payment_count, rng.randint(3, 9))
        amount_ratio = amount / max(baseline, 1.0) if amount else 0.0
        risk_points = (
            risky_flags
            + trust_failures
            + int(beneficiary_age < 7)
            + int(amount_ratio >= 2.0)
            + int(payment_count >= 3)
            + int(_numeric(attributes, "urgency_level") >= 4)
        )
        value_at_risk = (
            float(truth.value_at_risk_inr) * baseline / 2_500.0
            if truth.value_at_risk_inr
            else 0.0
        )
        rows.append(
            {
                "event_id": event.event_id,
                "event_ts": event.occurred_at,
                "case_id": case.case_id,
                "scenario_id": truth.scenario_id if is_attack else None,
                "sequence": event.sequence,
                "split": split,
                "case_role": "attack" if is_attack else "legitimate_control",
                "label_fraud": is_attack,
                "attack_family": vector.simulation_family if is_attack else None,
                "attack_vector_id": vector.id if is_attack else None,
                "near_neighbor_family": None if is_attack else vector.simulation_family,
                "difficulty": difficulty if is_attack else None,
                "legitimate_control": case.control_name,
                "lifecycle_phase": event.lifecycle_phase,
                "event_type": event.event_type,
                "source_system": event.source_system,
                "decision_lane": event.decision_lane,
                "latency_budget_ms": event.latency_budget_ms,
                "payment_rail": rail,
                "channel": channel,
                "auth_method": _auth_for_rail(rail, rng),
                "merchant_category": (
                    "P2P_TRANSFER"
                    if vector.simulation_family in {"APP-01", "ATO-01", "MULE-01", "EVADE-01"}
                    else "ECOMMERCE"
                ),
                "currency": "INR",
                "transaction_status": "OBSERVED",
                "sender_account_id": sender,
                "beneficiary_id": beneficiary,
                "device_id": device,
                "ip_prefix": f"10.{int(uuid.uuid5(uuid.NAMESPACE_URL, network).hex[:2], 16) % 240 + 1}.0.0/16",
                "geo_region": rng.choice(REGIONS),
                "amount_inr": round(min(250_000.0, max(0.0, amount)), 2),
                "sender_baseline_amount_inr": round(baseline, 2),
                "amount_to_baseline": round(amount_ratio, 4),
                "session_age_seconds": int(
                    _numeric(attributes, "session_age_seconds", "request_age_seconds", default=300.0)
                ),
                "account_age_days": account_age,
                "beneficiary_age_days": int(max(0.0, beneficiary_age)),
                "payment_count": int(max(1.0, payment_count)),
                "urgency_level": int(_numeric(attributes, "urgency_level", default=1.0)),
                "channel_switch_count": int(
                    _numeric(attributes, "channel_switch_count", default=0.0)
                ),
                "observable_signal_count": max(
                    1,
                    int(
                        round(
                            len(event.observable_signals)
                            * (signal_retention if is_attack else 1.0)
                        )
                    ),
                ),
                "device_is_new": sensor_flags["device_is_new"],
                "network_is_new": sensor_flags["network_is_new"],
                "behavior_automation_suspected": sensor_flags[
                    "behavior_automation_suspected"
                ],
                "identity_consistency_mismatch": sensor_flags[
                    "identity_consistency_mismatch"
                ],
                "evidence_conflict_present": sensor_flags["evidence_conflict_present"],
                "trust_failure_count": trust_failures,
                "independent_verification_count": verification_count,
                "risk_signal_density": round(
                    risk_points / max(5, len(event.observable_signals) + 3), 4
                ),
                "value_at_risk_inr": round(min(250_000.0, value_at_risk), 2),
                "source_reference_count": len(vector.source_ids),
            }
        )
    return rows


def generate_population_dataset(
    config: PopulationConfig | None = None,
    *,
    atlas: ThreatAtlas | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    active_config = config or PopulationConfig()
    active_config.validate()
    active_atlas = atlas or ThreatAtlas()
    rng = random.Random(active_config.seed)
    agent = RedTeamAgent()
    rows = _ordinary_rows(active_config, rng)
    scenario_count = 0
    control_case_count = 0
    for vector_index, vector in enumerate(active_atlas.vectors):
        for variant_index in range(active_config.variants_per_vector):
            split = _split_for_variant(
                vector, variant_index, active_config.variants_per_vector, active_atlas
            )
            difficulty = ("easy", "medium", "hard")[(vector_index + variant_index) % 3]
            scenario_seed = active_config.seed + (vector_index + 1) * 10_000 + variant_index
            scenario = _compile_variant(
                agent=agent,
                family=vector.simulation_family,
                difficulty=difficulty,
                seed=scenario_seed,
                rng=rng,
            )
            attack_case = simulate_attack(scenario)
            rows.extend(
                _normalize_case_rows(
                    case=attack_case,
                    split=split,
                    vector=vector,
                    difficulty=difficulty,
                    rng=rng,
                )
            )
            scenario_count += 1
            for control_case in simulate_legitimate_controls(
                scenario, scenario.legitimate_controls
            ):
                rows.extend(
                    _normalize_case_rows(
                        case=control_case,
                        split=split,
                        vector=vector,
                        difficulty=difficulty,
                        rng=rng,
                    )
                )
                control_case_count += 1

    frame = pd.DataFrame(rows)
    frame["event_ts"] = pd.to_datetime(frame["event_ts"], utc=True)
    frame = frame.sort_values(
        ["split", "event_ts", "case_id", "sequence"], kind="stable"
    ).reset_index(drop=True)
    split_counts = frame.groupby("split").size().reindex(SPLITS, fill_value=0)
    metadata = {
        "dataset_version": "population-v1",
        "seed": active_config.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grain": "one sanitized observable event per row",
        "event_count": len(frame),
        "attack_event_count": int(frame["label_fraud"].sum()),
        "legitimate_event_count": int((~frame["label_fraud"]).sum()),
        "ordinary_legitimate_event_count": int(
            frame["case_role"].eq("ordinary_legitimate").sum()
        ),
        "legitimate_control_event_count": int(
            frame["case_role"].eq("legitimate_control").sum()
        ),
        "scenario_count": scenario_count,
        "control_case_count": control_case_count,
        "attack_family_count": int(frame["attack_family"].dropna().nunique()),
        "attack_vector_count": int(frame["attack_vector_id"].dropna().nunique()),
        "vector_observable_profile_count": len(active_atlas.vectors),
        "rail_count": int(frame["payment_rail"].nunique()),
        "channel_count": int(frame["channel"].nunique()),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "family_event_counts": {
            str(key): int(value)
            for key, value in frame.loc[frame["label_fraud"], "attack_family"]
            .value_counts()
            .items()
        },
        "vector_event_counts": {
            str(key): int(value)
            for key, value in frame.loc[frame["label_fraud"], "attack_vector_id"]
            .value_counts()
            .items()
        },
        "novel_holdout_vector_ids": sorted(active_atlas.novel_holdout_vector_ids),
        "schema_fields": list(frame.columns),
        "safety_constraints": [
            "synthetic_entities_only",
            "no_real_payment_execution",
            "sealed_labels_excluded_from_defense_features",
            "entity_and_scenario_isolation_across_splits",
        ],
    }
    return frame, metadata
