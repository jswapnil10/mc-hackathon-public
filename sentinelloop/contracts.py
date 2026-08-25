"""Explicit contracts shared by Red, Blue, the simulator, and the Referee."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BLUE_ACTIONS = {"allow", "monitor", "step_up", "hold", "block"}
RISK_LEVELS = {"low", "medium", "high", "critical"}
BLUE_REASON_CODES = {
    "device_novelty",
    "network_novelty",
    "beneficiary_novelty",
    "amount_anomaly",
    "velocity",
    "graph_concentration",
    "identity_linkage",
    "communication_risk",
    "profile_change",
    "verification_inconsistency",
    "behavior_sequence",
    "legitimate_context",
    "insufficient_evidence",
}
EVIDENCE_TOOLS = {
    "timeline_summary",
    "entity_linkage",
    "velocity_profile",
    "payment_context",
    "legitimate_alternatives",
}


RED_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "attack_family": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "objective": {"type": "string", "minLength": 20, "maxLength": 400},
        "stage_emphasis": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "adaptation_hypothesis": {"type": "string", "minLength": 10, "maxLength": 500},
        "parameter_changes": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "parameter": {"type": "string"},
                    "value": {"type": "number"},
                    "rationale": {"type": "string", "minLength": 5, "maxLength": 220},
                },
                "required": ["parameter", "value", "rationale"],
            },
        },
        "reasoning_summary": {"type": "string", "minLength": 20, "maxLength": 600},
    },
    "required": [
        "attack_family",
        "difficulty",
        "objective",
        "stage_emphasis",
        "adaptation_hypothesis",
        "parameter_changes",
        "reasoning_summary",
    ],
}


BLUE_INVESTIGATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "preliminary_risk": {"type": "string", "maxLength": 360},
        "requested_tools": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(EVIDENCE_TOOLS)},
        },
        "investigation_focus": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 100},
        },
    },
    "required": ["preliminary_risk", "requested_tools", "investigation_focus"],
}


BLUE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "event_id": {"type": "string"},
        "action": {"type": "string", "enum": sorted(BLUE_ACTIONS)},
        "risk_level": {"type": "string", "enum": sorted(RISK_LEVELS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(BLUE_REASON_CODES)},
        },
        "evidence_refs": {
            "type": "array",
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "decision_summary": {"type": "string", "minLength": 10, "maxLength": 420},
        "mitigation": {"type": "string", "minLength": 5, "maxLength": 260},
    },
    "required": [
        "event_id",
        "action",
        "risk_level",
        "confidence",
        "reason_codes",
        "evidence_refs",
        "decision_summary",
        "mitigation",
    ],
}


@dataclass(frozen=True)
class ParameterChange:
    parameter: str
    value: float
    rationale: str


@dataclass(frozen=True)
class RedPlan:
    attack_family: str
    difficulty: str
    objective: str
    stage_emphasis: list[str]
    adaptation_hypothesis: str
    parameter_changes: list[ParameterChange]
    reasoning_summary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RedPlan":
        data = dict(payload)
        data["parameter_changes"] = [ParameterChange(**item) for item in data["parameter_changes"]]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservedEvent:
    """The only event view Blue is permitted to inspect."""

    event_id: str
    sequence: int
    occurred_at: str
    event_type: str
    observable_signals: list[str]
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TruthRecord:
    """Sealed ground truth. Never include this object in a Blue prompt."""

    event_id: str
    stage_id: str
    attack_family: str | None
    scenario_id: str
    intervention_point: str
    is_attack: bool
    value_at_risk_inr: float
    offset_seconds: int
    # Whether this stage is genuinely malicious. Some attack stages are benign in
    # isolation (e.g. ATO novel_session, MULE account_warmup, victim-authorised
    # context): they belong to the kill chain but are labelled non-contributing so a
    # future classifier does not learn "this event_type exists => fraud". Defaults
    # True so existing LLM/referee behaviour is unchanged (nothing in that path reads it).
    fraud_contributing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationCase:
    case_id: str
    events: list[ObservedEvent]
    truth: list[TruthRecord]
    control_name: str | None = None


@dataclass(frozen=True)
class InvestigationRequest:
    preliminary_risk: str
    requested_tools: list[str]
    investigation_focus: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InvestigationRequest":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePacket:
    evidence_id: str
    tool_name: str
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlueDecision:
    event_id: str
    action: str
    risk_level: str
    confidence: float
    reason_codes: list[str]
    evidence_refs: list[str]
    decision_summary: str
    mitigation: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlueDecision":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlueTurn:
    event: ObservedEvent
    investigation: InvestigationRequest
    evidence: list[EvidencePacket]
    decision: BlueDecision
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    policy_adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "investigation": self.investigation.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "decision": self.decision.to_dict(),
            "model_calls": self.model_calls,
            "policy_adjustments": self.policy_adjustments,
        }


@dataclass(frozen=True)
class RefereeReport:
    outcome: str
    detected_stage_id: str | None
    time_to_detect_seconds: int | None
    total_value_at_risk_inr: float
    value_prevented_inr: float
    value_prevented_ratio: float
    hard_false_positive_rate: float
    legitimate_friction_rate: float
    attack_detection_rate: float
    blue_score: float
    red_score: float
    coarse_reason_categories: list[str]
    # False-positive / friction on ordinary standalone legit traffic + hard-negative traps
    # (the realistic FP denominator, distinct from the hard look-alike controls). Defaults 0.0
    # so existing callers that pass no ambient cases are unaffected.
    ambient_false_positive_rate: float = 0.0
    ambient_friction_rate: float = 0.0
    scoring_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
