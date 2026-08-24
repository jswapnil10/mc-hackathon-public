"""Data contracts shared by the Red Team planner, compiler, and safety gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AttackCard:
    attack_family: str
    name: str
    observed_pattern: str
    genai_role: list[str]
    payment_surface: list[str]
    source_refs: list[dict[str, str]]
    stage_templates: list[dict[str, Any]]
    parameter_profiles: dict[str, dict[str, Any]]
    parameter_bounds: dict[str, dict[str, Any]]
    allowed_mutations: list[str]
    legitimate_controls: list[str]


@dataclass(frozen=True)
class PlannerDecision:
    attack_family: str
    difficulty: str
    objective: str
    stage_emphasis: list[str]
    reasoning_summary: str
    backend: str


@dataclass(frozen=True)
class ScenarioStage:
    stage_id: str
    sequence: int
    event_type: str
    offset_seconds: int
    blue_intervention_point: str
    observable_signals: list[str]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    schema_version: str
    scenario_id: str
    attack_family: str
    title: str
    objective: str
    difficulty: str
    seed: int
    source_refs: list[dict[str, str]]
    parameters: dict[str, Any]
    stages: list[ScenarioStage]
    legitimate_controls: list[str]
    safety_constraints: list[str]
    created_by: str
    reasoning_summary: str
    parent_scenario_id: str | None = None
    mutation_number: int = 0
    mutation_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioSpec":
        data = dict(payload)
        data["stages"] = [ScenarioStage(**stage) for stage in data.get("stages", [])]
        return cls(**data)


@dataclass(frozen=True)
class RefereeFeedback:
    outcome: str
    detected_stage_id: str | None
    value_prevented_ratio: float
    time_to_detect_seconds: int | None
    coarse_reason_categories: list[str]
    false_positive_rate: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefereeFeedback":
        return cls(**payload)


@dataclass(frozen=True)
class SafetyReport:
    approved: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PLANNER_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "attack_family": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "objective": {"type": "string", "minLength": 10, "maxLength": 400},
        "stage_emphasis": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "reasoning_summary": {"type": "string", "minLength": 10, "maxLength": 600},
    },
    "required": [
        "attack_family",
        "difficulty",
        "objective",
        "stage_emphasis",
        "reasoning_summary",
    ],
}
