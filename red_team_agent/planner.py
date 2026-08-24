"""Agent orchestration and deterministic scenario compilation."""

from __future__ import annotations

import copy
import random
from typing import Any, Protocol

from .catalog import AttackCatalog
from .models import AttackCard, PlannerDecision, ScenarioSpec, ScenarioStage
from .safety import ScenarioSafetyGate


SAFETY_CONSTRAINTS = [
    "synthetic_entities_only",
    "no_outbound_communications",
    "no_real_payment_rails",
    "no_credentials_or_personal_data",
    "defensive_observables_only",
    "deterministic_replay_with_seed",
]


class PlanningBackend(Protocol):
    name: str

    def choose(
        self,
        *,
        attack_family: str | None,
        difficulty: str,
        objective: str | None,
        seed: int,
    ) -> PlannerDecision: ...


class OfflinePlanningBackend:
    """Deterministic fallback used for tests and API-key-free demonstrations."""

    name = "offline-policy"

    def __init__(self, catalog: AttackCatalog) -> None:
        self.catalog = catalog

    def choose(
        self,
        *,
        attack_family: str | None,
        difficulty: str,
        objective: str | None,
        seed: int,
    ) -> PlannerDecision:
        rng = random.Random(seed)
        family = attack_family or rng.choice(self.catalog.families)
        card = self.catalog.get(family)
        available = [stage["stage_id"] for stage in card.stage_templates]
        emphasis_count = min(3, len(available))
        emphasis = sorted(rng.sample(available, k=emphasis_count)) if emphasis_count else []
        selected_objective = objective or (
            f"Stress-test whether Blue can recognize and mitigate the synthetic {card.name} "
            f"campaign before high-risk value is released."
        )
        return PlannerDecision(
            attack_family=family,
            difficulty=difficulty,
            objective=selected_objective,
            stage_emphasis=emphasis,
            reasoning_summary=(
                f"Selected the curated {family} card at {difficulty} difficulty and emphasized "
                f"{', '.join(emphasis) or 'the complete campaign'} for a reproducible safe test."
            ),
            backend=self.name,
        )


def _resolve_template_value(value: Any, parameters: dict[str, Any], synthetic_ids: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$param."):
        return parameters[value.removeprefix("$param.")]
    if isinstance(value, str) and value.startswith("$synthetic."):
        return synthetic_ids[value.removeprefix("$synthetic.")]
    if isinstance(value, dict):
        return {key: _resolve_template_value(child, parameters, synthetic_ids) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_template_value(child, parameters, synthetic_ids) for child in value]
    return value


class ScenarioCompiler:
    def __init__(self, catalog: AttackCatalog) -> None:
        self.catalog = catalog

    def compile(
        self,
        decision: PlannerDecision,
        *,
        seed: int,
        parameter_overrides: dict[str, Any] | None = None,
        parent_scenario_id: str | None = None,
        mutation_number: int = 0,
        mutation_reason: list[str] | None = None,
    ) -> ScenarioSpec:
        card = self.catalog.get(decision.attack_family)
        if decision.difficulty not in card.parameter_profiles:
            raise ValueError(f"Attack card {card.attack_family} does not define {decision.difficulty!r} difficulty.")
        parameters = copy.deepcopy(card.parameter_profiles[decision.difficulty])
        parameters.update(parameter_overrides or {})
        rng = random.Random(seed)
        synthetic_ids = {
            "sender": f"syn_sender_{rng.randint(100000, 999999)}",
            "beneficiary": f"syn_beneficiary_{rng.randint(100000, 999999)}",
            "device": f"syn_device_{rng.randint(100000, 999999)}",
            "network": f"syn_network_{rng.randint(100000, 999999)}",
            "merchant": f"syn_merchant_{rng.randint(100000, 999999)}",
            "identity": f"syn_identity_{rng.randint(100000, 999999)}",
            "supplier": f"syn_supplier_{rng.randint(100000, 999999)}",
        }

        stages: list[ScenarioStage] = []
        for sequence, template in enumerate(card.stage_templates, start=1):
            attributes = _resolve_template_value(template.get("attributes", {}), parameters, synthetic_ids)
            if template["stage_id"] in decision.stage_emphasis:
                attributes["planner_emphasis"] = True
            stages.append(
                ScenarioStage(
                    stage_id=template["stage_id"],
                    sequence=sequence,
                    event_type=template["event_type"],
                    offset_seconds=int(template["offset_seconds"]),
                    blue_intervention_point=template["blue_intervention_point"],
                    observable_signals=list(template["observable_signals"]),
                    attributes=attributes,
                )
            )

        scenario_suffix = f"M{mutation_number:02d}" if mutation_number else "BASE"
        scenario_id = f"RT-{card.attack_family}-{seed:08d}-{scenario_suffix}"
        return ScenarioSpec(
            schema_version="1.0",
            scenario_id=scenario_id,
            attack_family=card.attack_family,
            title=f"{card.name} — {decision.difficulty} synthetic campaign",
            objective=decision.objective,
            difficulty=decision.difficulty,
            seed=seed,
            source_refs=copy.deepcopy(card.source_refs),
            parameters=parameters,
            stages=stages,
            legitimate_controls=list(card.legitimate_controls),
            safety_constraints=list(SAFETY_CONSTRAINTS),
            created_by=decision.backend,
            reasoning_summary=decision.reasoning_summary,
            parent_scenario_id=parent_scenario_id,
            mutation_number=mutation_number,
            mutation_reason=list(mutation_reason or []),
        )


class RedTeamAgent:
    """Plan, compile, and validate safe synthetic attack campaigns."""

    def __init__(self, *, catalog: AttackCatalog | None = None, backend: PlanningBackend | None = None) -> None:
        self.catalog = catalog or AttackCatalog()
        self.backend = backend or OfflinePlanningBackend(self.catalog)
        self.compiler = ScenarioCompiler(self.catalog)
        self.safety_gate = ScenarioSafetyGate(self.catalog)

    def plan(
        self,
        *,
        attack_family: str | None = None,
        difficulty: str = "medium",
        objective: str | None = None,
        seed: int = 20260819,
    ) -> ScenarioSpec:
        decision = self.backend.choose(
            attack_family=attack_family,
            difficulty=difficulty,
            objective=objective,
            seed=seed,
        )
        scenario = self.compiler.compile(decision, seed=seed)
        report = self.safety_gate.validate(scenario)
        if not report.approved:
            raise ValueError(f"Scenario rejected by safety gate: {report.errors}")
        return scenario
