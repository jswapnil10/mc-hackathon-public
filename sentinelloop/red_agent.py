"""GenAI Red agent with bounded compilation and fail-closed validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from red_team_agent.catalog import AttackCatalog
from red_team_agent.models import LIFECYCLE_PHASES, PlannerDecision, ScenarioSpec
from red_team_agent.planner import ScenarioCompiler, lifecycle_phase_for_template
from red_team_agent.safety import ScenarioSafetyGate

from .config import AgentLabConfig
from .contracts import RED_PLAN_SCHEMA, RedPlan
from .model_gateway import ModelCall, StructuredModelGateway
from .prompts import RED_SYSTEM_PROMPT
from .trace import trace


@dataclass(frozen=True)
class RedTurn:
    plan: RedPlan
    scenario: ScenarioSpec
    model_call: ModelCall

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "scenario": self.scenario.to_dict(),
            "model_call": self.model_call.__dict__,
        }


class GenAIRedAgent:
    def __init__(
        self,
        *,
        gateway: StructuredModelGateway,
        config: AgentLabConfig,
        catalog: AttackCatalog | None = None,
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.catalog = catalog or AttackCatalog()
        self.compiler = ScenarioCompiler(self.catalog)
        self.safety_gate = ScenarioSafetyGate(self.catalog)

    @staticmethod
    def _parameter_refs(value: Any) -> set[str]:
        if isinstance(value, str) and value.startswith("$param."):
            return {value.removeprefix("$param.")}
        if isinstance(value, dict):
            return set().union(*(GenAIRedAgent._parameter_refs(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(GenAIRedAgent._parameter_refs(child) for child in value))
        return set()

    @classmethod
    def _card_payload(cls, card: Any) -> dict[str, Any]:
        return {
            "attack_family": card.attack_family,
            "name": card.name,
            "observed_pattern": card.observed_pattern,
            "genai_role": card.genai_role,
            "payment_surface": card.payment_surface,
            "source_refs": card.source_refs,
            "stages": [
                {
                    "stage_id": stage["stage_id"],
                    "lifecycle_phase": lifecycle_phase_for_template(stage),
                    "event_type": stage["event_type"],
                    "intervention_point": stage["blue_intervention_point"],
                    "observable_signals": stage["observable_signals"],
                    "mutable_parameters": sorted(
                        cls._parameter_refs(stage.get("attributes", {})).intersection(
                            card.allowed_mutations
                        )
                    ),
                }
                for stage in card.stage_templates
            ],
            "parameter_profiles": card.parameter_profiles,
            "parameter_bounds": card.parameter_bounds,
            "allowed_mutations": card.allowed_mutations,
            "legitimate_controls": card.legitimate_controls,
        }

    def _validate_plan(
        self,
        plan: RedPlan,
        *,
        requested_family: str | None,
        requested_difficulty: str,
    ) -> tuple[Any, dict[str, Any]]:
        if plan.attack_family not in self.catalog.families:
            raise ValueError(f"Red selected unknown attack family {plan.attack_family!r}.")
        if requested_family and plan.attack_family != requested_family:
            raise ValueError("Red changed the user-bounded attack family.")
        if plan.difficulty != requested_difficulty:
            raise ValueError("Red changed the user-bounded difficulty.")
        card = self.catalog.get(plan.attack_family)
        stage_by_id = {stage["stage_id"]: stage for stage in card.stage_templates}
        allowed_stages = set(stage_by_id)
        unknown_stages = set(plan.focus_stage_ids).difference(allowed_stages)
        if unknown_stages:
            raise ValueError(f"Red focused on unknown stages: {sorted(unknown_stages)}")
        if not plan.focus_stage_ids:
            raise ValueError("Red must select at least one focus stage.")
        if plan.target_lifecycle_phase not in LIFECYCLE_PHASES:
            raise ValueError("Red selected an unknown lifecycle phase.")
        wrong_phase = [
            stage_id
            for stage_id in plan.focus_stage_ids
            if lifecycle_phase_for_template(stage_by_id[stage_id]) != plan.target_lifecycle_phase
        ]
        if wrong_phase:
            raise ValueError(
                f"Red focus stages do not belong to {plan.target_lifecycle_phase}: {wrong_phase}"
            )
        focus_parameters = set().union(
            *(
                self._parameter_refs(stage_by_id[stage_id].get("attributes", {}))
                for stage_id in plan.focus_stage_ids
            )
        ).intersection(card.allowed_mutations)
        if not focus_parameters:
            raise ValueError("Red selected focus stages with no bounded mutable behavior.")

        overrides = copy.deepcopy(card.parameter_profiles[plan.difficulty])
        seen_parameters: set[str] = set()
        for change in plan.parameter_changes:
            if change.parameter in seen_parameters:
                raise ValueError(f"Red repeated parameter {change.parameter!r}.")
            seen_parameters.add(change.parameter)
            if change.parameter not in card.allowed_mutations:
                raise ValueError(f"Red attempted a forbidden mutation: {change.parameter!r}.")
            bound = card.parameter_bounds[change.parameter]
            if "min" not in bound or "max" not in bound:
                raise ValueError(f"Red mutation {change.parameter!r} has no numeric safety bound.")
            value = float(change.value)
            if not float(bound["min"]) <= value <= float(bound["max"]):
                raise ValueError(
                    f"Red mutation {change.parameter!r}={value} is outside {bound}."
                )
            original = overrides[change.parameter]
            overrides[change.parameter] = int(round(value)) if isinstance(original, int) else value
        unrelated_changes = seen_parameters.difference(focus_parameters)
        if unrelated_changes:
            raise ValueError(
                f"Red mutations are not tied to its focus stages: {sorted(unrelated_changes)}"
            )
        return card, overrides

    def plan(
        self,
        *,
        attack_family: str | None,
        difficulty: str,
        seed: int,
        previous_scenario: ScenarioSpec | None = None,
        referee_feedback: dict[str, Any] | None = None,
    ) -> RedTurn:
        trace(
            "red.plan.started",
            "Red is preparing a bounded planning request.",
            requested_attack_family=attack_family,
            difficulty=difficulty,
            seed=seed,
            has_previous_scenario=previous_scenario is not None,
            has_referee_feedback=referee_feedback is not None,
        )
        if attack_family is not None:
            cards = [self.catalog.get(attack_family)]
        else:
            cards = self.catalog.list()
        schema = copy.deepcopy(RED_PLAN_SCHEMA)
        schema["properties"]["attack_family"]["enum"] = [card.attack_family for card in cards]
        available_stages = [stage for card in cards for stage in self._card_payload(card)["stages"]]
        available_focus_stages = [
            stage for stage in available_stages if stage["mutable_parameters"]
        ]
        schema["properties"]["focus_stage_ids"]["items"]["enum"] = [
            stage["stage_id"] for stage in available_focus_stages
        ]
        schema["properties"]["target_lifecycle_phase"]["enum"] = sorted(
            {stage["lifecycle_phase"] for stage in available_focus_stages}
        )
        payload = {
            "task": "Create the next bounded synthetic campaign plan.",
            "requested_attack_family": attack_family,
            "requested_difficulty": difficulty,
            "seed": seed,
            "attack_cards": [self._card_payload(card) for card in cards],
            "previous_campaign": (
                {
                    "attack_family": previous_scenario.attack_family,
                    "difficulty": previous_scenario.difficulty,
                    "parameters": previous_scenario.parameters,
                    "target_lifecycle_phase": previous_scenario.target_lifecycle_phase,
                    "focus_stage_ids": previous_scenario.focus_stage_ids,
                    "adaptation_goal": previous_scenario.adaptation_goal,
                    "mutation_number": previous_scenario.mutation_number,
                }
                if previous_scenario
                else None
            ),
            "referee_feedback": referee_feedback,
            "output_contract": schema,
        }
        result, model_trace = self.gateway.generate_json(
            agent_name="red_planner",
            model=self.config.red_model_id,
            system_prompt=RED_SYSTEM_PROMPT,
            user_payload=payload,
            schema_name="red_campaign_plan",
            schema=schema,
            temperature=self.config.red_temperature,
            seed=seed,
        )
        try:
            plan = RedPlan.from_dict(result)
        except (KeyError, TypeError) as exc:
            raise ValueError("Red returned an incomplete campaign plan.") from exc
        _, overrides = self._validate_plan(
            plan,
            requested_family=attack_family,
            requested_difficulty=difficulty,
        )
        trace(
            "red.plan.model_complete",
            "Qwen returned a structured Red campaign plan.",
            attack_family=plan.attack_family,
            target_lifecycle_phase=plan.target_lifecycle_phase,
            focus_stage_ids=plan.focus_stage_ids,
            parameter_changes=[change.parameter for change in plan.parameter_changes],
            model=model_trace.model,
            latency_ms=model_trace.latency_ms,
        )
        mutation_number = previous_scenario.mutation_number + 1 if previous_scenario else 0
        decision = PlannerDecision(
            attack_family=plan.attack_family,
            difficulty=plan.difficulty,
            objective=plan.objective,
            target_lifecycle_phase=plan.target_lifecycle_phase,
            focus_stage_ids=plan.focus_stage_ids,
            adaptation_goal=plan.adaptation_goal,
            reasoning_summary=f"{plan.reasoning_summary} Adaptation: {plan.adaptation_hypothesis}",
            backend=f"open-model:{self.config.red_model_id}",
        )
        scenario = self.compiler.compile(
            decision,
            seed=seed,
            parameter_overrides=overrides,
            parent_scenario_id=previous_scenario.scenario_id if previous_scenario else None,
            mutation_number=mutation_number,
            mutation_reason=[change.rationale for change in plan.parameter_changes],
        )
        report = self.safety_gate.validate(scenario)
        if not report.approved:
            raise ValueError(f"Red campaign rejected by the safety gate: {report.errors}")
        trace(
            "red.safety.approved",
            "The deterministic safety gate approved the compiled campaign.",
            scenario_id=scenario.scenario_id,
            stage_count=len(scenario.stages),
            warnings=report.warnings,
        )
        return RedTurn(plan=plan, scenario=scenario, model_call=model_trace)
