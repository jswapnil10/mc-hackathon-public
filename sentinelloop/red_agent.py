"""GenAI Red agent with bounded compilation and fail-closed validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from red_team_agent.catalog import AttackCatalog
from red_team_agent.models import PlannerDecision, ScenarioSpec
from red_team_agent.planner import ScenarioCompiler
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
    def _card_payload(card: Any) -> dict[str, Any]:
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
                    "event_type": stage["event_type"],
                    "intervention_point": stage["blue_intervention_point"],
                    "observable_signals": stage["observable_signals"],
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
        allowed_stages = {stage["stage_id"] for stage in card.stage_templates}
        unknown_stages = set(plan.stage_emphasis).difference(allowed_stages)
        if unknown_stages:
            raise ValueError(f"Red emphasized unknown stages: {sorted(unknown_stages)}")

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
                    "stage_ids": [stage.stage_id for stage in previous_scenario.stages],
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
            stage_emphasis=plan.stage_emphasis,
            parameter_changes=[change.parameter for change in plan.parameter_changes],
            model=model_trace.model,
            latency_ms=model_trace.latency_ms,
        )
        mutation_number = previous_scenario.mutation_number + 1 if previous_scenario else 0
        decision = PlannerDecision(
            attack_family=plan.attack_family,
            difficulty=plan.difficulty,
            objective=plan.objective,
            stage_emphasis=plan.stage_emphasis,
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
