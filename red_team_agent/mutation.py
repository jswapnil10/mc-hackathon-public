"""Bounded feedback-driven mutation for the next synthetic episode."""

from __future__ import annotations

from typing import Any

from .catalog import AttackCatalog
from .models import PlannerDecision, RefereeFeedback, ScenarioSpec
from .planner import ScenarioCompiler
from .safety import ScenarioSafetyGate


REASON_TO_PARAMETERS = {
    "device_novelty": ["new_device_probability"],
    "network_novelty": ["new_network_probability"],
    "beneficiary_novelty": ["beneficiary_age_days"],
    "amount_anomaly": ["amount_multiplier"],
    "velocity": ["campaign_duration_minutes", "payment_interval_minutes"],
    "graph_concentration": ["beneficiary_count", "mule_count"],
    "identity_linkage": ["linked_attribute_count"],
    "communication_risk": ["urgency_level"],
}

ALLOWED_OUTCOMES = {"prevented", "detected", "contained", "missed"}

INCREASE_WHEN_DETECTED = {
    "beneficiary_age_days",
    "campaign_duration_minutes",
    "payment_interval_minutes",
    "beneficiary_count",
    "mule_count",
}


def _mutate_numeric(value: float, bound: dict[str, Any], *, increase: bool) -> float | int:
    minimum = float(bound["min"])
    maximum = float(bound["max"])
    step = max((maximum - minimum) * 0.20, 1.0 if isinstance(value, int) else 0.01)
    mutated = min(maximum, value + step) if increase else max(minimum, value - step)
    return int(round(mutated)) if isinstance(value, int) else round(mutated, 4)


class MutationController:
    def __init__(self, catalog: AttackCatalog | None = None) -> None:
        self.catalog = catalog or AttackCatalog()
        self.compiler = ScenarioCompiler(self.catalog)
        self.safety_gate = ScenarioSafetyGate(self.catalog)

    def mutate(self, scenario: ScenarioSpec, feedback: RefereeFeedback) -> ScenarioSpec:
        feedback_errors: list[str] = []
        if feedback.outcome not in ALLOWED_OUTCOMES:
            feedback_errors.append(f"Unknown Referee outcome {feedback.outcome!r}.")
        if not 0.0 <= feedback.value_prevented_ratio <= 1.0:
            feedback_errors.append("value_prevented_ratio must be between 0 and 1.")
        if not 0.0 <= feedback.false_positive_rate <= 1.0:
            feedback_errors.append("false_positive_rate must be between 0 and 1.")
        if feedback.time_to_detect_seconds is not None and feedback.time_to_detect_seconds < 0:
            feedback_errors.append("time_to_detect_seconds cannot be negative.")
        known_stage_ids = {stage.stage_id for stage in scenario.stages}
        if feedback.detected_stage_id is not None and feedback.detected_stage_id not in known_stage_ids:
            feedback_errors.append("detected_stage_id is not present in the parent scenario.")
        unknown_categories = set(feedback.coarse_reason_categories).difference(REASON_TO_PARAMETERS)
        if unknown_categories:
            feedback_errors.append(f"Unknown coarse reason categories: {sorted(unknown_categories)}")
        if feedback_errors:
            raise ValueError(f"Referee feedback rejected: {feedback_errors}")

        card = self.catalog.get(scenario.attack_family)
        overrides = dict(scenario.parameters)
        mutation_reasons: list[str] = []
        candidates: list[str] = []
        for reason in feedback.coarse_reason_categories:
            candidates.extend(REASON_TO_PARAMETERS.get(reason, []))

        for parameter in candidates:
            if parameter not in card.allowed_mutations or parameter not in overrides:
                continue
            bound = card.parameter_bounds.get(parameter, {})
            value = overrides[parameter]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and "min" in bound and "max" in bound:
                overrides[parameter] = _mutate_numeric(
                    value,
                    bound,
                    increase=parameter in INCREASE_WHEN_DETECTED,
                )
                mutation_reasons.append(f"Adjusted {parameter} after coarse signal feedback.")

        if not mutation_reasons:
            for parameter in card.allowed_mutations:
                if parameter not in overrides:
                    continue
                bound = card.parameter_bounds.get(parameter, {})
                value = overrides[parameter]
                if isinstance(value, (int, float)) and not isinstance(value, bool) and "min" in bound and "max" in bound:
                    overrides[parameter] = _mutate_numeric(value, bound, increase=False)
                    mutation_reasons.append(f"Adjusted {parameter} after {feedback.outcome} outcome.")
                    break

        decision = PlannerDecision(
            attack_family=scenario.attack_family,
            difficulty=scenario.difficulty,
            objective=scenario.objective,
            target_lifecycle_phase=scenario.target_lifecycle_phase,
            focus_stage_ids=list(scenario.focus_stage_ids),
            adaptation_goal=scenario.adaptation_goal,
            reasoning_summary=(
                f"Mutation {scenario.mutation_number + 1} used only the Referee's coarse categories: "
                f"{', '.join(feedback.coarse_reason_categories) or 'no named category'}."
            ),
            backend="bounded-mutation-controller",
        )
        mutated = self.compiler.compile(
            decision,
            seed=scenario.seed + 1,
            parameter_overrides=overrides,
            parent_scenario_id=scenario.scenario_id,
            mutation_number=scenario.mutation_number + 1,
            mutation_reason=mutation_reasons,
        )
        report = self.safety_gate.validate(mutated)
        if not report.approved:
            raise ValueError(f"Mutated scenario rejected by safety gate: {report.errors}")
        return mutated
