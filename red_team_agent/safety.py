"""Fail-closed validation for synthetic Red Team scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .catalog import AttackCatalog
from .models import SafetyReport, ScenarioSpec


ALLOWED_EVENT_TYPES = {
    "COMMUNICATION_RISK_CONTEXT",
    "SESSION_STARTED",
    "AUTHENTICATION_CONTEXT_CHANGED",
    "IDENTITY_APPLICATION_SUBMITTED",
    "IDENTITY_VERIFICATION_ATTEMPTED",
    "ACCOUNT_OPENED",
    "ACCOUNT_WARMUP_ACTIVITY",
    "BENEFICIARY_ADDED",
    "SUPPLIER_PROFILE_CHANGED",
    "PAYMENT_INITIATED",
    "PAYMENT_REPEATED",
    "FUNDS_RECEIVED",
    "FUNDS_DISPERSED",
    "CAMPAIGN_REPLAYED",
}

ALLOWED_INTERVENTION_POINTS = {"PREVENT", "DECIDE", "CONTAIN"}

FORBIDDEN_KEY_FRAGMENTS = {
    "password",
    "credential",
    "otp_value",
    "pin_value",
    "cvv",
    "pan",
    "real_name",
    "phone_number",
    "email_address",
    "message_content",
    "phishing_url",
    "malware",
    "exploit_code",
}

REQUIRED_SAFETY_CONSTRAINTS = {
    "synthetic_entities_only",
    "no_outbound_communications",
    "no_real_payment_rails",
    "no_credentials_or_personal_data",
}


def _walk(value: Any, path: str = "root") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _in_bounds(value: Any, bound: dict[str, Any]) -> bool:
    if "allowed" in bound:
        return value in bound["allowed"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return bound.get("min", value) <= value <= bound.get("max", value)


class ScenarioSafetyGate:
    def __init__(self, catalog: AttackCatalog) -> None:
        self.catalog = catalog

    def validate(self, scenario: ScenarioSpec) -> SafetyReport:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            card = self.catalog.get(scenario.attack_family)
        except KeyError as exc:
            return SafetyReport(False, [str(exc)], [])

        if not scenario.scenario_id.startswith("RT-"):
            errors.append("scenario_id must start with 'RT-'.")
        if scenario.difficulty not in {"easy", "medium", "hard"}:
            errors.append("difficulty must be easy, medium, or hard.")
        if not 1 <= len(scenario.stages) <= 12:
            errors.append("A scenario must contain between 1 and 12 stages.")
        if not scenario.source_refs:
            errors.append("At least one authoritative source reference is required.")

        missing_constraints = REQUIRED_SAFETY_CONSTRAINTS.difference(scenario.safety_constraints)
        if missing_constraints:
            errors.append(f"Missing safety constraints: {sorted(missing_constraints)}")

        sequences = [stage.sequence for stage in scenario.stages]
        if sequences != list(range(1, len(sequences) + 1)):
            errors.append("Stage sequence values must be contiguous and start at 1.")
        offsets = [stage.offset_seconds for stage in scenario.stages]
        if offsets != sorted(offsets):
            errors.append("Stage offsets must be chronological.")

        stage_ids = [stage.stage_id for stage in scenario.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Stage identifiers must be unique within a scenario.")
        approved_stage_ids = {template["stage_id"] for template in card.stage_templates}
        unknown_stage_ids = set(stage_ids).difference(approved_stage_ids)
        if unknown_stage_ids:
            errors.append(f"Scenario contains stages not approved by its attack card: {sorted(unknown_stage_ids)}")

        for stage in scenario.stages:
            if stage.event_type not in ALLOWED_EVENT_TYPES:
                errors.append(f"Stage {stage.stage_id} uses forbidden event_type {stage.event_type!r}.")
            if stage.blue_intervention_point not in ALLOWED_INTERVENTION_POINTS:
                errors.append(
                    f"Stage {stage.stage_id} has invalid intervention point {stage.blue_intervention_point!r}."
                )
            if not stage.observable_signals:
                warnings.append(f"Stage {stage.stage_id} has no observable defensive signals.")

        for parameter, bound in card.parameter_bounds.items():
            if parameter not in scenario.parameters:
                errors.append(f"Required bounded parameter {parameter!r} is missing.")
            elif not _in_bounds(scenario.parameters[parameter], bound):
                errors.append(
                    f"Parameter {parameter!r}={scenario.parameters[parameter]!r} is outside its approved bounds."
                )

        for item_path, item_value in _walk(scenario.to_dict()):
            key = item_path.rsplit(".", 1)[-1].lower()
            if any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS):
                errors.append(f"Forbidden field detected at {item_path}.")
            if key.endswith("_id") and isinstance(item_value, str):
                if key not in {"scenario_id", "parent_scenario_id", "stage_id"} and not item_value.startswith("syn_"):
                    errors.append(f"Entity identifier at {item_path} must use the 'syn_' prefix.")
            if key == "amount_inr" and isinstance(item_value, (int, float)):
                if not 1 <= float(item_value) <= 250_000:
                    errors.append(f"Synthetic amount at {item_path} exceeds the lab safety limit.")

        unbounded_parameters = scenario.parameters.keys() - card.parameter_bounds.keys()
        if unbounded_parameters:
            errors.append(f"Scenario contains unbounded parameters: {sorted(unbounded_parameters)}")

        return SafetyReport(not errors, errors, warnings)
