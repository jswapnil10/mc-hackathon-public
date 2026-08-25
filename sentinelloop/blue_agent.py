"""Tool-using GenAI Blue agent for investigation and proportionate mitigation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import AgentLabConfig
from .contracts import (
    BLUE_ACTIONS,
    BLUE_DECISION_SCHEMA,
    BLUE_INVESTIGATION_SCHEMA,
    BLUE_REASON_CODES,
    EVIDENCE_TOOLS,
    RISK_LEVELS,
    BlueDecision,
    BlueTurn,
    EvidencePacket,
    InvestigationRequest,
    ObservedEvent,
)
from .evidence import EvidenceWorkbench
from .model_gateway import StructuredModelGateway
from .prompts import BLUE_DECIDER_SYSTEM_PROMPT, BLUE_INVESTIGATOR_SYSTEM_PROMPT
from .trace import trace


def _load_detector(config: AgentLabConfig):
    """Load the champion detector if ML fusion is enabled; return (detector, model_hash) or (None, None).

    Never fatal: a missing/unloadable model just falls back to LLM-only Blue with a trace note."""
    if not getattr(config, "ml_detector_enabled", False):
        return None, None
    try:
        import hashlib
        from pathlib import Path

        from .blue_ml.detector import FraudDetector

        detector = FraudDetector.load(config.ml_model_dir)
        model_file = Path(config.ml_model_dir) / "model.joblib"
        model_hash = hashlib.md5(model_file.read_bytes()).hexdigest()[:12] if model_file.exists() else None
        trace("blue.ml.loaded", "ML detector loaded for hybrid fusion.",
              model_dir=str(config.ml_model_dir), threshold=detector.threshold, model_hash=model_hash)
        return detector, model_hash
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to LLM-only
        trace("blue.ml.load_failed", "ML detector could not be loaded; Blue stays LLM-only.", error=str(exc))
        return None, None


class GenAIBlueAgent:
    def __init__(
        self,
        *,
        gateway: StructuredModelGateway,
        config: AgentLabConfig,
        workbench: EvidenceWorkbench | None = None,
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.workbench = workbench or EvidenceWorkbench()
        self.detector, self.model_hash = _load_detector(config)

    def reload_detector(self) -> bool:
        """Re-load the champion from disk (used after a mid-run retrain promotes a new model).
        Returns True if a detector is now active."""
        self.detector, self.model_hash = _load_detector(self.config)
        return self.detector is not None

    def _ml_evidence(
        self, event: ObservedEvent, visible_history: list[ObservedEvent], prior_turns: list[BlueTurn]
    ) -> tuple[EvidencePacket | None, dict[str, Any] | None]:
        """Deterministic ML risk packet for this event: p_t + running-max session risk + threshold.
        Injected as evidence the LLM decider CONSUMES; it never forces the action (evidence-only fusion)."""
        if self.detector is None:
            return None, None
        import pandas as pd

        from .blue_ml.features import FEATURES, build_features
        from .blue_ml.session import cumulative_session_risk

        prior_events = [item.to_dict() for item in visible_history[:-1]]  # strictly-prior
        feats = build_features(event.to_dict(), prior_events)
        p_t = float(self.detector.score(pd.DataFrame([feats], columns=FEATURES))[0])
        cumulative = cumulative_session_risk(p_t, prior_turns)
        threshold = self.detector.threshold
        above = bool(threshold is not None and cumulative >= threshold)
        facts = {
            "per_event_risk": round(p_t, 6),
            "cumulative_session_risk": round(cumulative, 6),
            "alert_threshold": round(float(threshold), 6) if threshold is not None else None,
            "above_threshold": above,
            "model_hash": self.model_hash,
            "note": "Calibrated model risk in [0,1]; treat as evidence, not a truth label.",
        }
        packet = EvidencePacket(
            evidence_id=f"ml_risk_{event.event_id[-8:]}", tool_name="ml_risk_score", facts=facts
        )
        return packet, facts

    @staticmethod
    def _prior_decision_view(turns: list[BlueTurn]) -> list[dict[str, Any]]:
        return [
            {
                "event_id": turn.decision.event_id,
                "action": turn.decision.action,
                "risk_level": turn.decision.risk_level,
                "reason_codes": turn.decision.reason_codes,
                "decision_summary": turn.decision.decision_summary,
            }
            for turn in turns
        ]

    @staticmethod
    def _decision_errors(
        decision: BlueDecision,
        *,
        event: ObservedEvent,
        prior_turns: list[BlueTurn],
        available_evidence: set[str],
    ) -> list[str]:
        errors: list[str] = []
        if decision.event_id != event.event_id:
            errors.append("The decision references a different event.")
        if decision.action not in BLUE_ACTIONS or decision.risk_level not in RISK_LEVELS:
            errors.append("The decision uses an unknown action or risk level.")
        if not 0.0 <= float(decision.confidence) <= 1.0:
            errors.append("Confidence must be between 0 and 1.")
        unknown_reasons = set(decision.reason_codes).difference(BLUE_REASON_CODES)
        if unknown_reasons or not decision.reason_codes:
            errors.append(f"Reason codes are invalid: {sorted(unknown_reasons)}")
        unknown_refs = set(decision.evidence_refs).difference(available_evidence)
        if unknown_refs:
            errors.append(f"Evidence references are unavailable: {sorted(unknown_refs)}")

        allowed_risks = {
            "allow": {"low"},
            "monitor": {"low", "medium", "high"},
            "step_up": {"medium", "high", "critical"},
            "hold": {"high", "critical"},
            "block": {"critical"},
        }
        if decision.risk_level not in allowed_risks.get(decision.action, set()):
            errors.append(
                f"Action {decision.action!r} is inconsistent with risk {decision.risk_level!r}."
            )
        prior_alert = any(
            turn.decision.action in {"step_up", "hold", "block"} for turn in prior_turns
        )
        has_resolving_context = "legitimate_context" in decision.reason_codes
        if prior_alert and not has_resolving_context and decision.risk_level == "low":
            errors.append(
                "An unresolved earlier alert keeps a medium risk floor until resolving context appears."
            )
        unresolved_step_up = any(turn.decision.action == "step_up" for turn in prior_turns)
        value_moving_event = event.event_type in {
            "PAYMENT_INITIATED",
            "PAYMENT_REPEATED",
            "FUNDS_RECEIVED",
            "FUNDS_DISPERSED",
        }
        if (
            unresolved_step_up
            and value_moving_event
            and not has_resolving_context
            and decision.action not in {"hold", "block"}
        ):
            errors.append(
                "An unresolved step-up must escalate to hold or block when value starts moving."
            )
        if decision.action == "allow":
            if prior_alert and not has_resolving_context:
                errors.append(
                    "An unresolved earlier alert may be downgraded only with legitimate_context."
                )
            elif not prior_alert and not {
                "legitimate_context",
                "insufficient_evidence",
            }.intersection(decision.reason_codes):
                errors.append("Allow must cite legitimate_context or insufficient_evidence.")
        return errors

    @staticmethod
    def _normalize_policy_labels(
        decision: BlueDecision, *, prior_turns: list[BlueTurn]
    ) -> tuple[BlueDecision, list[str]]:
        """Correct safe label-only inconsistencies without changing the model's chosen action."""
        adjustments: list[str] = []
        normalized_risk = decision.risk_level
        if decision.action == "allow" and decision.risk_level != "low":
            normalized_risk = "low"
        elif decision.action == "step_up" and decision.risk_level == "low":
            normalized_risk = "medium"
        elif decision.action == "hold" and decision.risk_level in {"low", "medium"}:
            normalized_risk = "high"
        elif decision.action == "block" and decision.risk_level != "critical":
            normalized_risk = "critical"

        prior_alert = any(
            turn.decision.action in {"step_up", "hold", "block"} for turn in prior_turns
        )
        if (
            prior_alert
            and "legitimate_context" not in decision.reason_codes
            and normalized_risk == "low"
            and decision.action != "allow"
        ):
            normalized_risk = "medium"
        if normalized_risk != decision.risk_level:
            adjustments.append(
                f"Promoted risk from {decision.risk_level} to {normalized_risk} to match action {decision.action}."
            )
            decision = replace(decision, risk_level=normalized_risk)
        return decision, adjustments

    def investigate_event(
        self,
        *,
        event: ObservedEvent,
        visible_history: list[ObservedEvent],
        prior_turns: list[BlueTurn],
        seed: int,
    ) -> BlueTurn:
        trace(
            "blue.event.started",
            "Blue received one sanitized observable event.",
            event_id=event.event_id,
            event_type=event.event_type,
            sequence=event.sequence,
            visible_history_count=len(visible_history),
        )
        sanitized_timeline = [item.to_dict() for item in visible_history]
        common = {
            "current_event": event.to_dict(),
            "visible_event_timeline": sanitized_timeline,
            "prior_blue_decisions": self._prior_decision_view(prior_turns),
        }
        investigation_result, investigation_call = self.gateway.generate_json(
            agent_name="blue_investigator",
            model=self.config.blue_model_id,
            system_prompt=BLUE_INVESTIGATOR_SYSTEM_PROMPT,
            user_payload={**common, "approved_tools": sorted(EVIDENCE_TOOLS)},
            schema_name="blue_investigation_request",
            schema=BLUE_INVESTIGATION_SCHEMA,
            temperature=self.config.blue_temperature,
            seed=seed,
        )
        try:
            investigation = InvestigationRequest.from_dict(investigation_result)
        except TypeError as exc:
            raise ValueError("Blue Investigator returned an incomplete request.") from exc
        if not investigation.requested_tools:
            raise ValueError("Blue Investigator must request at least one evidence tool.")
        unknown_tools = set(investigation.requested_tools).difference(EVIDENCE_TOOLS)
        if unknown_tools:
            raise ValueError(f"Blue requested unknown tools: {sorted(unknown_tools)}")
        if len(set(investigation.requested_tools)) != len(investigation.requested_tools):
            raise ValueError("Blue requested duplicate tools.")

        trace(
            "blue.investigator.tools_selected",
            "The Blue Investigator selected read-only evidence tools.",
            event_id=event.event_id,
            tools=investigation.requested_tools,
            model=investigation_call.model,
            latency_ms=investigation_call.latency_ms,
        )

        evidence = self.workbench.run(investigation.requested_tools, visible_history)
        # Always-on ML risk (when enabled): injected as evidence the decider reads, regardless of
        # which tools the investigator requested. Deterministic, so it is comparable round-over-round.
        ml_packet, ml_risk_info = self._ml_evidence(event, visible_history, prior_turns)
        if ml_packet is not None:
            evidence.append(ml_packet)
            trace(
                "blue.ml.scored",
                "ML detector produced a per-event and cumulative session risk.",
                event_id=event.event_id,
                per_event_risk=ml_risk_info["per_event_risk"],
                cumulative_session_risk=ml_risk_info["cumulative_session_risk"],
                above_threshold=ml_risk_info["above_threshold"],
            )
        trace(
            "blue.tools.completed",
            "Deterministic evidence tools returned factual case context.",
            event_id=event.event_id,
            evidence_ids=[item.evidence_id for item in evidence],
        )
        decision_payload = {
            **common,
            "investigation": investigation.to_dict(),
            "tool_evidence": [item.to_dict() for item in evidence],
            "policy": {
                "actions": sorted(BLUE_ACTIONS),
                "risk_levels": sorted(RISK_LEVELS),
                "reason_codes": sorted(BLUE_REASON_CODES),
                "objective": "Prevent attack value while minimizing legitimate customer friction.",
                "risk_continuity": (
                    "Do not downgrade an unresolved prior alert without positive legitimate context. "
                    "Escalate unresolved step-up to hold or block before value moves."
                ),
                "ml_risk_score": (
                    "If an ml_risk_score evidence item is present, weigh its per_event_risk and "
                    "cumulative_session_risk as a calibrated model signal, but corroborate with other "
                    "evidence; it is input, not a truth label, and never by itself forces an action."
                ),
            },
        }
        decision_result, decision_call = self.gateway.generate_json(
            agent_name="blue_decider",
            model=self.config.blue_model_id,
            system_prompt=BLUE_DECIDER_SYSTEM_PROMPT,
            user_payload=decision_payload,
            schema_name="blue_payment_decision",
            schema=BLUE_DECISION_SCHEMA,
            temperature=self.config.blue_temperature,
            seed=seed + 1,
        )
        try:
            decision = BlueDecision.from_dict(decision_result)
        except TypeError as exc:
            raise ValueError("Blue Decision Agent returned an incomplete decision.") from exc
        decision, policy_adjustments = self._normalize_policy_labels(
            decision, prior_turns=prior_turns
        )
        trace(
            "blue.decision.model_complete",
            "The Blue Decision Agent returned an action candidate.",
            event_id=event.event_id,
            action=decision.action,
            risk_level=decision.risk_level,
            reason_codes=decision.reason_codes,
            latency_ms=decision_call.latency_ms,
        )
        available_evidence = {item.evidence_id for item in evidence}
        model_calls = [investigation_call.__dict__, decision_call.__dict__]
        decision_errors = self._decision_errors(
            decision,
            event=event,
            prior_turns=prior_turns,
            available_evidence=available_evidence,
        )
        if decision_errors:
            trace(
                "blue.policy.repair_requested",
                "The deterministic policy gate requested one bounded repair.",
                event_id=event.event_id,
                violations=decision_errors,
            )
            repaired_result, repair_call = self.gateway.generate_json(
                agent_name="blue_decider_repair",
                model=self.config.blue_model_id,
                system_prompt=(
                    f"{BLUE_DECIDER_SYSTEM_PROMPT}\n\nYour previous candidate violated the deterministic "
                    "decision policy. Correct only the final decision using the listed violations."
                ),
                user_payload={
                    **decision_payload,
                    "rejected_candidate": decision.to_dict(),
                    "policy_violations": decision_errors,
                },
                schema_name="blue_payment_decision_repair",
                schema=BLUE_DECISION_SCHEMA,
                temperature=0.0,
                seed=seed + 2,
            )
            model_calls.append(repair_call.__dict__)
            try:
                decision = BlueDecision.from_dict(repaired_result)
            except TypeError as exc:
                raise ValueError("Blue policy repair returned an incomplete decision.") from exc
            decision, repair_adjustments = self._normalize_policy_labels(
                decision, prior_turns=prior_turns
            )
            policy_adjustments.extend(repair_adjustments)
            remaining_errors = self._decision_errors(
                decision,
                event=event,
                prior_turns=prior_turns,
                available_evidence=available_evidence,
            )
            if remaining_errors:
                raise ValueError(f"Blue decision rejected by the policy gate: {remaining_errors}")
        if policy_adjustments:
            trace(
                "blue.policy.adjusted",
                "The policy guard normalized safe label-level inconsistencies.",
                event_id=event.event_id,
                adjustments=policy_adjustments,
            )
        trace(
            "blue.decision.approved",
            "The policy gate approved Blue's final operational decision.",
            event_id=event.event_id,
            action=decision.action,
            risk_level=decision.risk_level,
            mitigation=decision.mitigation,
        )
        return BlueTurn(
            event=event,
            investigation=investigation,
            evidence=evidence,
            decision=decision,
            model_calls=model_calls,
            policy_adjustments=policy_adjustments,
            ml_risk=ml_risk_info,
        )

    def run_case(
        self,
        events: list[ObservedEvent],
        *,
        seed: int,
        stop_on_decisive_action: bool = True,
    ) -> list[BlueTurn]:
        turns: list[BlueTurn] = []
        visible: list[ObservedEvent] = []
        for index, event in enumerate(events):
            visible.append(event)
            turn = self.investigate_event(
                event=event,
                visible_history=list(visible),
                prior_turns=turns,
                seed=seed + index * 10,
            )
            turns.append(turn)
            if stop_on_decisive_action and turn.decision.action in {"hold", "block"}:
                break
        return turns
