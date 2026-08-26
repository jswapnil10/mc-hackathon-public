"""Tool-using GenAI Blue agent for investigation and proportionate mitigation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import AgentLabConfig
from .contracts import (
    BLUE_ACTIONS,
    BLUE_DECISION_SCHEMA,
    BLUE_EVENT_RESPONSE_SCHEMA,
    BLUE_REASON_CODES,
    RISK_LEVELS,
    BlueDecision,
    BlueTurn,
    DefensePlaybook,
    EvidencePacket,
    InvestigationRequest,
    ObservedEvent,
)
from .evidence import EvidenceWorkbench, synthesize_case_risk
from .model_gateway import StructuredModelGateway
from .prompts import BLUE_DECIDER_SYSTEM_PROMPT, BLUE_EVENT_AGENT_SYSTEM_PROMPT
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
            "note": "Model score in [0,1]; not probability-calibrated and never a truth label.",
        }
        packet = EvidencePacket(
            evidence_id=f"ml_risk_{event.event_id[-8:]}",
            tool_name="ml_risk_score",
            facts=facts,
            source="ml_detector",
            as_of_event_id=event.event_id,
            confidence=round(p_t, 6),
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
    def _unresolved_alert_state(turns: list[BlueTurn]) -> str | None:
        state: str | None = None
        for turn in turns:
            if (
                "legitimate_context" in turn.decision.reason_codes
                and turn.decision.action in {"allow", "monitor", "step_up"}
            ):
                state = None
            elif turn.decision.action in {"step_up", "hold", "block"}:
                state = turn.decision.action
        return state

    @staticmethod
    def _route_evidence_tools(
        event: ObservedEvent, playbook: DefensePlaybook
    ) -> list[str]:
        """Choose a small evidence bundle without spending a model round trip."""
        event_type = event.event_type.upper()
        routed: list[str] = [
            "case_risk_synthesis",
            "legitimate_alternatives",
            "timeline_summary",
        ]
        if any(token in event_type for token in ("AUTH", "LOGIN", "SESSION", "PROFILE")):
            routed.extend(["behavioral_biometrics", "evidence_quality"])
        if any(token in event_type for token in ("BENEFICIARY", "FUNDS", "PAYOUT", "AGENTIC")):
            routed.extend(["entity_linkage", "velocity_profile"])
        if any(token in event_type for token in ("PAYMENT", "FUNDS", "PAYOUT", "REFUND")):
            routed.extend(["payment_context", "velocity_profile"])
        if "COMMUNICATION" in event_type:
            routed.extend(["communication_risk", "entity_linkage"])
        if any(token in event_type for token in ("DISPUTE", "DOCUMENT", "EVIDENCE")):
            routed.extend(["evidence_quality", "payment_context"])
        routed.extend(playbook.preferred_tools)
        return list(dict.fromkeys(routed))[:7]

    @staticmethod
    def _decision_errors(
        decision: BlueDecision,
        *,
        event: ObservedEvent,
        prior_turns: list[BlueTurn],
        available_evidence: set[str],
        risk_synthesis: dict[str, Any] | None = None,
        legitimate_context_verified: bool = False,
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
        unresolved_alert = GenAIBlueAgent._unresolved_alert_state(prior_turns)
        prior_alert = unresolved_alert is not None
        has_resolving_context = "legitimate_context" in decision.reason_codes
        if has_resolving_context and not legitimate_context_verified:
            errors.append(
                "legitimate_context is not backed by independent verification or established history."
            )
        action_rank = {"allow": 0, "monitor": 1, "step_up": 2, "hold": 3, "block": 4}
        minimum_key = (
            "minimum_action"
            if has_resolving_context and legitimate_context_verified
            else "operational_minimum_action"
        )
        minimum_action = str(
            (risk_synthesis or {}).get(
                minimum_key, (risk_synthesis or {}).get("minimum_action", "allow")
            )
        )
        if action_rank.get(decision.action, -1) < action_rank.get(minimum_action, 0):
            errors.append(
                f"Action {decision.action!r} is below the observable sequence-risk floor {minimum_action!r}."
            )
        if prior_alert and not has_resolving_context and decision.risk_level == "low":
            errors.append(
                "An unresolved earlier alert keeps a medium risk floor until resolving context appears."
            )
        unresolved_step_up = unresolved_alert == "step_up"
        value_moving_event = event.event_type in {
            "PAYMENT_INITIATED",
            "AGENTIC_PAYMENT_INITIATED",
            "PAYMENT_REPEATED",
            "FUNDS_RECEIVED",
            "FUNDS_DISPERSED",
            "PAYOUT_REQUESTED",
            "PAYOUT_SETTLED",
            "DISPUTE_REFUND_ISSUED",
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
        unresolved_hold = unresolved_alert == "hold"
        if (
            unresolved_hold
            and value_moving_event
            and not has_resolving_context
            and decision.action not in {"hold", "block"}
        ):
            errors.append(
                "An unresolved hold must remain hold or escalate to block while value is moving."
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
            if minimum_action in {"step_up", "hold", "block"}:
                errors.append("Allow cannot override a converging sequence-risk floor.")
        return errors

    @staticmethod
    def _apply_fast_path_guard(
        decision: BlueDecision,
        *,
        risk_synthesis: dict[str, Any],
        legitimate_context_verified: bool,
        risk_evidence_id: str,
    ) -> tuple[BlueDecision, list[str]]:
        """Enforce the fast observable-only minimum without inventing a fraud label."""
        adjustments: list[str] = []
        reasons = list(decision.reason_codes)
        if "legitimate_context" in reasons and not legitimate_context_verified:
            reasons = [reason for reason in reasons if reason != "legitimate_context"]
            adjustments.append(
                "Removed unsupported legitimate_context; same-case continuity is not verified history."
            )

        action_rank = {"allow": 0, "monitor": 1, "step_up": 2, "hold": 3, "block": 4}
        minimum_key = (
            "minimum_action"
            if "legitimate_context" in reasons and legitimate_context_verified
            else "operational_minimum_action"
        )
        minimum_action = str(
            risk_synthesis.get(minimum_key, risk_synthesis.get("minimum_action", "allow"))
        )
        action = decision.action
        risk_level = decision.risk_level
        summary = decision.decision_summary
        mitigation = decision.mitigation
        evidence_refs = list(decision.evidence_refs)
        if action_rank.get(action, -1) < action_rank.get(minimum_action, 0):
            original_action = action
            action = minimum_action
            risk_level = str(risk_synthesis.get("risk_band", "medium"))
            reasons = [reason for reason in reasons if reason != "insufficient_evidence"]
            if "cross_phase_progression" in {
                item.get("code") for item in risk_synthesis.get("indicators", [])
            }:
                guard_reason = "cross_phase_pattern"
            else:
                guard_reason = "behavior_sequence"
            if guard_reason not in reasons:
                reasons.append(guard_reason)
            reasons = reasons[:5]
            if risk_evidence_id not in evidence_refs:
                evidence_refs.append(risk_evidence_id)
            evidence_refs = evidence_refs[:5]
            summary = (
                f"The fast sequence guard raised {original_action} to {action}: "
                f"{risk_synthesis.get('indicator_count', 0)} observable indicators accumulated "
                "without sufficient independent resolution."
            )
            mitigation = (
                "Apply the fast-path minimum action now; continue the GenAI investigation for "
                "explanation, corroboration, and safe resolution."
            )
            adjustments.append(
                f"Raised action from {original_action} to {action} using the pre-model sequence-risk floor."
            )
        if not reasons:
            reasons = ["behavior_sequence"]
        return (
            replace(
                decision,
                action=action,
                risk_level=risk_level,
                reason_codes=reasons,
                evidence_refs=evidence_refs,
                decision_summary=summary,
                mitigation=mitigation,
            ),
            adjustments,
        )

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

        prior_alert = GenAIBlueAgent._unresolved_alert_state(prior_turns) is not None
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
        playbook: DefensePlaybook,
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
        risk_synthesis = synthesize_case_risk(visible_history)
        action_rank = {"allow": 0, "monitor": 1, "step_up": 2, "hold": 3, "block": 4}
        unresolved_alert = self._unresolved_alert_state(prior_turns)
        value_moving_event = event.event_type in {
            "PAYMENT_INITIATED",
            "AGENTIC_PAYMENT_INITIATED",
            "PAYMENT_REPEATED",
            "FUNDS_RECEIVED",
            "FUNDS_DISPERSED",
            "PAYOUT_REQUESTED",
            "PAYOUT_SETTLED",
            "DISPUTE_REFUND_ISSUED",
        }
        continuity_minimum = "allow"
        if (
            unresolved_alert in {"step_up", "hold"}
            and value_moving_event
        ):
            continuity_minimum = "hold"
        sequence_minimum = str(risk_synthesis["minimum_action"])
        operational_minimum = max(
            (sequence_minimum, continuity_minimum), key=lambda action: action_rank[action]
        )
        risk_synthesis = {
            **risk_synthesis,
            "continuity_minimum_action": continuity_minimum,
            "operational_minimum_action": operational_minimum,
        }
        trace(
            "blue.fast_path.ready",
            "The observable-only sequence guard produced a pre-model minimum action.",
            event_id=event.event_id,
            risk_score=risk_synthesis["observable_risk_score"],
            minimum_action=risk_synthesis["operational_minimum_action"],
            indicator_count=risk_synthesis["indicator_count"],
        )
        common = {
            "current_event": event.to_dict(),
            "visible_event_timeline": sanitized_timeline,
            "prior_blue_decisions": self._prior_decision_view(prior_turns),
            "active_defense_playbook": playbook.to_dict(),
            "pre_model_sequence_guard": risk_synthesis,
        }
        selected_tools = self._route_evidence_tools(event, playbook)
        evidence = self.workbench.run(selected_tools, visible_history)
        # Always-on ML risk (when ML_DETECTOR_ENABLED): a deterministic ml_risk_score EvidencePacket
        # the single-call Blue agent consumes. ml_risk_info is None when disabled (LLM-only Blue).
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
        risk_packet = next(item for item in evidence if item.tool_name == "case_risk_synthesis")
        legitimacy_packet = next(
            item for item in evidence if item.tool_name == "legitimate_alternatives"
        )
        legitimate_context_verified = bool(
            legitimacy_packet.facts.get("independent_verification_found")
        )
        trace(
            "blue.tools.completed",
            "The deterministic router prepared a compact evidence bundle before the model call.",
            event_id=event.event_id,
            tools=selected_tools,
            evidence_ids=[item.evidence_id for item in evidence],
        )
        if (
            legitimate_context_verified
            and risk_synthesis["operational_minimum_action"] == "allow"
            and unresolved_alert is None
            and not (ml_risk_info and ml_risk_info.get("above_threshold"))
        ):
            investigation = InvestigationRequest(
                preliminary_risk=(
                    "Independent verification or established history resolves the observable "
                    "ambiguity without a model escalation."
                ),
                requested_tools=selected_tools,
                investigation_focus=["verified context", "sequence floor"],
            )
            decision = BlueDecision(
                event_id=event.event_id,
                action="allow",
                risk_level="low",
                confidence=0.98,
                reason_codes=["legitimate_context"],
                evidence_refs=[legitimacy_packet.evidence_id, risk_packet.evidence_id],
                decision_summary=(
                    "Independent verification or established history supports this event, and "
                    "the observable sequence has no unresolved risk floor."
                ),
                mitigation=(
                    "Allow the event, retain the verification evidence, and continue normal monitoring."
                ),
            )
            trace(
                "blue.verified_context.fast_exit",
                "Independent legitimate context resolved a low-risk event without a model call.",
                event_id=event.event_id,
                evidence_id=legitimacy_packet.evidence_id,
            )
            return BlueTurn(
                event=event,
                investigation=investigation,
                evidence=evidence,
                decision=decision,
                risk_synthesis=risk_synthesis,
                ml_risk=ml_risk_info,
                model_calls=[],
                policy_adjustments=[
                    "Resolved on the deterministic verified-context lane; no Qwen call was required."
                ],
            )
        policy = {
            "actions": sorted(BLUE_ACTIONS),
            "risk_levels": sorted(RISK_LEVELS),
            "reason_codes": sorted(BLUE_REASON_CODES),
            "objective": "Prevent attack value while minimizing legitimate customer friction.",
            "risk_continuity": (
                "Do not downgrade an unresolved prior alert without positive legitimate context. "
                "Escalate unresolved step-up to hold or block before value moves. A hold pauses "
                "value but remains under evaluation until legitimate resolution or block."
            ),
            "fast_path_guard": (
                "The observable-only sequence guard is a minimum action, not a fraud label. "
                "You may choose a stronger action when evidence supports it, but never a weaker one."
            ),
            "ml_evidence": (
                "The ml_risk_score packet is a Blue-only statistical signal, not a fraud label or "
                "automatic verdict. Investigate an above-threshold score and reconcile it with "
                "timeline, entity, payment, and independently verified legitimate context."
            ),
            "legitimate_resolution": (
                "Use legitimate_context only when the legitimate-alternatives packet reports "
                "independent_verification_found=true. Repeated identifiers within this case are not history."
            ),
        }
        decision_payload = {
            **common,
            "available_evidence_tools": selected_tools,
            "tool_evidence": [item.to_dict() for item in evidence],
            "policy": policy,
        }
        combined_result, decision_call = self.gateway.generate_json(
            agent_name="blue_event_agent",
            model=self.config.blue_model_id,
            system_prompt=BLUE_EVENT_AGENT_SYSTEM_PROMPT,
            user_payload=decision_payload,
            schema_name="blue_event_response",
            schema=BLUE_EVENT_RESPONSE_SCHEMA,
            temperature=self.config.blue_temperature,
            seed=seed,
        )
        try:
            investigation = InvestigationRequest.from_dict(
                {
                    key: combined_result[key]
                    for key in ("preliminary_risk", "requested_tools", "investigation_focus")
                }
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Blue Event Agent returned an incomplete investigation.") from exc
        material_tools = [
            tool for tool in dict.fromkeys(investigation.requested_tools) if tool in selected_tools
        ]
        investigation = replace(
            investigation,
            requested_tools=material_tools or selected_tools,
        )

        trace(
            "blue.event.model_complete",
            "Blue investigated the evidence and returned one action candidate in a single call.",
            event_id=event.event_id,
            tools=investigation.requested_tools,
            model=decision_call.model,
            latency_ms=decision_call.latency_ms,
        )
        try:
            decision = BlueDecision.from_dict(
                {
                    key: combined_result[key]
                    for key in (
                        "event_id",
                        "action",
                        "risk_level",
                        "confidence",
                        "reason_codes",
                        "evidence_refs",
                        "decision_summary",
                        "mitigation",
                    )
                }
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Blue Event Agent returned an incomplete decision.") from exc
        decision, policy_adjustments = self._normalize_policy_labels(
            decision, prior_turns=prior_turns
        )
        decision, guard_adjustments = self._apply_fast_path_guard(
            decision,
            risk_synthesis=risk_synthesis,
            legitimate_context_verified=legitimate_context_verified,
            risk_evidence_id=risk_packet.evidence_id,
        )
        policy_adjustments.extend(guard_adjustments)
        decision, guard_label_adjustments = self._normalize_policy_labels(
            decision, prior_turns=prior_turns
        )
        policy_adjustments.extend(guard_label_adjustments)
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
        model_calls = [decision_call.__dict__]
        decision_errors = self._decision_errors(
            decision,
            event=event,
            prior_turns=prior_turns,
            available_evidence=available_evidence,
            risk_synthesis=risk_synthesis,
            legitimate_context_verified=legitimate_context_verified,
        )
        if decision_errors:
            trace(
                "blue.policy.repair_requested",
                "The deterministic policy gate requested one bounded repair.",
                event_id=event.event_id,
                violations=decision_errors,
            )
            repaired_result, repair_call = self.gateway.generate_json(
                agent_name="blue_event_repair",
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
            decision, repair_guard_adjustments = self._apply_fast_path_guard(
                decision,
                risk_synthesis=risk_synthesis,
                legitimate_context_verified=legitimate_context_verified,
                risk_evidence_id=risk_packet.evidence_id,
            )
            policy_adjustments.extend(repair_guard_adjustments)
            decision, repair_guard_label_adjustments = self._normalize_policy_labels(
                decision, prior_turns=prior_turns
            )
            policy_adjustments.extend(repair_guard_label_adjustments)
            remaining_errors = self._decision_errors(
                decision,
                event=event,
                prior_turns=prior_turns,
                available_evidence=available_evidence,
                risk_synthesis=risk_synthesis,
                legitimate_context_verified=legitimate_context_verified,
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
            risk_synthesis=risk_synthesis,
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
        playbook: DefensePlaybook | None = None,
    ) -> list[BlueTurn]:
        active_playbook = playbook or DefensePlaybook.baseline()
        turns: list[BlueTurn] = []
        visible: list[ObservedEvent] = []
        for index, event in enumerate(events):
            visible.append(event)
            turn = self.investigate_event(
                event=event,
                visible_history=list(visible),
                prior_turns=turns,
                playbook=active_playbook,
                seed=seed + index * 10,
            )
            turns.append(turn)
            if stop_on_decisive_action and turn.decision.action == "block":
                break
        return turns
