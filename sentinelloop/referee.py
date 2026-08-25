"""A deterministic Referee with exclusive access to sealed ground truth."""

from __future__ import annotations

from .contracts import BlueTurn, RefereeReport, SimulationCase


ALERT_ACTIONS = {"step_up", "hold", "block"}
DECISIVE_ACTIONS = {"hold", "block"}
RED_FEEDBACK_REASONS = {
    "device_novelty",
    "network_novelty",
    "beneficiary_novelty",
    "amount_anomaly",
    "velocity",
    "graph_concentration",
    "identity_linkage",
    "communication_risk",
}
REASON_NORMALIZATION = {
    "profile_change": "beneficiary_novelty",
    "verification_inconsistency": "identity_linkage",
    "behavior_sequence": "velocity",
}


class DeterministicReferee:
    """Scores outcomes; an LLM is never allowed to alter these numbers."""

    @staticmethod
    def _truth_by_event(case: SimulationCase) -> dict[str, object]:
        return {record.event_id: record for record in case.truth}

    def score(
        self,
        *,
        attack_case: SimulationCase,
        attack_turns: list[BlueTurn],
        control_results: list[tuple[SimulationCase, list[BlueTurn]]],
        ambient_results: list[tuple[SimulationCase, list[BlueTurn]]] | None = None,
    ) -> RefereeReport:
        truth_by_event = self._truth_by_event(attack_case)
        alert_turn = next(
            (turn for turn in attack_turns if turn.decision.action in ALERT_ACTIONS), None
        )
        decisive_turn = next(
            (turn for turn in attack_turns if turn.decision.action in DECISIVE_ACTIONS), None
        )
        detected_truth = truth_by_event.get(alert_turn.event.event_id) if alert_turn else None
        decisive_truth = truth_by_event.get(decisive_turn.event.event_id) if decisive_turn else None

        total_value = round(sum(record.value_at_risk_inr for record in attack_case.truth), 2)
        prevented_value = 0.0
        if decisive_truth is not None and decisive_truth.intervention_point in {"PREVENT", "DECIDE"}:
            prevented_value = sum(
                record.value_at_risk_inr
                for record in attack_case.truth
                if record.offset_seconds >= decisive_truth.offset_seconds
            )
        prevented_value = round(min(total_value, prevented_value), 2)
        prevented_ratio = round(prevented_value / total_value, 4) if total_value else 0.0

        if alert_turn is None:
            outcome = "missed"
        elif decisive_truth is None:
            outcome = "detected"
        elif decisive_truth.intervention_point == "CONTAIN":
            outcome = "contained"
        elif prevented_ratio >= 0.999:
            outcome = "prevented"
        else:
            outcome = "contained"

        control_count = len(control_results)
        hard_false_positives = sum(
            any(turn.decision.action in DECISIVE_ACTIONS for turn in turns)
            for _, turns in control_results
        )
        friction_cases = sum(
            any(turn.decision.action in ALERT_ACTIONS for turn in turns)
            for _, turns in control_results
        )
        hard_fp_rate = hard_false_positives / control_count if control_count else 0.0
        friction_rate = friction_cases / control_count if control_count else 0.0
        detection_rate = 1.0 if alert_turn else 0.0

        # Ordinary standalone legit traffic + traps: the realistic FP denominator, tracked
        # separately from the hard look-alike controls (recorded, not folded into blue_score,
        # so the existing scoring formula is unchanged).
        ambient_results = ambient_results or []
        ambient_count = len(ambient_results)
        ambient_false_positives = sum(
            any(turn.decision.action in DECISIVE_ACTIONS for turn in turns)
            for _, turns in ambient_results
        )
        ambient_friction = sum(
            any(turn.decision.action in ALERT_ACTIONS for turn in turns)
            for _, turns in ambient_results
        )
        ambient_fp_rate = ambient_false_positives / ambient_count if ambient_count else 0.0
        ambient_friction_rate = ambient_friction / ambient_count if ambient_count else 0.0

        if detected_truth is None:
            early_credit = 0.0
            time_to_detect = None
            detected_stage_id = None
        else:
            early_credit = {"PREVENT": 1.0, "DECIDE": 0.65, "CONTAIN": 0.25}.get(
                detected_truth.intervention_point, 0.0
            )
            time_to_detect = detected_truth.offset_seconds
            detected_stage_id = detected_truth.stage_id
        score = 100.0 * (
            0.35 * detection_rate
            + 0.35 * prevented_ratio
            + 0.20 * early_credit
            + 0.10 * (1.0 - hard_fp_rate)
        )
        score -= 5.0 * friction_rate
        blue_score = round(max(0.0, min(100.0, score)), 2)

        reason_codes: list[str] = []
        if alert_turn:
            for reason in alert_turn.decision.reason_codes:
                normalized = REASON_NORMALIZATION.get(reason, reason)
                if normalized in RED_FEEDBACK_REASONS and normalized not in reason_codes:
                    reason_codes.append(normalized)
        notes = [
            "Detection, protected value, intervention timing, and legitimate-case safety are scored deterministically.",
            "Hold/block can protect current and downstream payment value only at PREVENT or DECIDE stages.",
            "Step-up verification counts as detection and customer friction, but not guaranteed value prevention.",
        ]
        return RefereeReport(
            outcome=outcome,
            detected_stage_id=detected_stage_id,
            time_to_detect_seconds=time_to_detect,
            total_value_at_risk_inr=total_value,
            value_prevented_inr=prevented_value,
            value_prevented_ratio=prevented_ratio,
            hard_false_positive_rate=round(hard_fp_rate, 4),
            legitimate_friction_rate=round(friction_rate, 4),
            attack_detection_rate=detection_rate,
            blue_score=blue_score,
            red_score=round(100.0 - blue_score, 2),
            coarse_reason_categories=reason_codes,
            ambient_false_positive_rate=round(ambient_fp_rate, 4),
            ambient_friction_rate=round(ambient_friction_rate, 4),
            scoring_notes=notes,
        )

    @staticmethod
    def feedback_for_red(report: RefereeReport) -> dict[str, object]:
        """This is the entire declassified channel from Referee to Red."""
        return {
            "outcome": report.outcome,
            "detected_stage_id": report.detected_stage_id,
            "time_to_detect_seconds": report.time_to_detect_seconds,
            "value_prevented_ratio": report.value_prevented_ratio,
            "false_positive_rate": report.hard_false_positive_rate,
            "coarse_reason_categories": report.coarse_reason_categories,
        }
