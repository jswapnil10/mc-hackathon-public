"""A deterministic Referee with exclusive access to sealed ground truth."""

from __future__ import annotations

from .contracts import EVIDENCE_TOOLS, BlueTurn, RefereeReport, SimulationCase


ALERT_ACTIONS = {"step_up", "hold", "block"}
DECISIVE_ACTIONS = {"hold", "block"}  # money-stopping actions: ambient hard-FP denominator
LIFECYCLE_PHASES = ("pre_transaction", "transaction", "post_transaction")
ACTION_FRICTION = {
    "allow": 0.0,
    "monitor": 0.03,
    "step_up": 0.25,
    "hold": 0.70,
    "block": 1.0,
}
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
    "cross_phase_pattern": "velocity",
}


def _effective_decisive_turn(turns: list[BlueTurn]) -> BlueTurn | None:
    """Return the first unresolved hold, or a terminal block, after all case events."""
    active_hold: BlueTurn | None = None
    for turn in turns:
        if turn.decision.action == "block":
            return turn
        if turn.decision.action == "hold" and active_hold is None:
            active_hold = turn
            continue
        if (
            active_hold is not None
            and "legitimate_context" in turn.decision.reason_codes
            and turn.decision.action in {"allow", "monitor", "step_up"}
        ):
            active_hold = None
    return active_hold


class DeterministicReferee:
    """Scores outcomes; an LLM is never allowed to alter these numbers."""

    @staticmethod
    def _truth_by_event(case: SimulationCase) -> dict[str, object]:
        return {record.event_id: record for record in case.truth}

    @staticmethod
    def _lifecycle_scorecard(
        *,
        attack_case: SimulationCase,
        attack_turns: list[BlueTurn],
        control_results: list[tuple[SimulationCase, list[BlueTurn]]],
        decisive_turn: BlueTurn | None,
        hard_fp_rate: float,
        friction_rate: float,
    ) -> tuple[dict[str, dict[str, object]], float, str | None, float, float, float]:
        truth_by_event = DeterministicReferee._truth_by_event(attack_case)
        decisive_truth = (
            truth_by_event.get(decisive_turn.event.event_id) if decisive_turn else None
        )
        phase_index = {phase: index for index, phase in enumerate(LIFECYCLE_PHASES)}
        event_phase_by_id = {
            event.event_id: event.lifecycle_phase for event in attack_case.events
        }
        planned_phases = [
            phase
            for phase in LIFECYCLE_PHASES
            if any(event.lifecycle_phase == phase for event in attack_case.events)
        ]
        reached_phases = [
            phase
            for phase in planned_phases
            if any(turn.event.lifecycle_phase == phase for turn in attack_turns)
        ]
        scorecard: dict[str, dict[str, object]] = {}
        scored: list[tuple[str, float]] = []
        stealth_values: list[float] = []

        for phase in LIFECYCLE_PHASES:
            phase_events = [
                event for event in attack_case.events if event.lifecycle_phase == phase
            ]
            phase_turns = [
                turn for turn in attack_turns if turn.event.lifecycle_phase == phase
            ]
            if not phase_events:
                scorecard[phase] = {
                    "status": "not_in_scenario",
                    "phase_score": None,
                    "event_count": 0,
                    "evaluated_event_count": 0,
                }
                continue
            if not phase_turns:
                scorecard[phase] = {
                    "status": "prevented_before_phase",
                    "phase_score": None,
                    "event_count": len(phase_events),
                    "evaluated_event_count": 0,
                    "evaluation_coverage": 0.0,
                    "first_actionable_event": phase_events[0].event_type,
                }
                continue

            phase_truth = [truth_by_event[event.event_id] for event in phase_events]
            phase_start = min(item.offset_seconds for item in phase_truth)
            phase_end = max(item.offset_seconds for item in phase_truth)
            later_truth = [
                record
                for record in attack_case.truth
                if record.offset_seconds > phase_end
                and phase_index[event_phase_by_id[record.event_id]] > phase_index[phase]
            ]
            deadline = min(
                (record.offset_seconds for record in later_truth),
                default=phase_end,
            )
            response_window = max(1, deadline - phase_start)
            phase_alert = next(
                (
                    turn for turn in phase_turns if turn.decision.action in ALERT_ACTIONS
                ),
                None,
            )
            detected = 1.0 if phase_alert else 0.0

            # A single Arena battle contains one attack case plus bounded legitimate
            # look-alikes. These diagnostic classification metrics make that small
            # denominator explicit; population claims remain the job of the sealed
            # Scenario Foundry benchmark.
            phase_control_turns = [
                [
                    turn
                    for turn in control_turns
                    if turn.event.lifecycle_phase == phase
                ]
                for _, control_turns in control_results
            ]
            phase_control_turns = [turns for turns in phase_control_turns if turns]
            true_positives = int(bool(phase_alert))
            false_negatives = 1 - true_positives
            false_positives = sum(
                any(turn.decision.action in ALERT_ACTIONS for turn in turns)
                for turns in phase_control_turns
            )
            true_negatives = len(phase_control_turns) - false_positives
            alerted_cases = true_positives + false_positives
            phase_precision = (
                true_positives / alerted_cases if alerted_cases else None
            )
            phase_recall = true_positives / (true_positives + false_negatives)
            phase_f1 = (
                2.0 * phase_precision * phase_recall / (phase_precision + phase_recall)
                if phase_precision is not None and phase_precision + phase_recall
                else 0.0
            )
            if phase_alert:
                alert_offset = truth_by_event[phase_alert.event.event_id].offset_seconds
                response_time = max(0, alert_offset - phase_start)
                response_score = (
                    1.0
                    if alert_offset <= phase_start
                    else max(0.0, 1.0 - response_time / response_window)
                )
            else:
                response_time = None
                response_score = 0.0

            downstream_value = sum(
                record.value_at_risk_inr
                for record in attack_case.truth
                if record.offset_seconds >= phase_start
            )
            controlled_value = 0.0
            decisive_in_time = bool(
                decisive_truth is not None and decisive_truth.offset_seconds <= deadline
            )
            if decisive_in_time:
                controlled_value = sum(
                    record.value_at_risk_inr
                    for record in attack_case.truth
                    if record.offset_seconds
                    >= max(phase_start, decisive_truth.offset_seconds)
                )
            if downstream_value:
                consequence_control = min(1.0, controlled_value / downstream_value)
            else:
                consequence_control = 1.0 if decisive_in_time else 0.0

            phase_control_costs: list[float] = []
            phase_control_event_count = 0
            for matching in phase_control_turns:
                phase_control_event_count += len(matching)
                phase_control_costs.append(
                    max(ACTION_FRICTION[turn.decision.action] for turn in matching)
                )
            if phase_control_costs:
                legitimate_safety = 1.0 - sum(phase_control_costs) / len(
                    phase_control_costs
                )
            else:
                legitimate_safety = 1.0 - min(
                    1.0, 0.7 * hard_fp_rate + 0.3 * friction_rate
                )

            phase_alert_index = next(
                (
                    index
                    for index, turn in enumerate(phase_turns)
                    if turn.decision.action in ALERT_ACTIONS
                ),
                None,
            )
            earlier_alert = any(
                turn.decision.action in ALERT_ACTIONS
                and phase_index[turn.event.lifecycle_phase] < phase_index[phase]
                for turn in attack_turns
            )
            if earlier_alert:
                red_stealth = 0.0
            elif phase_alert_index is None:
                red_stealth = 1.0
            else:
                red_stealth = phase_alert_index / max(1, len(phase_turns))
            stealth_values.append(red_stealth)

            current_planned_index = planned_phases.index(phase)
            next_phase = (
                planned_phases[current_planned_index + 1]
                if current_planned_index + 1 < len(planned_phases)
                else None
            )
            transition_escape = (
                any(turn.event.lifecycle_phase == next_phase for turn in attack_turns)
                if next_phase
                else None
            )
            packet_count = sum(len(turn.evidence) for turn in phase_turns)
            cited_count = sum(len(turn.decision.evidence_refs) for turn in phase_turns)
            citation_rate = min(1.0, cited_count / packet_count) if packet_count else 0.0
            action_counts = {
                action: sum(turn.decision.action == action for turn in phase_turns)
                for action in ("allow", "monitor", "step_up", "hold", "block")
            }
            phase_score = round(
                100.0
                * (
                    0.25 * detected
                    + 0.20 * response_score
                    + 0.35 * consequence_control
                    + 0.20 * legitimate_safety
                ),
                2,
            )
            scored.append((phase, phase_score))
            scorecard[phase] = {
                "status": "reached",
                "phase_score": phase_score,
                "event_count": len(phase_events),
                "evaluated_event_count": len(phase_turns),
                "evaluation_coverage": round(len(phase_turns) / len(phase_events), 4),
                "first_actionable_event": phase_events[0].event_type,
                "opportunity_detected": bool(detected),
                "classification_metrics": {
                    "scope": "single_battle",
                    "attack_opportunity_count": 1,
                    "legitimate_comparison_count": len(phase_control_turns),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "true_negatives": true_negatives,
                    "precision": (
                        round(phase_precision, 4)
                        if phase_precision is not None
                        else None
                    ),
                    "recall": round(phase_recall, 4),
                    "f1": round(phase_f1, 4),
                },
                "response_time_seconds": response_time,
                "response_score": round(response_score, 4),
                "downstream_value_exposed_inr": round(downstream_value, 2),
                "value_controlled_from_phase_inr": round(controlled_value, 2),
                "consequence_control_ratio": round(consequence_control, 4),
                "legitimate_safety_rate": round(legitimate_safety, 4),
                "phase_matched_control_event_count": phase_control_event_count,
                "evidence_citation_rate": round(citation_rate, 4),
                "red_stealth_ratio": round(red_stealth, 4),
                "transition_escape_rate": (
                    float(transition_escape) if transition_escape is not None else None
                ),
                "actions": action_counts,
            }

        scores = [score for _, score in scored]
        macro = sum(scores) / len(scores) if scores else 0.0
        worst_phase, worst_score = min(scored, key=lambda item: item[1]) if scored else (None, 0.0)
        balance_gap = max(scores) - min(scores) if scores else 0.0
        balanced_score = 0.6 * macro + 0.4 * worst_score
        phase_reach = len(reached_phases) / len(planned_phases) if planned_phases else 0.0
        phase_breadth = len(planned_phases) / len(LIFECYCLE_PHASES)
        stealth = sum(stealth_values) / len(stealth_values) if stealth_values else 0.0
        stage_depth = min(1.0, len(attack_case.events) / 6.0)
        red_capability = 100.0 * (
            0.35 * phase_reach
            + 0.30 * stealth
            + 0.20 * phase_breadth
            + 0.15 * stage_depth
        )
        return (
            scorecard,
            round(balanced_score, 2),
            worst_phase,
            round(worst_score, 2),
            round(balance_gap, 2),
            round(red_capability, 2),
        )

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
        decisive_turn = _effective_decisive_turn(attack_turns)
        detected_truth = truth_by_event.get(alert_turn.event.event_id) if alert_turn else None
        decisive_truth = truth_by_event.get(decisive_turn.event.event_id) if decisive_turn else None

        total_value = round(sum(record.value_at_risk_inr for record in attack_case.truth), 2)
        prevented_value = 0.0
        if decisive_truth is not None and decisive_truth.intervention_point in {
            "PREVENT",
            "DECIDE",
            "CONTAIN",
        }:
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
            _effective_decisive_turn(turns) is not None for _, turns in control_results
        )
        friction_cases = sum(
            any(turn.decision.action in ALERT_ACTIONS for turn in turns)
            for _, turns in control_results
        )
        hard_fp_rate = hard_false_positives / control_count if control_count else 0.0
        friction_rate = friction_cases / control_count if control_count else 0.0
        detection_rate = 1.0 if alert_turn else 0.0
        event_evaluation_ratio = (
            len(attack_turns) / len(attack_case.events) if attack_case.events else 0.0
        )
        tools_used = {
            packet.tool_name for turn in attack_turns for packet in turn.evidence
        }
        evidence_tool_coverage = len(tools_used) / len(EVIDENCE_TOOLS)

        (
            lifecycle_metrics,
            balanced_lifecycle_score,
            worst_lifecycle_phase,
            worst_phase_score,
            lifecycle_balance_gap,
            red_capability_score,
        ) = self._lifecycle_scorecard(
            attack_case=attack_case,
            attack_turns=attack_turns,
            control_results=control_results,
            decisive_turn=decisive_turn,
            hard_fp_rate=hard_fp_rate,
            friction_rate=friction_rate,
        )
        realized_impact = round(max(0.0, total_value - prevented_value), 2)
        realized_impact_ratio = (
            round(realized_impact / total_value, 4) if total_value else 0.0
        )

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
            detected_lifecycle_phase = None
        else:
            early_credit = {"PREVENT": 1.0, "DECIDE": 0.65, "CONTAIN": 0.25}.get(
                detected_truth.intervention_point, 0.0
            )
            time_to_detect = detected_truth.offset_seconds
            detected_stage_id = detected_truth.stage_id
            detected_lifecycle_phase = alert_turn.event.lifecycle_phase
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
            "Hold can pause value while evaluation continues; block terminates the simulated case.",
            "Hold/block protects current or downstream synthetic value; CONTAIN receives lower timing credit than prevention.",
            "Step-up verification counts as detection and customer friction, but not guaranteed value prevention.",
        ]
        return RefereeReport(
            outcome=outcome,
            detected_stage_id=detected_stage_id,
            detected_lifecycle_phase=detected_lifecycle_phase,
            time_to_detect_seconds=time_to_detect,
            total_value_at_risk_inr=total_value,
            value_prevented_inr=prevented_value,
            value_prevented_ratio=prevented_ratio,
            hard_false_positive_rate=round(hard_fp_rate, 4),
            legitimate_friction_rate=round(friction_rate, 4),
            attack_detection_rate=detection_rate,
            event_evaluation_ratio=round(event_evaluation_ratio, 4),
            evidence_tool_coverage=round(evidence_tool_coverage, 4),
            lifecycle_metrics=lifecycle_metrics,
            balanced_lifecycle_defense_score=balanced_lifecycle_score,
            worst_lifecycle_phase=worst_lifecycle_phase,
            worst_phase_score=worst_phase_score,
            lifecycle_balance_gap=lifecycle_balance_gap,
            red_capability_score=red_capability_score,
            realized_impact_inr=realized_impact,
            realized_impact_ratio=realized_impact_ratio,
            blue_score=blue_score,
            red_score=round(100.0 - blue_score, 2),
            coarse_reason_categories=reason_codes,
            ambient_false_positive_rate=round(ambient_fp_rate, 4),
            ambient_friction_rate=round(ambient_friction_rate, 4),
            no_defense_loss_inr=total_value,
            loss_avoided_inr=prevented_value,
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

    @staticmethod
    def feedback_for_blue(
        *,
        report: RefereeReport,
        attack_case: SimulationCase,
        attack_turns: list[BlueTurn],
        control_results: list[tuple[SimulationCase, list[BlueTurn]]],
    ) -> dict[str, object]:
        """Declassified post-episode packet used only for bounded defense improvement."""
        return {
            "outcome": report.outcome,
            "blue_score": report.blue_score,
            "detected_lifecycle_phase": report.detected_lifecycle_phase,
            "time_to_detect_seconds": report.time_to_detect_seconds,
            "value_prevented_ratio": report.value_prevented_ratio,
            "hard_false_positive_rate": report.hard_false_positive_rate,
            "legitimate_friction_rate": report.legitimate_friction_rate,
            "event_evaluation_ratio": report.event_evaluation_ratio,
            "evidence_tool_coverage": report.evidence_tool_coverage,
            "balanced_lifecycle_defense_score": report.balanced_lifecycle_defense_score,
            "worst_lifecycle_phase": report.worst_lifecycle_phase,
            "lifecycle_balance_gap": report.lifecycle_balance_gap,
            "realized_impact_ratio": report.realized_impact_ratio,
            "coarse_reason_categories": report.coarse_reason_categories,
            "evaluated_case_observations": {
                "event_types": [event.event_type for event in attack_case.events],
                "lifecycle_phases": [event.lifecycle_phase for event in attack_case.events],
                "actions": [turn.decision.action for turn in attack_turns],
                "risk_levels": [turn.decision.risk_level for turn in attack_turns],
                "tools_used": sorted(
                    {
                        packet.tool_name
                        for turn in attack_turns
                        for packet in turn.evidence
                    }
                ),
            },
            "legitimate_control_observations": [
                {
                    "event_types": [event.event_type for event in case.events],
                    "actions": [turn.decision.action for turn in turns],
                    "resolved_with_legitimate_context": any(
                        "legitimate_context" in turn.decision.reason_codes for turn in turns
                    ),
                }
                for case, turns in control_results
            ],
        }
