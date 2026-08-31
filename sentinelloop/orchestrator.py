"""Closed-loop episode orchestration for Red GenAI versus Blue GenAI."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from red_team_agent.models import ScenarioSpec

from .blue_agent import GenAIBlueAgent
from .blue_strategy import BlueStrategyTurn, GenAIBlueStrategist
from .config import AgentLabConfig
from .contracts import BlueTurn, DefensePlaybook, RefereeReport, SimulationCase
from .evaluation import catalog_submission_profile, round_submission_evaluation
from .model_gateway import OpenAICompatibleGateway, StructuredModelGateway
from .red_agent import GenAIRedAgent, RedTurn
from .referee import DeterministicReferee
from .simulation import (
    simulate_ambient_cases,
    simulate_attack,
    simulate_legitimate_controls,
    simulate_trap_cases,
)
from .trace import trace


@dataclass(frozen=True)
class BlueAdaptationResult:
    strategy_turn: BlueStrategyTurn
    replay_attack_turns: list[BlueTurn]
    replay_control_results: list[tuple[SimulationCase, list[BlueTurn]]]
    replay_report: RefereeReport
    promoted: bool
    promotion_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_turn.to_dict(),
            "replay": {
                "attack_turns": [turn.to_dict() for turn in self.replay_attack_turns],
                "control_summaries": [
                    {
                        "case_id": case.case_id,
                        "control_name_revealed_after_scoring": case.control_name,
                        "decisions": [turn.decision.to_dict() for turn in turns],
                    }
                    for case, turns in self.replay_control_results
                ],
                "referee": self.replay_report.to_dict(),
            },
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
        }


def _clone_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Snapshot the grouped event stream for a progress payload (events copied so later
    action-fills don't mutate an already-published snapshot)."""
    return [{**g, "events": [dict(e) for e in g.get("events", [])]} for g in groups]


def _case_event_rows(case: Any, turns: list[Any]) -> list[dict[str, Any]]:
    """Per-event rows (sequence, type, attributes, Blue action) for any case — attack or benign."""
    action_by_id = {}
    for turn in turns:
        event = getattr(turn, "event", None)
        if event is not None:
            action_by_id[getattr(event, "event_id", None)] = getattr(turn, "decision", None)
    rows = []
    for event in case.events:
        decision = action_by_id.get(getattr(event, "event_id", None))
        rows.append(
            {
                "sequence": getattr(event, "sequence", len(rows) + 1),
                "event_type": getattr(event, "event_type", "EVENT"),
                "lifecycle_phase": getattr(event, "lifecycle_phase", ""),
                "attributes": dict(getattr(event, "attributes", {}) or {}),
                "action": getattr(decision, "action", None) if decision else None,
                "risk_level": getattr(decision, "risk_level", None) if decision else None,
            }
        )
    return rows


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    red_turn: RedTurn
    attack_case: SimulationCase
    attack_blue_turns: list[BlueTurn]
    control_results: list[tuple[SimulationCase, list[BlueTurn]]]
    referee_report: RefereeReport
    feedback_released_to_red: dict[str, object]
    active_blue_playbook: DefensePlaybook
    blue_adaptation: BlueAdaptationResult | None = None
    submission_evaluation: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    ambient_results: list[tuple[SimulationCase, list[BlueTurn]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "red": self.red_turn.to_dict(),
            "simulation": {
                "attack_case": {
                    "case_id": self.attack_case.case_id,
                    "events": [event.to_dict() for event in self.attack_case.events],
                    "sealed_truth_revealed_after_scoring": [
                        item.to_dict() for item in self.attack_case.truth
                    ],
                },
                "legitimate_control_count": len(self.control_results),
                "ambient_case_count": len(self.ambient_results),
            },
            "blue": {
                "active_playbook": self.active_blue_playbook.to_dict(),
                "attack_turns": [turn.to_dict() for turn in self.attack_blue_turns],
                "control_summaries": [
                    {
                        "case_id": case.case_id,
                        "control_name_revealed_after_scoring": case.control_name,
                        "decisions": [turn.decision.to_dict() for turn in turns],
                    }
                    for case, turns in self.control_results
                ],
                "ambient_summaries": [
                    {
                        "case_id": case.case_id,
                        "kind_revealed_after_scoring": case.case_id.split("-")[0],
                        "decisions": [turn.decision.to_dict() for turn in turns],
                    }
                    for case, turns in self.ambient_results
                ],
            },
            # Grouped event view: the real attack, the benign look-alikes, and ordinary traffic —
            # each with per-event attributes + Blue's action, distinctly labelled for the UI.
            "event_groups": [
                {
                    "kind": "attack",
                    "label": self.red_turn.scenario.attack_family or "Attack",
                    "events": _case_event_rows(self.attack_case, self.attack_blue_turns),
                },
                *[
                    {
                        "kind": "lookalike",
                        "label": case.control_name or "Legitimate look-alike",
                        "events": _case_event_rows(case, turns),
                    }
                    for case, turns in self.control_results
                ],
                *[
                    {
                        "kind": "ambient",
                        "label": case.case_id.split("-")[0].replace("_", " ") or "Ordinary traffic",
                        "events": _case_event_rows(case, turns),
                    }
                    for case, turns in self.ambient_results
                ],
            ],
            "referee": self.referee_report.to_dict(),
            "feedback_released_to_red": self.feedback_released_to_red,
            "blue_adaptation": self.blue_adaptation.to_dict() if self.blue_adaptation else None,
            "submission_evaluation": self.submission_evaluation or {},
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class LabRun:
    run_id: str
    model_configuration: dict[str, Any]
    rounds: list[RoundResult]
    final_defense_playbook: DefensePlaybook
    submission_profile: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "architecture": "red_genai_vs_two_speed_learning_blue_with_deterministic_referee",
            "model_configuration": self.model_configuration,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "final_defense_playbook": self.final_defense_playbook.to_dict(),
            "submission_profile": self.submission_profile or {},
            "duration_ms": self.duration_ms,
        }


class SentinelLoopOrchestrator:
    def __init__(
        self,
        *,
        config: AgentLabConfig | None = None,
        gateway: StructuredModelGateway | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or AgentLabConfig.from_env()
        self.gateway = gateway or OpenAICompatibleGateway(self.config)
        self.red = GenAIRedAgent(gateway=self.gateway, config=self.config)
        self.blue = GenAIBlueAgent(gateway=self.gateway, config=self.config)
        self.blue_strategist = GenAIBlueStrategist(gateway=self.gateway, config=self.config)
        self.referee = DeterministicReferee()
        self.progress_callback = progress_callback

    def _progress(
        self,
        stage: str,
        headline: str,
        detail: str,
        *,
        round_number: int,
        total_rounds: int,
        **facts: Any,
    ) -> None:
        """Publish sanitized, judge-facing progress without exposing agent prompts or truth."""
        if self.progress_callback is None:
            return
        payload = {
            "status": "completed" if stage == "completed" else "running",
            "stage": stage,
            "headline": headline,
            "detail": detail,
            "round_number": round_number,
            "total_rounds": total_rounds,
            **facts,
        }
        try:
            self.progress_callback(payload)
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a battle
            trace(
                "progress.publish_failed",
                "Judge-facing progress reporting failed; the battle continued.",
                stage=stage,
                error=str(exc),
            )

    def _run_blue_cases(
        self,
        *,
        attack_case: SimulationCase,
        controls: list[SimulationCase],
        attack_seed: int,
        control_seed_base: int,
        playbook: DefensePlaybook,
        round_number: int,
        total_rounds: int,
        replay: bool = False,
        event_groups: list[dict[str, Any]] | None = None,
    ) -> tuple[list[BlueTurn], list[tuple[SimulationCase, list[BlueTurn]]]]:
        """Evaluate isolated cases concurrently while preserving order within each case."""
        jobs = [
            (attack_case, attack_seed),
            *[
                (control, control_seed_base + index * 100)
                for index, control in enumerate(controls)
            ],
        ]

        total_event_capacity = sum(len(case.events) for case, _ in jobs)
        completed_events = 0
        # Grouped stream pre-populated by Red (all events + attributes, action=None). Groups align
        # with jobs order: [attack, *look-alikes]. Blue fills in each event's action as it scores.
        groups = event_groups if event_groups is not None else []
        seq_index_by_case = {}
        for job_index, (case, _seed) in enumerate(jobs):
            if job_index < len(groups):
                seq_index_by_case[id(case)] = {
                    entry.get("sequence"): entry for entry in groups[job_index].get("events", [])
                }
        progress_lock = Lock()
        stage = "blue_replay" if replay else "blue_investigation"
        headline = (
            "Blue is replay-testing a candidate defense"
            if replay
            else "Blue is evaluating the attack and legitimate look-alikes"
        )
        self._progress(
            stage,
            headline,
            (
                f"0 event decisions completed across {len(jobs)} isolated cases. "
                "Events remain ordered inside each case."
            ),
            round_number=round_number,
            total_rounds=total_rounds,
            completed_events=0,
            total_event_capacity=total_event_capacity,
            case_count=len(jobs),
        )

        def evaluate(job: tuple[SimulationCase, int]) -> list[BlueTurn]:
            nonlocal completed_events
            case, case_seed = job
            case_seq_index = seq_index_by_case.get(id(case))

            def event_completed(_event: Any, _turn: Any) -> None:
                nonlocal completed_events
                with progress_lock:
                    completed_events += 1
                    current = completed_events
                    if case_seq_index is not None:
                        decision = getattr(_turn, "decision", None)
                        entry = case_seq_index.get(getattr(_event, "sequence", None))
                        if entry is not None:
                            entry["action"] = getattr(decision, "action", None)
                            entry["risk_level"] = getattr(decision, "risk_level", None)
                    groups_snapshot = _clone_groups(groups)
                self._progress(
                    stage,
                    headline,
                    (
                        f"{current} event decision{'s' if current != 1 else ''} completed "
                        f"across {len(jobs)} isolated cases."
                    ),
                    round_number=round_number,
                    total_rounds=total_rounds,
                    completed_events=current,
                    total_event_capacity=total_event_capacity,
                    case_count=len(jobs),
                    event_groups=groups_snapshot,
                )

            return self.blue.run_case(
                case.events,
                seed=case_seed,
                stop_on_decisive_action=True,
                playbook=playbook,
                event_completed=event_completed,
            )

        worker_count = min(self.config.case_parallelism, len(jobs))
        trace(
            "blue.case_batch.started",
            "Blue started isolated attack and look-alike cases with bounded concurrency.",
            case_count=len(jobs),
            worker_count=worker_count,
        )
        if worker_count == 1:
            outcomes = [evaluate(job) for job in jobs]
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count, thread_name_prefix="sentinelloop-blue"
            ) as executor:
                outcomes = list(executor.map(evaluate, jobs))
        attack_turns = outcomes[0]
        control_results = list(zip(controls, outcomes[1:]))
        trace(
            "blue.case_batch.completed",
            "Blue completed the isolated case batch.",
            attack_event_count=len(attack_turns),
            control_event_counts=[len(turns) for _, turns in control_results],
        )
        return attack_turns, control_results

    @staticmethod
    def _promotion_decision(
        baseline: RefereeReport, candidate: RefereeReport
    ) -> tuple[bool, str]:
        safety_preserved = (
            candidate.hard_false_positive_rate <= baseline.hard_false_positive_rate
            and candidate.legitimate_friction_rate <= baseline.legitimate_friction_rate
            and candidate.value_prevented_ratio >= baseline.value_prevented_ratio
            and candidate.blue_score >= baseline.blue_score
            and candidate.balanced_lifecycle_defense_score
            >= baseline.balanced_lifecycle_defense_score
            and candidate.realized_impact_ratio <= baseline.realized_impact_ratio
        )
        earlier_detection = (
            candidate.time_to_detect_seconds is not None
            and (
                baseline.time_to_detect_seconds is None
                or candidate.time_to_detect_seconds < baseline.time_to_detect_seconds
            )
        )
        measurable_gain = (
            candidate.blue_score > baseline.blue_score
            or candidate.value_prevented_ratio > baseline.value_prevented_ratio
            or candidate.realized_impact_ratio < baseline.realized_impact_ratio
            or candidate.balanced_lifecycle_defense_score
            > baseline.balanced_lifecycle_defense_score
            or candidate.worst_phase_score > baseline.worst_phase_score
            or earlier_detection
        )
        if safety_preserved and measurable_gain:
            return True, (
                "Promoted: deterministic replay preserved legitimate-case safety and improved "
                "protected value, lifecycle resilience, realized impact, or detection timing."
            )
        if not safety_preserved:
            return False, (
                "Rejected: replay reduced score, protected value, lifecycle resilience, or legitimate-case safety."
            )
        return False, "Rejected: replay produced no measurable defensive gain."

    def run(
        self,
        *,
        attack_family: str | None = None,
        difficulty: str = "medium",
        rounds: int = 2,
        seed: int = 20260824,
        include_legitimate_controls: bool = True,
        include_ambient: bool = False,  # opt-in: keeps default runs aligned with MasterGuard's tests
        ambient_sample: int = 20,
        trap_sample_each: int = 5,
        training_log_path: str | None = None,
        retrain_every: int | None = None,
    ) -> LabRun:
        lab_started = time.monotonic()
        trace(
            "lab.started",
            "SentinelLoop accepted a new adversarial lab run.",
            attack_family=attack_family,
            difficulty=difficulty,
            rounds=rounds,
            seed=seed,
            red_model=self.config.red_model_id,
            blue_model=self.config.blue_model_id,
        )
        # Cap lifted to 200 so the generational retraining loop (Phase 3) can run many rounds.
        if rounds < 1 or rounds > 200:
            raise ValueError("A lab run must contain between 1 and 200 rounds.")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("Difficulty must be easy, medium, or hard.")
        previous: ScenarioSpec | None = None
        feedback: dict[str, object] | None = None
        results: list[RoundResult] = []
        bounded_family = attack_family
        active_playbook = DefensePlaybook.baseline()
        self._progress(
            "preparing",
            "Preparing the synthetic payment arena",
            "The server validated the battle settings and isolated Red, Blue, and Referee data.",
            round_number=1,
            total_rounds=rounds,
        )
        for index in range(rounds):
            round_started = time.monotonic()
            round_seed = seed + index
            round_playbook = active_playbook
            trace(
                "round.started",
                "A feedback-loop round started.",
                round_number=index + 1,
                round_seed=round_seed,
            )
            self._progress(
                "red_planning",
                "Red is designing a bounded attack plan",
                (
                    f"Round {index + 1} of {rounds}. The Red model is choosing a safe objective, "
                    "lifecycle focus, and synthetic parameter changes."
                ),
                round_number=index + 1,
                total_rounds=rounds,
            )
            red_turn = self.red.plan(
                attack_family=bounded_family,
                difficulty=difficulty,
                seed=round_seed,
                previous_scenario=previous,
                referee_feedback=feedback,
            )
            if bounded_family is None:
                bounded_family = red_turn.scenario.attack_family
            attack_case = simulate_attack(red_turn.scenario)
            trace(
                "simulation.attack_compiled",
                "The deterministic simulator materialized Red's campaign.",
                case_id=attack_case.case_id,
                observable_event_count=len(attack_case.events),
                sealed_truth_record_count=len(attack_case.truth),
                event_types=[event.event_type for event in attack_case.events],
            )
            controls = (
                simulate_legitimate_controls(
                    red_turn.scenario, red_turn.scenario.legitimate_controls
                )
                if include_legitimate_controls
                else []
            )
            # Stream events the moment Red generates them (with ALL attributes) — the real attack
            # and the benign look-alikes, grouped and labelled. Blue's action per event is filled in
            # as it scores; ambient traffic is appended after. Shown live and persisted in the report.
            event_groups: list[dict[str, Any]] = [
                {
                    "kind": "attack",
                    "label": red_turn.scenario.attack_family or "Attack",
                    "events": _case_event_rows(attack_case, []),
                }
            ]
            for control in controls:
                event_groups.append(
                    {
                        "kind": "lookalike",
                        "label": getattr(control, "control_name", None) or "Legitimate look-alike",
                        "events": _case_event_rows(control, []),
                    }
                )
            self._progress(
                "simulation",
                "The arena built the synthetic payment events",
                (
                    f"Safety checks passed. Built {len(attack_case.events)} attack events and "
                    f"{len(controls)} legitimate look-alike cases; sealed labels remain with the Referee."
                ),
                round_number=index + 1,
                total_rounds=rounds,
                attack_event_count=len(attack_case.events),
                control_case_count=len(controls),
                event_groups=_clone_groups(event_groups),
            )
            attack_turns, control_results = self._run_blue_cases(
                attack_case=attack_case,
                controls=controls,
                attack_seed=round_seed * 100,
                control_seed_base=round_seed * 1000,
                playbook=round_playbook,
                round_number=index + 1,
                total_rounds=rounds,
                event_groups=event_groups,
            )
            trace(
                "blue.attack_case.completed",
                "Blue completed or decisively stopped the attack case.",
                processed_event_count=len(attack_turns),
                actions=[turn.decision.action for turn in attack_turns],
            )
            # Controls were already scored by _run_blue_cases above; add the ambient/trap pass here
            # (realistic false-positive denominator) so the referee can report ambient friction.
            ambient_results: list[tuple[SimulationCase, list[BlueTurn]]] = []
            if include_ambient:
                ambient_cases = simulate_ambient_cases(seed=round_seed, count=ambient_sample)
                ambient_cases += simulate_trap_cases(seed=round_seed, count_each=trap_sample_each)
                for ambient_index, ambient_case in enumerate(ambient_cases):
                    turns = self.blue.run_case(
                        ambient_case.events,
                        seed=round_seed * 10000 + ambient_index * 10,
                        stop_on_decisive_action=True,
                    )
                    ambient_results.append((ambient_case, turns))
                    event_groups.append(
                        {
                            "kind": "ambient",
                            "label": ambient_case.case_id.split("-")[0].replace("_", " ")
                            or "Ordinary traffic",
                            "events": _case_event_rows(ambient_case, turns),
                        }
                    )
                trace(
                    "ambient.completed",
                    "Blue processed standalone legit + trap traffic (realistic FP denominator).",
                    ambient_case_count=len(ambient_results),
                )
            self._progress(
                "referee_scoring",
                "The Referee is opening the sealed answer key",
                "Blue's decisions are complete. The Referee is measuring detection, value protected, timing, and legitimate-customer harm.",
                round_number=index + 1,
                total_rounds=rounds,
                event_groups=_clone_groups(event_groups),
            )
            report = self.referee.score(
                attack_case=attack_case,
                attack_turns=attack_turns,
                control_results=control_results,
                ambient_results=ambient_results,
            )
            feedback = self.referee.feedback_for_red(report)
            trace(
                "referee.scored",
                "The deterministic Referee opened sealed truth and calculated the result.",
                outcome=report.outcome,
                blue_score=report.blue_score,
                red_score=report.red_score,
                value_prevented_ratio=report.value_prevented_ratio,
                hard_false_positive_rate=report.hard_false_positive_rate,
            )
            trace(
                "referee.feedback_released",
                "Only the declassified feedback packet was released to Red.",
                feedback=feedback,
            )
            if training_log_path:
                # Log every materialised stage (attack chain + controls) for the retraining loop.
                from .blue_ml.labeling import log_round

                logged = log_round(attack_case, [case for case, _ in control_results], training_log_path)
                trace("labeling.round_logged", "Appended round rows to the training log.",
                      rows=logged, path=training_log_path)
            blue_adaptation: BlueAdaptationResult | None = None
            if index < rounds - 1:
                self._progress(
                    "blue_adaptation",
                    "Blue is proposing a safer defense for the next round",
                    "Only post-outcome evidence is available. The candidate must beat the current playbook without increasing customer harm.",
                    round_number=index + 1,
                    total_rounds=rounds,
                )
                post_episode_packet = self.referee.feedback_for_blue(
                    report=report,
                    attack_case=attack_case,
                    attack_turns=attack_turns,
                    control_results=control_results,
                )
                strategy_turn = self.blue_strategist.propose(
                    current_playbook=round_playbook,
                    post_episode_packet=post_episode_packet,
                    seed=round_seed * 10_000 + 700,
                )
                candidate = strategy_turn.proposed_playbook
                replay_attack_turns, replay_control_results = self._run_blue_cases(
                    attack_case=attack_case,
                    controls=[control_case for control_case, _ in control_results],
                    attack_seed=round_seed * 10_000 + 1_000,
                    control_seed_base=round_seed * 10_000 + 2_000,
                    playbook=candidate,
                    round_number=index + 1,
                    total_rounds=rounds,
                    replay=True,
                )
                replay_report = self.referee.score(
                    attack_case=attack_case,
                    attack_turns=replay_attack_turns,
                    control_results=replay_control_results,
                )
                promoted, promotion_reason = self._promotion_decision(report, replay_report)
                trace(
                    "blue.strategy.replay_scored",
                    "The Referee scored Blue's candidate on the same attack and controls.",
                    current_playbook_version=round_playbook.version,
                    candidate_playbook_version=candidate.version,
                    baseline_score=report.blue_score,
                    replay_score=replay_report.blue_score,
                    promoted=promoted,
                    promotion_reason=promotion_reason,
                )
                blue_adaptation = BlueAdaptationResult(
                    strategy_turn=strategy_turn,
                    replay_attack_turns=replay_attack_turns,
                    replay_control_results=replay_control_results,
                    replay_report=replay_report,
                    promoted=promoted,
                    promotion_reason=promotion_reason,
                )
                if promoted:
                    active_playbook = candidate
            round_duration_ms = round((time.monotonic() - round_started) * 1000)
            results.append(
                RoundResult(
                    round_number=index + 1,
                    red_turn=red_turn,
                    attack_case=attack_case,
                    attack_blue_turns=attack_turns,
                    control_results=control_results,
                    referee_report=report,
                    feedback_released_to_red=feedback,
                    active_blue_playbook=round_playbook,
                    blue_adaptation=blue_adaptation,
                    submission_evaluation=round_submission_evaluation(
                        attack_case=attack_case,
                        attack_turns=attack_turns,
                        control_results=control_results,
                        report=report,
                        round_duration_ms=round_duration_ms,
                        case_parallelism=self.config.case_parallelism,
                    ),
                    duration_ms=round_duration_ms,
                    ambient_results=ambient_results,
                )
            )
            previous = red_turn.scenario
            self._progress(
                "round_complete",
                f"Round {index + 1} is complete",
                (
                    f"Outcome: {report.outcome}. "
                    + (
                        "The next round will use only the permitted feedback."
                        if index < rounds - 1
                        else "The final battle report is being prepared."
                    )
                ),
                round_number=index + 1,
                total_rounds=rounds,
                outcome=report.outcome,
            )

            # End of a generation: retrain the challenger and hot-reload if it beats the champion.
            if retrain_every and training_log_path and (index + 1) % retrain_every == 0:
                from .blue_ml.retrain import retrain as _retrain

                generation = (index + 1) // retrain_every
                trace("retrain.started", "Generation boundary reached; retraining challenger.",
                      generation=generation, after_round=index + 1)
                try:
                    decision = _retrain(
                        training_log_path,
                        champion_dir=self.config.ml_model_dir,
                        generation=generation,
                    )
                    if decision.get("promoted") and self.blue.reload_detector():
                        trace("retrain.promoted", "Challenger promoted; Blue hot-reloaded the new champion.",
                              generation=generation, challenger=decision["challenger"], model_hash=self.blue.model_hash)
                    else:
                        trace("retrain.rejected", "Challenger did not beat the incumbent; champion unchanged.",
                              generation=generation, challenger=decision["challenger"], incumbent=decision["incumbent"])
                except Exception as exc:  # noqa: BLE001 - retraining must never crash a live run
                    trace("retrain.failed", "Retraining raised; continuing with the current champion.",
                          generation=generation, error=str(exc))
        lab_duration_ms = round((time.monotonic() - lab_started) * 1000)
        result = LabRun(
            run_id=f"LAB-{seed}-{bounded_family or 'AUTO'}",
            model_configuration={
                "red": self.config.red_model_id,
                "blue": self.config.blue_model_id,
                "blue_detector": {
                    "type": "HistGradientBoostingClassifier",
                    "configured": self.config.ml_detector_enabled,
                    "active": self.blue.detector is not None,
                    "model_hash": self.blue.model_hash,
                    "role": "Blue evidence only; never exposed to Red",
                },
                "blue_strategist": self.config.blue_model_id,
                "referee": "deterministic-policy-v2",
                "blue_execution": "single_call_per_event",
                "case_parallelism": self.config.case_parallelism,
            },
            rounds=results,
            final_defense_playbook=active_playbook,
            submission_profile=catalog_submission_profile(self.red.catalog),
            duration_ms=lab_duration_ms,
        )
        trace(
            "lab.completed",
            "The adversarial lab run completed.",
            run_id=result.run_id,
            completed_rounds=len(result.rounds),
            duration_ms=result.duration_ms,
        )
        self._progress(
            "completed",
            "Battle complete",
            "The scored report is ready. No real payment rail, customer, or external recipient was touched.",
            round_number=rounds,
            total_rounds=rounds,
            run_id=result.run_id,
            duration_ms=result.duration_ms,
        )
        return result
