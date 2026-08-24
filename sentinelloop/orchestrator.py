"""Closed-loop episode orchestration for Red GenAI versus Blue GenAI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from red_team_agent.models import ScenarioSpec

from .blue_agent import GenAIBlueAgent
from .config import AgentLabConfig
from .contracts import BlueTurn, RefereeReport, SimulationCase
from .model_gateway import OpenAICompatibleGateway, StructuredModelGateway
from .red_agent import GenAIRedAgent, RedTurn
from .referee import DeterministicReferee
from .simulation import simulate_attack, simulate_legitimate_controls
from .trace import trace


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    red_turn: RedTurn
    attack_case: SimulationCase
    attack_blue_turns: list[BlueTurn]
    control_results: list[tuple[SimulationCase, list[BlueTurn]]]
    referee_report: RefereeReport
    feedback_released_to_red: dict[str, object]

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
            },
            "blue": {
                "attack_turns": [turn.to_dict() for turn in self.attack_blue_turns],
                "control_summaries": [
                    {
                        "case_id": case.case_id,
                        "control_name_revealed_after_scoring": case.control_name,
                        "decisions": [turn.decision.to_dict() for turn in turns],
                    }
                    for case, turns in self.control_results
                ],
            },
            "referee": self.referee_report.to_dict(),
            "feedback_released_to_red": self.feedback_released_to_red,
        }


@dataclass(frozen=True)
class LabRun:
    run_id: str
    model_configuration: dict[str, str]
    rounds: list[RoundResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "architecture": "red_genai_vs_blue_genai_with_deterministic_referee",
            "model_configuration": self.model_configuration,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
        }


class SentinelLoopOrchestrator:
    def __init__(
        self,
        *,
        config: AgentLabConfig | None = None,
        gateway: StructuredModelGateway | None = None,
    ) -> None:
        self.config = config or AgentLabConfig.from_env()
        self.gateway = gateway or OpenAICompatibleGateway(self.config)
        self.red = GenAIRedAgent(gateway=self.gateway, config=self.config)
        self.blue = GenAIBlueAgent(gateway=self.gateway, config=self.config)
        self.referee = DeterministicReferee()

    def run(
        self,
        *,
        attack_family: str | None = None,
        difficulty: str = "medium",
        rounds: int = 2,
        seed: int = 20260824,
        include_legitimate_controls: bool = True,
    ) -> LabRun:
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
        if rounds < 1 or rounds > 5:
            raise ValueError("A lab run must contain between 1 and 5 rounds.")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("Difficulty must be easy, medium, or hard.")
        previous: ScenarioSpec | None = None
        feedback: dict[str, object] | None = None
        results: list[RoundResult] = []
        bounded_family = attack_family
        for index in range(rounds):
            round_seed = seed + index
            trace(
                "round.started",
                "A feedback-loop round started.",
                round_number=index + 1,
                round_seed=round_seed,
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
            attack_turns = self.blue.run_case(
                attack_case.events,
                seed=round_seed * 100,
                stop_on_decisive_action=True,
            )
            trace(
                "blue.attack_case.completed",
                "Blue completed or decisively stopped the attack case.",
                processed_event_count=len(attack_turns),
                actions=[turn.decision.action for turn in attack_turns],
            )
            control_results: list[tuple[SimulationCase, list[BlueTurn]]] = []
            if include_legitimate_controls:
                controls = simulate_legitimate_controls(
                    red_turn.scenario, red_turn.scenario.legitimate_controls
                )
                for control_index, control_case in enumerate(controls):
                    trace(
                        "control.started",
                        "Blue received a fresh-memory legitimate look-alike case.",
                        control_number=control_index + 1,
                        event_count=len(control_case.events),
                    )
                    turns = self.blue.run_case(
                        control_case.events,
                        seed=round_seed * 1000 + control_index * 100,
                        stop_on_decisive_action=True,
                    )
                    control_results.append((control_case, turns))
                    trace(
                        "control.completed",
                        "Blue finished the legitimate look-alike without seeing its label.",
                        control_number=control_index + 1,
                        actions=[turn.decision.action for turn in turns],
                    )
            report = self.referee.score(
                attack_case=attack_case,
                attack_turns=attack_turns,
                control_results=control_results,
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
            results.append(
                RoundResult(
                    round_number=index + 1,
                    red_turn=red_turn,
                    attack_case=attack_case,
                    attack_blue_turns=attack_turns,
                    control_results=control_results,
                    referee_report=report,
                    feedback_released_to_red=feedback,
                )
            )
            previous = red_turn.scenario
        result = LabRun(
            run_id=f"LAB-{seed}-{bounded_family or 'AUTO'}",
            model_configuration={
                "red": self.config.red_model_id,
                "blue": self.config.blue_model_id,
                "referee": "deterministic-policy-v1",
            },
            rounds=results,
        )
        trace(
            "lab.completed",
            "The adversarial lab run completed.",
            run_id=result.run_id,
            completed_rounds=len(result.rounds),
        )
        return result