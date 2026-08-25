"""Post-episode Blue strategy proposals with deterministic bounded validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AgentLabConfig
from .contracts import (
    BLUE_REASON_CODES,
    BLUE_STRATEGY_SCHEMA,
    EVIDENCE_TOOLS,
    DefensePlaybook,
)
from .model_gateway import ModelCall, StructuredModelGateway
from .prompts import BLUE_STRATEGIST_SYSTEM_PROMPT
from .trace import trace


@dataclass(frozen=True)
class BlueStrategyTurn:
    proposed_playbook: DefensePlaybook
    model_call: ModelCall

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_playbook": self.proposed_playbook.to_dict(),
            "model_call": self.model_call.__dict__,
        }


class GenAIBlueStrategist:
    """Proposes bounded playbook changes; it never promotes its own proposal."""

    def __init__(self, *, gateway: StructuredModelGateway, config: AgentLabConfig) -> None:
        self.gateway = gateway
        self.config = config

    def propose(
        self,
        *,
        current_playbook: DefensePlaybook,
        post_episode_packet: dict[str, Any],
        seed: int,
    ) -> BlueStrategyTurn:
        trace(
            "blue.strategy.started",
            "Blue Strategist received a declassified post-episode learning packet.",
            current_playbook_version=current_playbook.version,
        )
        result, model_call = self.gateway.generate_json(
            agent_name="blue_strategist",
            model=self.config.blue_model_id,
            system_prompt=BLUE_STRATEGIST_SYSTEM_PROMPT,
            user_payload={
                "task": "Propose one bounded defense playbook for deterministic replay.",
                "current_playbook": current_playbook.to_dict(),
                "post_episode_packet": post_episode_packet,
                "approved_tools": sorted(EVIDENCE_TOOLS),
                "approved_reason_codes": sorted(BLUE_REASON_CODES),
                "promotion_policy": (
                    "The candidate is promoted only when replay preserves legitimate-case safety, "
                    "protected value, and lifecycle balance while improving prevention, realized impact, "
                    "the weakest lifecycle phase, or detection timing. More tool usage alone never qualifies."
                ),
                "output_contract": BLUE_STRATEGY_SCHEMA,
            },
            schema_name="blue_defense_strategy",
            schema=BLUE_STRATEGY_SCHEMA,
            temperature=0.2,
            seed=seed,
        )
        unknown_tools = set(result.get("preferred_tools", [])).difference(EVIDENCE_TOOLS)
        unknown_reasons = set(result.get("focus_reason_codes", [])).difference(BLUE_REASON_CODES)
        if unknown_tools or unknown_reasons:
            raise ValueError(
                "Blue Strategist proposed unknown capabilities: "
                f"tools={sorted(unknown_tools)}, reasons={sorted(unknown_reasons)}"
            )
        try:
            playbook = DefensePlaybook.from_proposal(
                result, version=current_playbook.version + 1
            )
        except TypeError as exc:
            raise ValueError("Blue Strategist returned an incomplete playbook.") from exc
        if not playbook.preferred_tools or not playbook.focus_reason_codes:
            raise ValueError("Blue Strategist must propose tools and defensive focus codes.")
        trace(
            "blue.strategy.proposed",
            "Blue Strategist proposed a bounded candidate for replay.",
            candidate_version=playbook.version,
            preferred_tools=playbook.preferred_tools,
            focus_reason_codes=playbook.focus_reason_codes,
            latency_ms=model_call.latency_ms,
        )
        return BlueStrategyTurn(proposed_playbook=playbook, model_call=model_call)
