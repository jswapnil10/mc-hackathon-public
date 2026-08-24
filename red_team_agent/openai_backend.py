"""Optional OpenAI Responses API backend for bounded planning decisions.

The model chooses only an attack card, difficulty, objective, and stage emphasis.
It never writes payment events. The deterministic compiler and safety gate remain
the authority for executable scenario data.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .catalog import AttackCatalog
from .models import PLANNER_DECISION_JSON_SCHEMA, PlannerDecision


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_API_URL = "https://api.openai.com/v1/responses"


def _extract_output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    raise RuntimeError("OpenAI response did not contain structured output text.")


class OpenAIPlanningBackend:
    name = "openai-responses"

    def __init__(self, catalog: AttackCatalog, model: str | None = None) -> None:
        self.catalog = catalog
        self.model = model or os.environ.get("RED_AGENT_MODEL", DEFAULT_MODEL)
        self.api_url = os.environ.get("OPENAI_RESPONSES_URL", DEFAULT_API_URL)

    def choose(
        self,
        *,
        attack_family: str | None,
        difficulty: str,
        objective: str | None,
        seed: int,
    ) -> PlannerDecision:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Configure it in your local environment; do not paste it into chat."
            )

        cards = []
        for card in self.catalog.list():
            cards.append(
                {
                    "attack_family": card.attack_family,
                    "name": card.name,
                    "observed_pattern": card.observed_pattern,
                    "genai_role": card.genai_role,
                    "payment_surface": card.payment_surface,
                    "available_stages": [stage["stage_id"] for stage in card.stage_templates],
                }
            )

        schema = json.loads(json.dumps(PLANNER_DECISION_JSON_SCHEMA))
        schema["properties"]["attack_family"]["enum"] = self.catalog.families
        prompt = {
            "requested_attack_family": attack_family,
            "requested_difficulty": difficulty,
            "requested_objective": objective,
            "seed": seed,
            "attack_cards": cards,
        }
        payload = {
            "model": self.model,
            "instructions": (
                "You are a defensive Red Team planner for a synthetic payment-security lab. "
                "Choose only from the supplied attack cards. Work at the level of observable "
                "payment, identity, session, and graph behavior. Never produce phishing text, "
                "credentials, personal data, real targets, URLs, exploit code, or instructions "
                "for wrongdoing. The deterministic compiler will create all events. Return only "
                "the requested structured planning decision."
            ),
            "input": json.dumps(prompt, indent=2),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "red_team_planner_decision",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Unable to reach the OpenAI Responses API: {exc.reason}") from exc

        decision = json.loads(_extract_output_text(body))
        if attack_family and decision["attack_family"] != attack_family:
            raise RuntimeError("Planner selected a different attack family than the requested bounded family.")
        if difficulty and decision["difficulty"] != difficulty:
            decision["difficulty"] = difficulty
        return PlannerDecision(**decision, backend=f"{self.name}:{self.model}")
