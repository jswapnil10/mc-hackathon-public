"""A small OpenAI-compatible gateway for Qwen, vLLM, and similar servers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import AgentLabConfig


@dataclass(frozen=True)
class ModelCall:
    agent_name: str
    model: str
    schema_name: str
    latency_ms: int


class StructuredModelGateway(Protocol):
    def generate_json(
        self,
        *,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[dict[str, Any], ModelCall]: ...


def _extract_json_text(content: Any) -> str:
    if isinstance(content, list):
        text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
        content = "".join(text_parts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model response did not contain message content.")
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else cleaned


class OpenAICompatibleGateway:
    """Call a self-hosted endpoint without depending on a vendor SDK."""

    def __init__(self, config: AgentLabConfig) -> None:
        self.config = config

    def _response_format(self, schema_name: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        mode = self.config.structured_output_mode
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def generate_json(
        self,
        *,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
    ) -> tuple[dict[str, Any], ModelCall]:
        format_instruction = (
            "Return exactly one JSON object matching the supplied contract. Do not use Markdown fences."
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\n\n{format_instruction}"},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
            "temperature": temperature,
            "max_tokens": self.config.max_output_tokens,
            "seed": seed,
        }
        # Ollama enables thinking by default for Qwen 3-class models. For contract-bound agent
        # calls we need the output budget spent on the final JSON, not a private reasoning field.
        # `omit` remains available for older OpenAI-compatible servers that reject this field.
        if self.config.reasoning_effort != "omit":
            payload["reasoning_effort"] = self.config.reasoning_effort
        response_format = self._response_format(schema_name, schema)
        if response_format is not None:
            payload["response_format"] = response_format

        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.model_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model endpoint returned HTTP {exc.code} for {agent_name}: {detail[:1200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach the model endpoint at {self.config.chat_completions_url}: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"The model endpoint at {self.config.chat_completions_url} did not respond within "
                f"{self.config.request_timeout_seconds}s for {agent_name} ({exc}). On CPU-only hosts, "
                f"raise MODEL_TIMEOUT_SECONDS or use a smaller model."
            ) from exc

        try:
            message = body["choices"][0]["message"]
            raw_content = message.get("content")
            if not isinstance(raw_content, (str, list)) or not raw_content:
                finish_reason = body.get("choices", [{}])[0].get("finish_reason", "unknown")
                if message.get("reasoning"):
                    raise RuntimeError(
                        "Model returned reasoning but no final JSON content "
                        f"(finish_reason={finish_reason}). Set MODEL_REASONING_EFFORT=none and "
                        "restart the web service."
                    )
            content = _extract_json_text(raw_content)
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Model returned invalid structured JSON for {agent_name}.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Model output for {agent_name} must be a JSON object.")
        trace = ModelCall(
            agent_name=agent_name,
            model=model,
            schema_name=schema_name,
            latency_ms=round((time.monotonic() - started) * 1000),
        )
        return result, trace
