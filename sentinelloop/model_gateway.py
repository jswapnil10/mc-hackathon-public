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
    reasoning_effort: str | None = None
    compatibility_fallbacks: tuple[str, ...] = ()


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


class _ReasoningOnlyResponse(RuntimeError):
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        super().__init__(finish_reason)


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

    @staticmethod
    def _reasoning_parameter_unsupported(detail: str) -> bool:
        normalized = detail.lower()
        mentions_reasoning = "reasoning_effort" in normalized or '"reasoning"' in normalized
        rejected = any(
            marker in normalized
            for marker in (
                "unknown",
                "unsupported",
                "not supported",
                "does not support",
                "unrecognized",
                "not permitted",
                "extra_forbidden",
                "unexpected",
                "invalid field",
            )
        )
        return mentions_reasoning and rejected

    def _post(
        self,
        payload: dict[str, Any],
        *,
        agent_name: str,
        fallbacks: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.model_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                return (
                    json.loads(response.read().decode("utf-8")),
                    payload.get("reasoning_effort"),
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if (
                exc.code in {400, 422}
                and "reasoning_effort" in payload
                and self._reasoning_parameter_unsupported(detail)
            ):
                compatible_payload = dict(payload)
                compatible_payload.pop("reasoning_effort", None)
                fallbacks.append("reasoning_parameter_omitted")
                return self._post(
                    compatible_payload,
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                )
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

    @staticmethod
    def _parse_result(body: dict[str, Any], agent_name: str) -> dict[str, Any]:
        try:
            choice = body["choices"][0]
            message = choice["message"]
            raw_content = message.get("content")
            if not isinstance(raw_content, (str, list)) or not raw_content:
                if message.get("reasoning"):
                    raise _ReasoningOnlyResponse(str(choice.get("finish_reason", "unknown")))
            content = _extract_json_text(raw_content)
            result = json.loads(content)
        except _ReasoningOnlyResponse:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Model returned invalid structured JSON for {agent_name}.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Model output for {agent_name} must be a JSON object.")
        return result

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
        configured_effort = self.config.reasoning_effort_for(agent_name)
        # OpenAI-compatible endpoints that reject this optional field are retried automatically
        # without it, preserving compatibility with ordinary non-reasoning models.
        if configured_effort != "omit":
            payload["reasoning_effort"] = configured_effort
        response_format = self._response_format(schema_name, schema)
        if response_format is not None:
            payload["response_format"] = response_format

        started = time.monotonic()
        fallbacks: list[str] = []
        body, used_effort = self._post(payload, agent_name=agent_name, fallbacks=fallbacks)
        try:
            result = self._parse_result(body, agent_name)
        except _ReasoningOnlyResponse as exc:
            if used_effort in {None, "none"}:
                raise RuntimeError(
                    "Model returned reasoning but no final JSON content even after compatibility "
                    f"handling (finish_reason={exc.finish_reason}). Increase MODEL_MAX_OUTPUT_TOKENS "
                    "or use a model with reliable structured output."
                ) from exc
            fallback_payload = dict(payload)
            fallback_payload["reasoning_effort"] = "none"
            fallbacks.append("reasoning_exhausted_retry_without_thinking")
            body, used_effort = self._post(
                fallback_payload,
                agent_name=agent_name,
                fallbacks=fallbacks,
            )
            try:
                result = self._parse_result(body, agent_name)
            except _ReasoningOnlyResponse as retry_exc:
                raise RuntimeError(
                    "Model returned reasoning but no final JSON content after the automatic "
                    f"no-reasoning retry (finish_reason={retry_exc.finish_reason})."
                ) from retry_exc
        trace = ModelCall(
            agent_name=agent_name,
            model=model,
            schema_name=schema_name,
            latency_ms=round((time.monotonic() - started) * 1000),
            reasoning_effort=used_effort or "omitted",
            compatibility_fallbacks=tuple(fallbacks),
        )
        return result, trace
