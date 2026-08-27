"""A small OpenAI-compatible gateway for Qwen, vLLM, and similar servers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

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


class _GatewayTimeout(TimeoutError):
    def __init__(self, timeout_seconds: float, detail: object) -> None:
        self.timeout_seconds = timeout_seconds
        self.detail = detail
        super().__init__(str(detail))


class OpenAICompatibleGateway:
    """Call a self-hosted endpoint without depending on a vendor SDK."""

    def __init__(
        self,
        config: AgentLabConfig,
        *,
        api_key_provider: Callable[[], str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.api_key_provider = api_key_provider
        self.extra_body = extra_body

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

    def _format_instruction(self) -> str:
        if self.config.prompt_profile == "claude":
            return (
                "Analyze the supplied evidence carefully, but return only the final JSON object; "
                "do not expose private reasoning. The user payload includes an output_contract. "
                "Follow that contract literally: include every required key at the top level, "
                "use exact enum values, preserve supplied event, evidence, tool, family, and "
                "difficulty identifiers verbatim, and respect all array and string bounds. "
                "Never add wrapper keys, Markdown fences, comments, nulls for required values, "
                "or properties the contract does not define. Silently verify contract completeness "
                "before returning exactly one JSON object."
            )
        return (
            "Return exactly one JSON object matching the output_contract in the user payload. "
            "Include every required top-level key and do not use Markdown fences."
        )

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
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        effective_timeout = timeout_seconds or self.config.request_timeout_seconds
        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": (
                    f"Bearer {self.api_key_provider()}"
                    if self.api_key_provider
                    else f"Bearer {self.config.model_api_key}"
                ),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=effective_timeout
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
                    timeout_seconds=effective_timeout,
                )
            raise RuntimeError(
                f"Model endpoint returned HTTP {exc.code} for {agent_name}: {detail[:1200]}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                raise _GatewayTimeout(effective_timeout, reason) from exc
            raise RuntimeError(
                f"Could not reach the model endpoint at {self.config.chat_completions_url}: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise _GatewayTimeout(effective_timeout, exc) from exc

    def _timeout_error(
        self,
        *,
        agent_name: str,
        fallbacks: list[str],
        detail: object,
    ) -> RuntimeError:
        fallback_note = (
            " Optional reasoning was stopped early and direct structured generation was retried."
            if "reasoning_timeout_retry_without_thinking" in fallbacks
            else " Direct structured generation was already used (reasoning effort: none)."
        )
        return RuntimeError(
            f"The model endpoint at {self.config.chat_completions_url} did not complete "
            f"{agent_name} within the {self.config.request_timeout_seconds}s request window "
            f"({detail}).{fallback_note} Increase MODEL_TIMEOUT_SECONDS or use a smaller model."
        )

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
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Some compatible providers prepend a brief acknowledgement despite the JSON-only
                # instruction. Recover one complete object; truncated objects still fail and retry.
                object_start = content.find("{")
                if object_start < 0:
                    raise
                result, _ = json.JSONDecoder().raw_decode(content[object_start:])
        except _ReasoningOnlyResponse:
            raise
        except RuntimeError as exc:
            if "did not contain message content" in str(exc):
                raise RuntimeError(
                    f"Model returned invalid structured JSON for {agent_name}: empty content."
                ) from exc
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
        format_instruction = self._format_instruction()
        contract_payload = dict(user_payload)
        # Include the schema in-band even when response_format is also sent. Some enterprise
        # OpenAI-compatible proxies accept response_format but do not pass strict enforcement
        # through to every provider model.
        contract_payload.setdefault("output_contract", schema)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\n\n{format_instruction}"},
                {
                    "role": "user",
                    "content": json.dumps(contract_payload, separators=(",", ":")),
                },
            ],
            "max_tokens": self.config.max_output_tokens,
            "seed": seed,
        }
        # Current Claude endpoints choose their own sampling profile and reject a nonzero
        # temperature as deprecated. Other OpenAI-compatible backends retain the existing knob.
        if self.config.prompt_profile != "claude":
            payload["temperature"] = temperature
        configured_effort = self.config.reasoning_effort_for(agent_name)
        # OpenAI-compatible endpoints that reject this optional field are retried automatically
        # without it, preserving compatibility with ordinary non-reasoning models.
        if configured_effort != "omit":
            payload["reasoning_effort"] = configured_effort
        response_format = self._response_format(schema_name, schema)
        if response_format is not None:
            payload["response_format"] = response_format
        if self.extra_body is not None:
            payload["extra_body"] = self.extra_body

        started = time.monotonic()
        fallbacks: list[str] = []
        reasoning_enabled = configured_effort not in {"none", "omit"}
        first_timeout = (
            self.config.request_timeout_seconds
            if reasoning_enabled and self.config.prompt_profile == "claude"
            else min(
                    self.config.reasoning_attempt_timeout_seconds,
                    max(1, self.config.request_timeout_seconds - 1),
                )
            if reasoning_enabled
            else self.config.request_timeout_seconds
        )
        try:
            body, used_effort = self._post(
                payload,
                agent_name=agent_name,
                fallbacks=fallbacks,
                timeout_seconds=first_timeout,
            )
        except _GatewayTimeout as exc:
            elapsed = time.monotonic() - started
            remaining = self.config.request_timeout_seconds - elapsed
            if not reasoning_enabled or remaining < 1:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail=exc.detail,
                ) from exc
            fallback_payload = dict(payload)
            fallback_payload["reasoning_effort"] = "none"
            fallbacks.append("reasoning_timeout_retry_without_thinking")
            try:
                body, used_effort = self._post(
                    fallback_payload,
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    timeout_seconds=max(1, remaining),
                )
            except _GatewayTimeout as retry_exc:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail=retry_exc.detail,
                ) from retry_exc
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
            if self.config.prompt_profile == "claude":
                fallback_payload["reasoning_effort"] = configured_effort
                fallbacks.append("reasoning_exhausted_retry_same_effort")
            else:
                fallback_payload["reasoning_effort"] = "none"
                fallbacks.append("reasoning_exhausted_retry_without_thinking")
            remaining = self.config.request_timeout_seconds - (time.monotonic() - started)
            if remaining < 1:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail="no request time remained after the reasoning-only response",
                ) from exc
            try:
                body, used_effort = self._post(
                    fallback_payload,
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    timeout_seconds=max(1, remaining),
                )
            except _GatewayTimeout as retry_exc:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail=retry_exc.detail,
                ) from retry_exc
            try:
                result = self._parse_result(body, agent_name)
            except _ReasoningOnlyResponse as retry_exc:
                raise RuntimeError(
                    "Model returned reasoning but no final JSON content after the automatic "
                    f"structured retry (finish_reason={retry_exc.finish_reason})."
                ) from retry_exc
        except RuntimeError as exc:
            if f"invalid structured JSON for {agent_name}" not in str(exc):
                raise
            repair_payload = dict(payload)
            repair_messages = [dict(message) for message in payload["messages"]]
            repair_messages[0]["content"] = (
                f"{repair_messages[0]['content']}\n\nYour previous response was not valid JSON. "
                "Retry the same task and return one complete, parseable object matching the "
                "output_contract. Do not truncate, preface, or wrap it."
            )
            repair_payload["messages"] = repair_messages
            fallbacks.append("invalid_json_retry")
            remaining = self.config.request_timeout_seconds - (time.monotonic() - started)
            if remaining < 1:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail="no request time remained for the invalid-JSON retry",
                ) from exc
            try:
                body, used_effort = self._post(
                    repair_payload,
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    timeout_seconds=max(1, remaining),
                )
            except _GatewayTimeout as retry_exc:
                raise self._timeout_error(
                    agent_name=agent_name,
                    fallbacks=fallbacks,
                    detail=retry_exc.detail,
                ) from retry_exc
            try:
                result = self._parse_result(body, agent_name)
            except _ReasoningOnlyResponse as retry_exc:
                raise RuntimeError(
                    "Model returned reasoning but no final JSON content during the automatic "
                    f"invalid-JSON retry (finish_reason={retry_exc.finish_reason})."
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
