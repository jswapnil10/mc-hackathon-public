"""Environment-driven configuration for an open-weight model deployment."""

from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"
LOCAL_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def _detect_local_ollama_qwen() -> tuple[str, str] | None:
    """Find an already-installed local Qwen without downloading or changing anything."""
    request = urllib.request.Request(f"{LOCAL_OLLAMA_BASE_URL}/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=0.75) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    model_ids = [
        str(item.get("id", ""))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    preferences = ("qwen3.5:9b", "qwen3.5", "qwen3-coder:latest", "qwen3-coder")
    for preferred in preferences:
        match = next((model_id for model_id in model_ids if model_id == preferred), None)
        if match:
            return LOCAL_OLLAMA_BASE_URL, match
    fallback = next((model_id for model_id in model_ids if "qwen" in model_id.lower()), None)
    return (LOCAL_OLLAMA_BASE_URL, fallback) if fallback else None


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, received {raw!r}.") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw!r}.") from exc


@dataclass(frozen=True)
class AgentLabConfig:
    """All model choices are replaceable without changing application code."""

    model_base_url: str = "http://127.0.0.1:8000/v1"
    model_api_key: str = "local-development"
    red_model_id: str = DEFAULT_QWEN_MODEL
    blue_model_id: str = DEFAULT_QWEN_MODEL
    request_timeout_seconds: int = 120
    structured_output_mode: str = "json_schema"
    red_temperature: float = 0.65
    blue_temperature: float = 0.15
    max_output_tokens: int = 1400
    case_parallelism: int = 4
    ml_detector_enabled: bool = False           # off by default: Blue stays LLM-only unless enabled
    ml_model_dir: str = "data/loop/models/champion"

    @classmethod
    def from_env(cls) -> "AgentLabConfig":
        configured_base_url = os.environ.get("MODEL_BASE_URL")
        configured_red_model = os.environ.get("RED_MODEL_ID")
        configured_blue_model = os.environ.get("BLUE_MODEL_ID")
        detected_local = None
        if not configured_base_url and not configured_red_model and not configured_blue_model:
            detected_local = _detect_local_ollama_qwen()
        default_base_url, default_model = detected_local or (cls.model_base_url, DEFAULT_QWEN_MODEL)
        config = cls(
            model_base_url=(configured_base_url or default_base_url).rstrip("/"),
            model_api_key=os.environ.get("MODEL_API_KEY", cls.model_api_key),
            red_model_id=configured_red_model or default_model,
            blue_model_id=configured_blue_model or default_model,
            request_timeout_seconds=_int_env("MODEL_TIMEOUT_SECONDS", cls.request_timeout_seconds),
            structured_output_mode=os.environ.get(
                "MODEL_STRUCTURED_OUTPUT_MODE", cls.structured_output_mode
            ).lower(),
            red_temperature=_float_env("RED_AGENT_TEMPERATURE", cls.red_temperature),
            blue_temperature=_float_env("BLUE_AGENT_TEMPERATURE", cls.blue_temperature),
            max_output_tokens=_int_env("MODEL_MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            case_parallelism=_int_env("CASE_PARALLELISM", cls.case_parallelism),
            ml_detector_enabled=os.environ.get("ML_DETECTOR_ENABLED", "0").lower() in {"1", "true", "yes"},
            ml_model_dir=os.environ.get("ML_MODEL_DIR", cls.ml_model_dir),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_base_url.startswith(("http://", "https://")):
            raise ValueError("MODEL_BASE_URL must be an HTTP(S) URL.")
        if self.structured_output_mode not in {"json_schema", "json_object", "prompt"}:
            raise ValueError(
                "MODEL_STRUCTURED_OUTPUT_MODE must be json_schema, json_object, or prompt."
            )
        for name, value in (
            ("RED_AGENT_TEMPERATURE", self.red_temperature),
            ("BLUE_AGENT_TEMPERATURE", self.blue_temperature),
        ):
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} must be between 0 and 2.")
        if self.request_timeout_seconds < 1 or self.max_output_tokens < 100:
            raise ValueError("Model timeout and output-token limit must be positive.")
        if not 1 <= self.case_parallelism <= 8:
            raise ValueError("CASE_PARALLELISM must be between 1 and 8.")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.model_base_url}/chat/completions"
