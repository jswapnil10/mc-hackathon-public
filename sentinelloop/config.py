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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, received {raw!r}.")


@dataclass(frozen=True)
class AgentLabConfig:
    """All model choices are replaceable without changing application code."""

    model_base_url: str = "http://127.0.0.1:8000/v1"
    model_api_key: str = "local-development"
    red_model_id: str = DEFAULT_QWEN_MODEL
    blue_model_id: str = DEFAULT_QWEN_MODEL
    request_timeout_seconds: int = 120
    structured_output_mode: str = "json_schema"
    # `auto` selects role-specific effort. A concrete value remains as an emergency global
    # override, while `omit` supports endpoints with no reasoning-control parameter.
    reasoning_effort: str = "auto"
    red_reasoning_effort: str = "medium"
    blue_reasoning_effort: str = "low"
    blue_strategy_reasoning_effort: str = "medium"
    red_temperature: float = 0.65
    blue_temperature: float = 0.15
    max_output_tokens: int = 1400
    case_parallelism: int = 4
    # This detector is exclusively a Blue data-plane capability. Red receives only coarse
    # Referee feedback and never sees model scores, features, thresholds, or weights.
    ml_detector_enabled: bool = True
    ml_model_dir: str = "data/loop/models/champion"
    training_log_path: str = "data/loop/training_log.jsonl"
    retrain_every: int = 0
    include_ambient_evaluation: bool = False
    ambient_sample: int = 8
    trap_sample_each: int = 2

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
            reasoning_effort=os.environ.get(
                "MODEL_REASONING_EFFORT", cls.reasoning_effort
            ).lower(),
            red_reasoning_effort=os.environ.get(
                "RED_REASONING_EFFORT", cls.red_reasoning_effort
            ).lower(),
            blue_reasoning_effort=os.environ.get(
                "BLUE_REASONING_EFFORT", cls.blue_reasoning_effort
            ).lower(),
            blue_strategy_reasoning_effort=os.environ.get(
                "BLUE_STRATEGY_REASONING_EFFORT", cls.blue_strategy_reasoning_effort
            ).lower(),
            red_temperature=_float_env("RED_AGENT_TEMPERATURE", cls.red_temperature),
            blue_temperature=_float_env("BLUE_AGENT_TEMPERATURE", cls.blue_temperature),
            max_output_tokens=_int_env("MODEL_MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            case_parallelism=_int_env("CASE_PARALLELISM", cls.case_parallelism),
            ml_detector_enabled=_bool_env("ML_DETECTOR_ENABLED", cls.ml_detector_enabled),
            ml_model_dir=os.environ.get("ML_MODEL_DIR", cls.ml_model_dir),
            training_log_path=os.environ.get("BATTLE_TRAINING_LOG", cls.training_log_path),
            retrain_every=_int_env("BATTLE_RETRAIN_EVERY", cls.retrain_every),
            include_ambient_evaluation=_bool_env(
                "BATTLE_INCLUDE_AMBIENT", cls.include_ambient_evaluation
            ),
            ambient_sample=_int_env("BATTLE_AMBIENT_SAMPLE", cls.ambient_sample),
            trap_sample_each=_int_env("BATTLE_TRAP_SAMPLE_EACH", cls.trap_sample_each),
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
        if self.reasoning_effort not in {"auto", "none", "low", "medium", "high", "omit"}:
            raise ValueError(
                "MODEL_REASONING_EFFORT must be auto, none, low, medium, high, or omit."
            )
        for name, value in (
            ("RED_REASONING_EFFORT", self.red_reasoning_effort),
            ("BLUE_REASONING_EFFORT", self.blue_reasoning_effort),
            ("BLUE_STRATEGY_REASONING_EFFORT", self.blue_strategy_reasoning_effort),
        ):
            if value not in {"none", "low", "medium", "high", "omit"}:
                raise ValueError(f"{name} must be none, low, medium, high, or omit.")
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
        if self.retrain_every < 0:
            raise ValueError("BATTLE_RETRAIN_EVERY cannot be negative.")
        if self.ambient_sample < 0 or self.trap_sample_each < 0:
            raise ValueError("Ambient and trap sample sizes cannot be negative.")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.model_base_url}/chat/completions"

    def reasoning_effort_for(self, agent_name: str) -> str:
        """Choose test-time reasoning by role unless a global override is configured."""
        if self.reasoning_effort != "auto":
            return self.reasoning_effort
        if agent_name == "red_planner" or agent_name.startswith("red_"):
            return self.red_reasoning_effort
        if agent_name == "blue_strategist":
            return self.blue_strategy_reasoning_effort
        if agent_name.startswith("blue_"):
            return self.blue_reasoning_effort
        return self.blue_reasoning_effort
