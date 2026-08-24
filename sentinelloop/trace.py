"""Opt-in, secret-free execution tracing for guided local debugging."""

from __future__ import annotations

import json
import os
from typing import Any


def trace_enabled() -> bool:
    return os.environ.get("SENTINELLOOP_TRACE", "0").lower() in {"1", "true", "yes", "on"}


def trace(event: str, message: str, **details: Any) -> None:
    if not trace_enabled():
        return
    payload = {"event": event, "message": message, "details": details}
    print(f"[SentinelLoop Trace] {json.dumps(payload, default=str, sort_keys=True)}", flush=True)
