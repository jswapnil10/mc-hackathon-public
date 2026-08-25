"""Cumulative session risk for the streaming Blue detector.

Session risk is a running max of the per-event probabilities, matching the repo's existing
risk-continuity policy ("do not downgrade an unresolved alert"). A chain that spikes once stays
elevated for the rest of the session.
"""

from __future__ import annotations

from typing import Any


def cumulative_session_risk(per_event_risk: float, prior_turns: list[Any]) -> float:
    """running_max(p_1..p_t): the max per-event risk seen so far in this case, incl. the current."""
    prior_max = 0.0
    for turn in prior_turns:
        info = getattr(turn, "ml_risk", None)
        if info and info.get("per_event_risk") is not None:
            prior_max = max(prior_max, float(info["per_event_risk"]))
    return max(float(per_event_risk), prior_max)
