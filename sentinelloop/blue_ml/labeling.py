"""Append-only training log for the retraining loop (Phase 3).

After each round the orchestrator logs ONE row per materialised event (attack chain + controls),
joined to its sealed TruthRecord — ALL stages, not just the ones Blue processed before an
early-stop, so the log is free of selective-labelling / feedback-loop bias. Rows use the same
flat schema as `dataset.py` so `build_feature_frame` consumes them unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import SimulationCase
from ..dataset import _row


def _flatten_case(case: SimulationCase, *, source: str) -> list[dict[str, Any]]:
    return [
        _row(ev, tr, group_key=case.case_id, chain_id=case.case_id, source=source)
        for ev, tr in zip(case.events, case.truth)
    ]


def append_rows(rows: list[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return len(rows)


def log_round(
    attack_case: SimulationCase,
    control_cases: list[SimulationCase],
    path: str | Path,
) -> int:
    """Log every materialised stage of the attack chain + its controls for one round."""
    rows = _flatten_case(attack_case, source="attack")
    for control in control_cases:
        rows.extend(_flatten_case(control, source="control"))
    return append_rows(rows, path)


def load_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)
