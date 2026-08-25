"""Turn simulated event rows into the leakage-controlled model matrix.

The bridge between `dataset.py` (flattened per-event rows) and `features.build_features`
(the ONE causal, point-in-time feature function used identically at train and inference).
For each event we reconstruct its ObservedEvent shape and the strictly-prior events in the
same chain, then call `build_features` — so the training matrix is produced by exactly the
arithmetic that will run at inference time. No model here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    audit_leakage,
    build_features,
)

# Columns that are structural/bookkeeping or ground-truth — never part of an event's attributes
# and never a model feature.
STRUCTURAL = {
    "event_id", "occurred_at", "event_type", "sequence", "chain_id", "group_key",
    "attack_family", "is_attack", "fraud_contributing", "value_at_risk_inr", "label", "source",
}
META_COLUMNS = [
    "chain_id", "attack_family", "sequence", "value_at_risk_inr",
    "source", "is_attack", "fraud_contributing", "event_type", "event_id",
]


def _event_dict(row: pd.Series) -> dict[str, Any]:
    """Reconstruct an ObservedEvent.to_dict() shape from one flattened row."""
    attributes = {
        key: value
        for key, value in row.items()
        if key not in STRUCTURAL and not pd.isna(value)
    }
    return {
        "event_id": row.get("event_id"),
        "sequence": row.get("sequence"),
        "occurred_at": row.get("occurred_at"),
        "event_type": row.get("event_type"),
        "observable_signals": [],
        "attributes": attributes,
    }


def build_feature_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """(X[FEATURES], y, groups, meta) built causally per chain via `build_features`."""
    leaked = audit_leakage()
    assert not leaked, f"Feature allowlist leaks forbidden columns: {leaked}"

    df = df.reset_index(drop=True)
    events = [_event_dict(row) for _, row in df.iterrows()]

    # Group row indices by chain; ambient/trap rows (chain_id NaN) are singletons with no history.
    groups_idx: dict[Any, list[int]] = defaultdict(list)
    for i, chain_id in enumerate(df["chain_id"]):
        key = chain_id if not pd.isna(chain_id) else ("__standalone__", i)
        groups_idx[key].append(i)

    feats_rows: list[dict[str, Any] | None] = [None] * len(df)
    for idxs in groups_idx.values():
        ordered = sorted(idxs, key=lambda i: (df.at[i, "sequence"] if not pd.isna(df.at[i, "sequence"]) else 0))
        history: list[dict[str, Any]] = []
        for i in ordered:
            feats_rows[i] = build_features(events[i], list(history))
            history.append(events[i])

    X = pd.DataFrame(feats_rows, columns=FEATURES)
    for col in NUMERIC_FEATURES:
        # bools -> 1/0, strings -> numbers, missing -> NaN (kept as learnable signal).
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    assert list(X.columns) == FEATURES, "feature matrix columns must equal the FEATURES allowlist"

    y = df["label"].astype(int)
    groups = df["group_key"]
    meta = df[[c for c in META_COLUMNS if c in df.columns]].copy()
    return X, y, groups, meta


def build_feature_frame_from_seed(seed: int = 42, **build_kwargs: Any):
    """Convenience: build the dataset in-memory then the causal feature frame."""
    from ..dataset import build_dataset

    df = build_dataset(seed, **build_kwargs)
    return build_feature_frame(df)
