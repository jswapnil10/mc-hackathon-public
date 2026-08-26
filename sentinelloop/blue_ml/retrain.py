"""Generational batch retraining with a champion/challenger gate (Phase 3).

A *generation* = N rounds during which the champion is frozen. Retraining is full-from-scratch
(cheap at this scale; warm-starting a HistGBM only adds trees on the same data and ossifies dead
attack modes). Each retrain blends: the synthetic legit+fraud baseline, the accumulated battle log,
and a class/family-stratified replay buffer (so families Red abandons then revisits are retained).
A challenger is promoted only when it makes a measurable gain while keeping chain recall and
hard-negative false positives inside strict non-regression tolerances.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..dataset import build_dataset
from .detector import DEFAULT_MODEL_DIR, FraudDetector
from .feature_frame import build_feature_frame
from .labeling import load_log
from .metrics import summarise, threshold_for_budget

REPLAY_PATH = Path("data/loop/replay_buffer.jsonl")
ARCHIVE_DIR = Path("data/loop/models/archive")
_REPLAY_CAP_PER_FAMILY = 500


# --------------------------------------------------------------------------- replay buffer
def update_replay_buffer(df: pd.DataFrame, path: Path = REPLAY_PATH, cap: int = _REPLAY_CAP_PER_FAMILY) -> pd.DataFrame:
    """Class/family-stratified reservoir of positives; keeps up to `cap` per attack_family."""
    positives = df[df["label"] == 1]
    existing = load_log(path)
    merged = pd.concat([existing, positives], ignore_index=True) if len(existing) else positives
    if "event_id" in merged:
        merged = merged.drop_duplicates(subset="event_id")
    kept = merged.groupby("attack_family", group_keys=False).head(cap) if len(merged) else merged
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for _, row in kept.iterrows():
            fh.write(json.dumps(row.dropna().to_dict(), default=str) + "\n")
    return kept


# --------------------------------------------------------------------------- evaluation + gate
def _evaluate(df: pd.DataFrame, seed: int, alert_rate: float, test_share: float) -> dict[str, Any]:
    """Evaluate a challenger with the same grouped k-fold protocol used for the champion.

    A single held-out fold is not comparable with the incumbent's cross-validation mean and can
    promote or reject a model because of split luck. Averaging every grouped fold keeps complete
    attack chains together and makes the champion/challenger gate apples-to-apples.
    """
    X, y, groups, meta = build_feature_frame(df)
    strat = meta["attack_family"].fillna("none")
    n_splits = max(2, round(1 / test_share))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_reports: list[dict[str, Any]] = []
    for train_idx, test_idx in sgkf.split(X, strat, groups):
        detector = FraudDetector().fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = detector.score(X.iloc[test_idx])
        threshold = threshold_for_budget(scores, max(1, round(alert_rate * len(test_idx))))
        fold_reports.append(summarise(meta.iloc[test_idx], y.iloc[test_idx], scores, threshold))
    keys = (
        "chain_recall",
        "hard_false_positive_rate",
        "chain_precision",
        "prevented_share",
        "row_recall",
        "row_precision",
    )
    return {
        key: round(float(np.mean([report[key] for report in fold_reports])), 4)
        for key in keys
    }


def _gate_metrics(meta: dict[str, Any]) -> dict[str, float] | None:
    """Read judge-facing gate metrics from a champion bundle (flat or CV-nested)."""
    if not meta:
        return None
    if "gate_metrics" in meta:
        return meta["gate_metrics"]
    cv = meta.get("cv")
    if cv and "chain_recall" in cv:
        result = {
            "chain_recall": cv["chain_recall"]["mean"],
            "hard_false_positive_rate": cv["hard_false_positive_rate"]["mean"],
        }
        for key in ("chain_precision", "prevented_share"):
            if key in cv:
                result[key] = cv[key]["mean"]
        return result
    return None


def _incumbent_metrics(champion_dir: Path) -> dict[str, float] | None:
    try:
        bundle = FraudDetector.load(champion_dir)
    except Exception:  # noqa: BLE001 - no/incompatible incumbent
        return None
    import joblib

    raw = joblib.load(Path(champion_dir) / "model.joblib")
    return _gate_metrics(raw.get("meta", {}))


def _should_promote(
    challenger: dict[str, Any],
    incumbent: dict[str, float] | None,
    min_delta: float,
    fp_tol: float = 0.002,
    min_improvement: float = 0.001,
) -> bool:
    """Require a measurable gain plus strict recall and customer-safety non-regression.

    Tolerances absorb small evaluation noise; they are not themselves an improvement. At least one
    judge-facing metric must improve by `min_improvement`, so a slightly worse challenger cannot be
    promoted merely because its regressions happen to fit inside the tolerances.
    """
    if incumbent is None:
        return True  # no champion yet -> accept
    recall_ok = challenger["chain_recall"] >= incumbent["chain_recall"] - min_delta
    fp_ok = challenger["hard_false_positive_rate"] <= incumbent["hard_false_positive_rate"] + fp_tol
    gains = [
        challenger["chain_recall"] - incumbent["chain_recall"],
        incumbent["hard_false_positive_rate"] - challenger["hard_false_positive_rate"],
    ]
    for key in ("chain_precision", "prevented_share"):
        if key in challenger and key in incumbent:
            gains.append(challenger[key] - incumbent[key])
    measurable_gain = max(gains) >= min_improvement
    return recall_ok and fp_ok and measurable_gain


# --------------------------------------------------------------------------- retrain driver
def retrain(
    log_path: str | Path,
    *,
    champion_dir: str | Path = DEFAULT_MODEL_DIR,
    baseline_seed: int = 42,
    baseline_seeds_per_cell: int = 8,
    alert_rate: float = 0.01,
    test_share: float = 0.3,
    min_delta: float = 0.01,
    fp_tol: float = 0.002,
    min_improvement: float = 0.001,
    generation: int | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """One generation's retrain: assemble data -> CV-evaluate challenger -> gate vs incumbent ->
    promote (refit on all + archive) or reject. Returns the decision record."""
    champion_dir = Path(champion_dir)
    frames = [build_dataset(baseline_seed, seeds_per_cell=baseline_seeds_per_cell)]
    logged = load_log(log_path)
    if len(logged):
        frames.append(logged)
    replay = load_log(REPLAY_PATH)
    if len(replay):
        frames.append(replay)
    df = pd.concat(frames, ignore_index=True)
    if "event_id" in df:
        df = df.drop_duplicates(subset="event_id").reset_index(drop=True)

    challenger = _evaluate(df, baseline_seed, alert_rate, test_share)
    incumbent = _incumbent_metrics(champion_dir)
    promote = _should_promote(
        challenger,
        incumbent,
        min_delta,
        fp_tol,
        min_improvement,
    )

    decision: dict[str, Any] = {
        "generation": generation,
        "rows": int(len(df)),
        "fraud_rows": int((df["label"] == 1).sum()),
        "challenger": {k: challenger[k] for k in ("chain_recall", "hard_false_positive_rate", "chain_precision", "prevented_share")},
        "incumbent": incumbent,
        "promoted": promote,
        "min_delta": min_delta,
        "fp_tolerance": fp_tol,
        "minimum_improvement": min_improvement,
    }

    if promote:
        X, y, groups, meta = build_feature_frame(df)
        champion = FraudDetector().fit(X, y)
        champion.threshold = threshold_for_budget(champion.score(X), max(1, round(alert_rate * len(X))))
        gate_metrics = {
            "chain_recall": challenger["chain_recall"],
            "hard_false_positive_rate": challenger["hard_false_positive_rate"],
            "chain_precision": challenger["chain_precision"],
            "prevented_share": challenger["prevented_share"],
        }
        bundle_meta = {
            "generation": generation,
            "trained_at": now_iso or datetime.now(timezone.utc).isoformat(),
            "rows": int(len(df)),
            "gate_metrics": gate_metrics,
            "challenger": decision["challenger"],
        }
        champion.save(champion_dir, meta=bundle_meta)
        # Version the promoted model.
        stamp = (now_iso or datetime.now(timezone.utc).isoformat()).replace(":", "").replace("-", "")[:15]
        champion.save(ARCHIVE_DIR / f"gen_{generation or 0}_{stamp}", meta=bundle_meta)
        update_replay_buffer(df)
        decision["champion_threshold"] = round(float(champion.threshold), 6)

    return decision
