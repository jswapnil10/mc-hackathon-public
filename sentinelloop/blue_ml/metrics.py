"""Chain-level evaluation for the Blue-team detector.

Headline metrics are chain/economic, not row accuracy (a kill chain is caught if ANY of its
stages is alerted, and it matters WHEN). Legit friction is reported separately for the hard
look-alike controls vs ordinary ambient traffic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def threshold_for_budget(scores: np.ndarray, budget: int) -> float:
    """Score cutoff that alerts on the top-`budget` rows (mirrors the sibling repo)."""
    scores = np.asarray(scores, dtype=float)
    if budget <= 0:
        return float("inf")
    if budget >= len(scores):
        return 0.0
    return float(np.partition(scores, -budget)[-budget])


def summarise(
    meta: pd.DataFrame,
    y: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    m = meta.reset_index(drop=True).copy()
    m["label"] = np.asarray(y).astype(int)
    m["score"] = np.asarray(scores, dtype=float)
    m["alert"] = m["score"] >= threshold
    # A chain = rows sharing chain_id; standalone (ambient/trap) rows are singleton chains.
    m["chain_key"] = m["chain_id"].where(m["chain_id"].notna(), other="solo_" + m["event_id"].astype(str))

    total = len(m)
    n_alert = int(m["alert"].sum())

    # ---- row metrics (for completeness) ----
    pos = m["label"] == 1
    row_recall = float(m.loc[pos, "alert"].mean()) if pos.any() else 0.0
    row_precision = float(m.loc[m["alert"], "label"].mean()) if n_alert else 0.0

    # ---- legit friction, split by source ----
    def _alert_rate(src: str) -> float:
        sub = m[m["source"] == src]
        return float(sub["alert"].mean()) if len(sub) else 0.0

    hard_false_positive_rate = _alert_rate("control")   # hard look-alike chains
    ambient_friction_rate = _alert_rate("ambient")       # ordinary traffic

    # ---- chain-level ----
    chains = m.groupby("chain_key")
    is_attack_chain = chains["is_attack"].any()
    caught = chains["alert"].any()
    family = chains["attack_family"].first()

    attack_keys = is_attack_chain[is_attack_chain].index
    legit_keys = is_attack_chain[~is_attack_chain].index
    attack_caught = caught.reindex(attack_keys).fillna(False)
    chains_total = int(len(attack_keys))
    chains_caught = int(attack_caught.sum())
    chain_recall = chains_caught / chains_total if chains_total else 0.0

    # per-family chain recall
    chain_recall_by_family: dict[str, float] = {}
    fam_of_attack = family.reindex(attack_keys)
    for fam in sorted(set(fam_of_attack.dropna())):
        keys = fam_of_attack[fam_of_attack == fam].index
        cr = caught.reindex(keys).fillna(False)
        chain_recall_by_family[str(fam)] = round(float(cr.mean()), 4) if len(cr) else 0.0

    # caught_at_stage + prevented (alerted at/before the first fraud-contributing stage)
    caught_at_stage: list[int] = []
    prevented = 0
    for key in attack_keys:
        rows = m[m["chain_key"] == key]
        alerted = rows[rows["alert"]]
        if alerted.empty:
            continue
        first_alert_seq = int(alerted["sequence"].min())
        caught_at_stage.append(first_alert_seq)
        malicious = rows[(rows["fraud_contributing"] == True) & (rows["value_at_risk_inr"].fillna(0) > 0)]  # noqa: E712
        if malicious.empty:
            malicious = rows[rows["fraud_contributing"] == True]  # noqa: E712
        first_malicious_seq = int(malicious["sequence"].min()) if not malicious.empty else 10**9
        if first_alert_seq <= first_malicious_seq:
            prevented += 1
    prevented_share = prevented / chains_caught if chains_caught else 0.0

    # chain precision: caught attack chains / (caught attack chains + legit chains alerted)
    legit_alerted = int(caught.reindex(legit_keys).fillna(False).sum())
    denom = chains_caught + legit_alerted
    chain_precision = chains_caught / denom if denom else 0.0

    return {
        "rows_test": total,
        "fraud_rows": int(pos.sum()),
        "alerts": n_alert,
        "threshold": round(float(threshold), 6),
        "chain_recall": round(chain_recall, 4),
        "chains_caught": chains_caught,
        "chains_total": chains_total,
        "chain_recall_by_family": chain_recall_by_family,
        "prevented_share": round(prevented_share, 4),
        "mean_caught_at_stage": round(float(np.mean(caught_at_stage)), 3) if caught_at_stage else None,
        "chain_precision": round(chain_precision, 4),
        "legit_chains_alerted": legit_alerted,
        "row_recall": round(row_recall, 4),
        "row_precision": round(row_precision, 4),
        "hard_false_positive_rate": round(hard_false_positive_rate, 5),
        "ambient_friction_rate": round(ambient_friction_rate, 5),
    }
