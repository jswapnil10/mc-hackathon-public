"""Bulk training-row builder for the Blue-team ML pipeline (Phase 1).

Derives ONE tabular row per ObservedEvent (the source of truth is the event stream, NOT the
disconnected src/simulator/generate_dataset.py), covering all 6 families, and mixes in a large
population of ordinary standalone legit transactions plus standalone hard-negatives, so a future
classifier trains on realistic (~0.75%) imbalance with matched look-alikes.

NO model is trained here. This produces + audits the dataset only; the classifier is deferred.
Run:  python -m sentinelloop.dataset --seed 42 --out data/loop/dataset_42.parquet
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from red_team_agent.catalog import AttackCatalog
from red_team_agent.planner import RedTeamAgent

from .contracts import ObservedEvent, TruthRecord
from .simulation import account_baseline_inr, simulate_attack, simulate_legitimate_controls

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS_PER_CELL = 4          # scenarios per (family, difficulty)
TARGET_FRAUD_SHARE = 0.0075  # matches the sibling repo's realistic imbalance


# --------------------------------------------------------------------------- row shaping
def _row(event: ObservedEvent, truth: TruthRecord, *, group_key: str, chain_id: str | None) -> dict[str, Any]:
    """Flatten one event+truth into a training row. label == genuinely-malicious stage."""
    row: dict[str, Any] = dict(event.attributes)  # attribute columns
    row.update(
        event_id=event.event_id,
        occurred_at=event.occurred_at,
        event_type=event.event_type,
        sequence=event.sequence,
        chain_id=chain_id,
        group_key=group_key,           # StratifiedGroupKFold group (chain, or account for ambient)
        attack_family=truth.attack_family,
        is_attack=truth.is_attack,
        fraud_contributing=truth.fraud_contributing,
        value_at_risk_inr=truth.value_at_risk_inr,
        label=int(truth.is_attack and truth.fraud_contributing),
        source="attack" if truth.is_attack else ("control" if chain_id else "ambient"),
    )
    return row


# --------------------------------------------------------------------------- legit traffic
def make_accounts(n: int, rng: random.Random) -> list[str]:
    return [f"syn_acct_{rng.randint(100000, 999999)}" for _ in range(n)]


def _ambient_events(accounts: list[str], rng: random.Random, n_rows: int, start: datetime) -> list[dict[str, Any]]:
    """Ordinary standalone legit transactions (no kill chain). Uses the catalog attribute space
    and the SAME per-account baseline as attacks, so amounts overlap (kills L4)."""
    # Common everyday events (high weight) + the rarer, process-bound event types (low weight) so
    # ordinary legit traffic ALSO exercises ACCOUNT_WARMUP_ACTIVITY / IDENTITY_* / auth-change /
    # supplier-change. Without this, any attribute living only on those events is NaN across all
    # ambient rows, and "feature missing" becomes a simulator-artifact proxy for chain membership.
    event_types = [
        "PAYMENT_INITIATED", "SESSION_STARTED", "BENEFICIARY_ADDED", "FUNDS_RECEIVED", "PAYMENT_REPEATED",
        "ACCOUNT_WARMUP_ACTIVITY", "IDENTITY_APPLICATION_SUBMITTED", "IDENTITY_VERIFICATION_ATTEMPTED",
        "ACCOUNT_OPENED", "AUTHENTICATION_CONTEXT_CHANGED", "SUPPLIER_PROFILE_CHANGED",
    ]
    weights = [40, 16, 10, 9, 7, 5, 3, 3, 3, 2, 2]
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        acct = rng.choice(accounts)
        et = rng.choices(event_types, weights=weights)[0]
        base = account_baseline_inr(acct)
        ratio = round(rng.uniform(0.3, 2.2), 4)
        ts = start + timedelta(minutes=rng.randint(0, 90 * 24 * 60))
        attrs: dict[str, Any] = {
            "sender_account_id": acct,
            "beneficiary_id": f"syn_beneficiary_{rng.randint(100000, 999999)}",
            "device_id": f"syn_device_{rng.randint(100000, 999999)}",
            "beneficiary_age_days": rng.randint(0, 1800),
            "session_age_seconds": rng.randint(30, 4000),
        }
        if et in {"PAYMENT_INITIATED", "PAYMENT_REPEATED", "FUNDS_RECEIVED"}:
            attrs["sender_baseline_amount_inr"] = base
            attrs["amount_inr"] = round(base * ratio, 2)
            attrs["amount_to_baseline_ratio"] = ratio
            attrs["customer_confirmation_complete"] = rng.random() < 0.8  # genuine payments mostly confirmed
        if et == "PAYMENT_REPEATED":
            attrs["payment_count"] = rng.randint(2, 6)
        # Phase-4 canonicals + the widened-allowlist signals: emit on the SAME event types the cards
        # attach them to, using the SAME overlapping legit draws as simulation._benignize, so every
        # feature appears in the ambient source too (closes the L2 presence gap).
        if et == "PAYMENT_INITIATED":
            attrs["balance_drain_ratio"] = round(rng.uniform(0.05, 0.9), 4)
            attrs["add_to_payment_seconds"] = (
                rng.randint(30, 1800) if rng.random() < 0.15 else rng.randint(120, 86400)
            )
            attrs["round_amount_ratio"] = round(rng.uniform(0.0, 0.8), 4)
        if et == "FUNDS_RECEIVED":
            attrs["receiver_inbound_sender_count"] = rng.randint(5, 150)
            attrs["funds_dwell_time_seconds"] = (
                rng.randint(30, 3600) if rng.random() < 0.15 else rng.randint(600, 172800)
            )
            attrs["sender_count"] = rng.randint(2, 20)  # ordinary receivers see a spread of senders
        if et in {"BENEFICIARY_ADDED", "FUNDS_RECEIVED"}:
            attrs["beneficiary_kyc_tier"] = rng.choices(
                ["full_kyc", "business_kyc", "min_kyc"], weights=[55, 25, 20]
            )[0]
        if et == "BENEFICIARY_ADDED":
            attrs["beneficiary_name_match_score"] = (
                round(rng.uniform(0.4, 0.85), 4) if rng.random() < 0.15 else round(rng.uniform(0.7, 1.0), 4)
            )
        if et == "SESSION_STARTED":
            attrs["device_is_new"] = rng.random() < 0.15
            attrs["network_is_new"] = rng.random() < 0.15
            attrs["device_binding_age_days"] = (
                rng.randint(0, 60) if rng.random() < 0.15 else rng.randint(30, 1500)
            )
            attrs["dormancy_days_before_activity"] = (
                rng.randint(120, 900) if rng.random() < 0.2 else rng.randint(0, 120)
            )
        if et == "ACCOUNT_WARMUP_ACTIVITY":
            attrs["account_age_days"] = (
                rng.randint(0, 120) if rng.random() < 0.15 else rng.randint(30, 2000)
            )
            attrs["warmup_event_count"] = rng.randint(0, 15)
            attrs["device_shared_across_accounts"] = rng.random() < 0.15
            attrs["network_shared_across_accounts"] = rng.random() < 0.15
            attrs["device_binding_age_days"] = (
                rng.randint(0, 60) if rng.random() < 0.15 else rng.randint(30, 1500)
            )
            attrs["dormancy_days_before_activity"] = (
                rng.randint(120, 900) if rng.random() < 0.2 else rng.randint(0, 120)
            )
        if et == "IDENTITY_APPLICATION_SUBMITTED":
            attrs["profile_consistency_score"] = round(rng.uniform(0.5, 1.0), 4)
            attrs["linked_attribute_count"] = rng.randint(0, 8)
        if et == "IDENTITY_VERIFICATION_ATTEMPTED":
            attrs["document_consistency_score"] = round(rng.uniform(0.6, 1.0), 4)
            attrs["liveness_confidence"] = round(rng.uniform(0.5, 1.0), 4)
            attrs["verification_retry_count"] = rng.randint(0, 5)  # accessibility retries happen legitimately
        if et == "ACCOUNT_OPENED":
            attrs["device_reused_across_profiles"] = rng.random() < 0.15
        if et == "AUTHENTICATION_CONTEXT_CHANGED":
            attrs["change_risk_band"] = rng.randint(1, 5)  # genuine security-setting changes span the band
        if et == "SUPPLIER_PROFILE_CHANGED":
            attrs["supplier_relationship_days"] = rng.randint(30, 3000)
            attrs["approval_step_count"] = rng.randint(1, 5)
            attrs["urgency_level"] = rng.randint(0, 4)  # legit business changes carry an urgency band too
        ev = ObservedEvent(f"syn_amb_{i:08d}", 1, ts.isoformat(), et, [], attrs)
        tr = TruthRecord(ev.event_id, "ambient", None, f"AMB-{acct}", "DECIDE", False, 0.0, 0, False)
        rows.append(_row(ev, tr, group_key=acct, chain_id=None))
    return rows


def _hard_negative_traps(accounts: list[str], rng: random.Random, n_each: int, start: datetime) -> list[dict[str, Any]]:
    """Standalone legit transactions that each mimic ONE attack stage in isolation, so no single
    suspicious-looking value (big amount / new payee / new device) is a fraud marker on its own."""
    rows: list[dict[str, Any]] = []
    kinds = ["big_ticket", "new_payee", "new_device", "reactivation"]
    for kind in kinds:
        for i in range(n_each):
            acct = rng.choice(accounts)
            base = account_baseline_inr(acct)
            ts = start + timedelta(minutes=rng.randint(0, 90 * 24 * 60))
            attrs: dict[str, Any] = {"sender_account_id": acct, "session_age_seconds": rng.randint(30, 4000)}
            et = "PAYMENT_INITIATED"
            if kind == "big_ticket":
                ratio = round(rng.uniform(6.0, 14.0), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, beneficiary_age_days=rng.randint(200, 1500))
            elif kind == "new_payee":
                ratio = round(rng.uniform(0.6, 1.8), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, beneficiary_age_days=0,
                             beneficiary_id=f"syn_beneficiary_{rng.randint(100000, 999999)}")
            elif kind == "new_device":
                et = "SESSION_STARTED"
                attrs.update(device_id=f"syn_device_{rng.randint(100000, 999999)}", device_is_new=True,
                             strong_authentication_passed=True)
            else:  # reactivation
                ratio = round(rng.uniform(0.5, 1.5), 4)
                attrs.update(sender_baseline_amount_inr=base, amount_inr=round(base * ratio, 2),
                             amount_to_baseline_ratio=ratio, dormancy_days_before_activity=rng.randint(180, 900))
            ev = ObservedEvent(f"syn_trap_{kind}_{i:06d}", 1, ts.isoformat(), et, [], attrs)
            tr = TruthRecord(ev.event_id, f"trap_{kind}", None, f"TRAP-{acct}", "DECIDE", False, 0.0, 0, False)
            rows.append(_row(ev, tr, group_key=acct, chain_id=None))
    return rows


# --------------------------------------------------------------------------- build + audit
def build_dataset(
    seed: int = 42,
    *,
    families: list[str] | None = None,
    n_accounts: int = 400,
    seeds_per_cell: int = SEEDS_PER_CELL,
    target_fraud_share: float = TARGET_FRAUD_SHARE,
) -> pd.DataFrame:
    rng = random.Random(seed)
    agent = RedTeamAgent()
    families = families or AttackCatalog().families
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)

    rows: list[dict[str, Any]] = []
    # 1. Attack chains (+ their full-length benign control chains) via the existing safe path.
    #    Deterministic seed per cell (no hash() — it is randomized per process).
    for fi, family in enumerate(families):
        for di, difficulty in enumerate(DIFFICULTIES):
            for k in range(seeds_per_cell):
                spec_seed = seed * 100000 + fi * 1000 + di * 100 + k
                spec = agent.plan(attack_family=family, difficulty=difficulty, seed=spec_seed)
                case = simulate_attack(spec)
                for ev, tr in zip(case.events, case.truth):
                    rows.append(_row(ev, tr, group_key=case.case_id, chain_id=case.case_id))
                for ctrl in simulate_legitimate_controls(spec, spec.legitimate_controls):
                    for ev, tr in zip(ctrl.events, ctrl.truth):
                        rows.append(_row(ev, tr, group_key=ctrl.case_id, chain_id=ctrl.case_id))

    # 2. Size ambient legit so genuinely-malicious rows land at ~TARGET_FRAUD_SHARE.
    n_fraud = sum(r["label"] for r in rows)
    accounts = make_accounts(n_accounts, rng)
    rows += _hard_negative_traps(accounts, rng, n_each=max(20, n_accounts // 10), start=start)
    n_ambient = max(0, int(n_fraud / target_fraud_share) - len(rows))
    rows += _ambient_events(accounts, rng, n_ambient, start)

    return pd.DataFrame(rows)


def coverage_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Presence/absence (L2) leakage check: every event_type must appear in BOTH attack and
    non-attack rows, else its mere presence separates classes. (We check `is_attack`, not
    `label` — benign-in-isolation stages legitimately never carry label=1.)"""
    per_type = df.groupby("event_type").agg(
        count=("label", "size"),
        malicious=("label", "sum"),
        in_attack=("is_attack", "max"),
        in_legit=("is_attack", lambda s: bool((~s).any())),
    )
    attack_only = per_type[per_type["in_attack"] & ~per_type["in_legit"]]
    return {
        "rows": len(df),
        "fraud_rows": int(df["label"].sum()),
        "fraud_share": round(df["label"].mean(), 5),
        "by_source": df["source"].value_counts().to_dict(),
        "event_types": per_type.reset_index().to_dict("records"),
        "attack_only_event_types": attack_only.index.tolist(),  # L2 leak if non-empty
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Blue-team ML training dataset (no model trained).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accounts", type=int, default=400)
    ap.add_argument("--seeds-per-cell", type=int, default=SEEDS_PER_CELL,
                    help="scenarios per family x difficulty; raise to grow fraud rows AND total (keeps imbalance)")
    ap.add_argument("--fraud-share", type=float, default=TARGET_FRAUD_SHARE,
                    help="target fraud fraction; lower it to add more ambient legit (bigger total, rarer positives)")
    ap.add_argument("--out", type=Path, default=Path("data/loop/dataset_42.parquet"))
    args = ap.parse_args()

    df = build_dataset(args.seed, n_accounts=args.accounts,
                       seeds_per_cell=args.seeds_per_cell, target_fraud_share=args.fraud_share)
    report = coverage_audit(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = args.out
    if args.out.suffix.lower() == ".csv":
        df.to_csv(written, index=False)          # honour an explicit .csv request
    else:
        try:
            df.to_parquet(args.out, index=False)
        except Exception:  # pyarrow may be absent; fall back to CSV
            written = args.out.with_suffix(".csv")
            df.to_csv(written, index=False)

    print(f"wrote {written}  ({report['rows']:,} rows, {report['fraud_rows']:,} fraud, "
          f"{report['fraud_share']:.3%} share)")
    print(f"by source: {report['by_source']}")
    if report["attack_only_event_types"]:
        print(f"WARNING attack-only event_types (L2 presence leak): {report['attack_only_event_types']}")
    else:
        print("coverage OK: every event_type appears in both attack and non-attack rows")


if __name__ == "__main__":
    main()
