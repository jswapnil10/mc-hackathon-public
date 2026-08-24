"""Generate deterministic, fully synthetic payment events for SentinelLoop.

The generator intentionally models coarse behavioral signals rather than real
payment identities or rails. It is suitable for model prototyping and demos,
not for operational fraud decisioning.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PAYMENT_RAILS = ("UPI", "CARD", "BANK_TRANSFER", "WALLET")
CHANNELS = ("MOBILE_APP", "WEB", "POS", "API")
REGIONS = ("MH", "DL", "KA", "TN", "GJ", "TS", "WB", "UP", "RJ", "KL")
MCCS = ("GROCERY", "FUEL", "DINING", "TRAVEL", "ECOMMERCE", "UTILITIES", "P2P_TRANSFER")


@dataclass(frozen=True)
class AccountProfile:
    account_id: str
    home_region: str
    account_age_days: int
    primary_device_id: str
    primary_ip_prefix: str
    mean_amount: float
    activity_weight: float


def _id(kind: str, sequence: int) -> str:
    return f"{kind}_{sequence:06d}"


def _event_id(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _auth_for_rail(payment_rail: str, rng: random.Random) -> str:
    """Choose a coarse authentication method that is plausible for the rail."""
    choices = {
        "UPI": ("PIN", "BIOMETRIC"),
        "CARD": ("CARD_CVV", "OTP"),
        "BANK_TRANSFER": ("PASSWORD", "OTP", "BIOMETRIC"),
        "WALLET": ("PIN", "PASSWORD", "BIOMETRIC"),
    }
    return rng.choice(choices[payment_rail])


def make_accounts(count: int, rng: random.Random) -> list[AccountProfile]:
    """Create synthetic account behavioral profiles."""
    accounts: list[AccountProfile] = []
    for index in range(count):
        home_region = rng.choice(REGIONS)
        subnet = rng.randint(1, 254)
        accounts.append(
            AccountProfile(
                account_id=_id("acct", index),
                home_region=home_region,
                account_age_days=rng.randint(45, 3650),
                primary_device_id=_id("dev", index),
                primary_ip_prefix=f"10.{rng.randint(1, 200)}.{subnet}.0/24",
                mean_amount=round(rng.lognormvariate(6.4, 0.65), 2),
                activity_weight=rng.uniform(0.45, 1.75),
            )
        )
    return accounts


def _base_event(
    *,
    rng: random.Random,
    account: AccountProfile,
    event_ts: datetime,
    beneficiary_id: str,
    amount_inr: float,
    payment_rail: str,
    channel: str,
    merchant_category: str,
    device_id: str | None = None,
    ip_prefix: str | None = None,
    geo_region: str | None = None,
    auth_method: str | None = None,
    session_age_seconds: int | None = None,
    beneficiary_age_days: int | None = None,
    label_fraud: bool = False,
    attack_family: str | None = None,
    scenario_id: str | None = None,
    legitimate_control: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": _event_id(rng),
        "event_ts": event_ts.isoformat(),
        "transaction_id": _event_id(rng),
        "sender_account_id": account.account_id,
        "beneficiary_id": beneficiary_id,
        "payment_rail": payment_rail,
        "channel": channel,
        "amount_inr": round(max(1.0, amount_inr), 2),
        "currency": "INR",
        "merchant_category": merchant_category,
        "transaction_status": "APPROVED",
        "device_id": device_id or account.primary_device_id,
        "ip_prefix": ip_prefix or account.primary_ip_prefix,
        "geo_region": geo_region or account.home_region,
        "auth_method": auth_method or _auth_for_rail(payment_rail, rng),
        "session_age_seconds": session_age_seconds if session_age_seconds is not None else rng.randint(30, 2400),
        "account_age_days": account.account_age_days,
        "beneficiary_age_days": beneficiary_age_days if beneficiary_age_days is not None else rng.randint(14, 900),
        "label_fraud": label_fraud,
        "attack_family": attack_family,
        "scenario_id": scenario_id,
        "legitimate_control": legitimate_control,
    }


def generate_legitimate_events(
    accounts: list[AccountProfile], count: int, start: datetime, rng: random.Random
) -> list[dict[str, Any]]:
    """Generate ordinary payments, including legitimate near-neighbors of fraud."""
    events: list[dict[str, Any]] = []
    for index in range(count):
        account = rng.choices(accounts, weights=[a.activity_weight for a in accounts], k=1)[0]
        event_ts = start + timedelta(minutes=rng.randint(0, 7 * 24 * 60 - 1), seconds=rng.randint(0, 59))
        is_p2p = rng.random() < 0.42
        amount = account.mean_amount * rng.lognormvariate(0.0, 0.60)
        known_device = rng.random() < 0.95
        known_beneficiary = rng.random() < 0.83
        events.append(
            _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=_id("p2p" if is_p2p else "merchant", rng.randint(0, max(100, len(accounts) // 2))),
                amount_inr=amount,
                payment_rail="UPI" if is_p2p else rng.choice(PAYMENT_RAILS),
                channel="MOBILE_APP" if is_p2p else rng.choice(CHANNELS),
                merchant_category="P2P_TRANSFER" if is_p2p else rng.choice(MCCS[:-1]),
                device_id=account.primary_device_id if known_device else _id("dev_legit", index),
                ip_prefix=account.primary_ip_prefix if known_device else f"10.{rng.randint(1, 200)}.{rng.randint(1, 254)}.0/24",
                geo_region=account.home_region if rng.random() < 0.93 else rng.choice(REGIONS),
                beneficiary_age_days=rng.randint(30, 1200) if known_beneficiary else rng.randint(0, 7),
            )
        )
    return events


def generate_legitimate_controls(
    accounts: list[AccountProfile], count: int, start: datetime, rng: random.Random
) -> list[dict[str, Any]]:
    """Generate difficult but legitimate near-neighbors of the attack scenarios."""
    if count <= 0:
        return []
    events: list[dict[str, Any]] = []
    household_count = max(2, min(20, count // 8))
    popular_beneficiaries = [_id("popular", index) for index in range(max(2, count // 25))]
    for index in range(count):
        account = accounts[index % len(accounts)]
        event_ts = start + timedelta(minutes=rng.randint(0, 7 * 24 * 60 - 1), seconds=rng.randint(0, 59))
        mode = index % 4
        if mode == 0:
            # Genuine travel or phone replacement resembles an ATO event.
            event = _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=_id("travel_payee", index),
                amount_inr=account.mean_amount * rng.uniform(1.5, 5.0),
                payment_rail=rng.choice(("UPI", "CARD", "WALLET")),
                channel="MOBILE_APP",
                merchant_category=rng.choice(("TRAVEL", "ECOMMERCE", "P2P_TRANSFER")),
                device_id=_id("dev_replacement", index),
                ip_prefix=f"10.210.{index % 200}.{rng.randint(1, 254)}.0/24",
                geo_region=rng.choice(REGIONS),
                session_age_seconds=rng.randint(90, 1800),
                beneficiary_age_days=rng.randint(0, 7),
                legitimate_control="TRAVEL_NEW_DEVICE",
            )
        elif mode == 1:
            # Multiple family accounts can legitimately share a device/network.
            household = index % household_count
            event = _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=_id("family_payee", household),
                amount_inr=account.mean_amount * rng.uniform(0.5, 2.5),
                payment_rail="UPI",
                channel="MOBILE_APP",
                merchant_category="P2P_TRANSFER",
                device_id=_id("dev_household", household),
                ip_prefix=f"10.220.{household}.0/24",
                geo_region=account.home_region,
                session_age_seconds=rng.randint(120, 2400),
                beneficiary_age_days=rng.randint(30, 800),
                legitimate_control="SHARED_HOUSEHOLD",
            )
        elif mode == 2:
            # A popular merchant naturally has high fan-in from unrelated accounts.
            event = _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=rng.choice(popular_beneficiaries),
                amount_inr=account.mean_amount * rng.uniform(0.4, 2.2),
                payment_rail=rng.choice(PAYMENT_RAILS),
                channel=rng.choice(CHANNELS),
                merchant_category=rng.choice(MCCS[:-1]),
                beneficiary_age_days=rng.randint(90, 1500),
                legitimate_control="POPULAR_MERCHANT",
            )
        else:
            # A genuine first payment to a common cause can resemble coordinated fan-in.
            event = _base_event(
                rng=rng,
                account=account,
                event_ts=start + timedelta(days=3, minutes=rng.randint(0, 180)),
                beneficiary_id="charity_campaign_001",
                amount_inr=rng.choice((251.0, 501.0, 1001.0, 2001.0)),
                payment_rail="UPI",
                channel="MOBILE_APP",
                merchant_category="P2P_TRANSFER",
                beneficiary_age_days=rng.randint(0, 5),
                legitimate_control="CROWDFUNDING_BURST",
            )
        events.append(event)
    return events


def generate_ato_campaign(
    accounts: list[AccountProfile], count: int, start: datetime, rng: random.Random, campaign_number: int = 1
) -> list[dict[str, Any]]:
    """Simulate ATO payment attempts with behavioral novelty and new recipients."""
    scenario_id = f"ATO-01-{campaign_number:03d}"
    difficulty = ("easy", "medium", "hard")[(campaign_number - 1) % 3]
    parameters = {
        "easy": {"new_device": 0.95, "new_ip": 0.95, "amount": (3.5, 8.0), "beneficiary_days": (0, 2), "session": (15, 240)},
        "medium": {"new_device": 0.65, "new_ip": 0.65, "amount": (1.5, 4.5), "beneficiary_days": (0, 90), "session": (45, 1800)},
        "hard": {"new_device": 0.35, "new_ip": 0.40, "amount": (0.8, 2.5), "beneficiary_days": (0, 900), "session": (120, 3600)},
    }[difficulty]
    victims = rng.sample(accounts, k=min(count, len(accounts)))
    events: list[dict[str, Any]] = []
    for index, account in enumerate(victims):
        event_ts = start + timedelta(hours=rng.randint(4, 164), minutes=rng.randint(0, 59))
        events.append(
            _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=_id("ato_payee", campaign_number * 1000 + index % max(3, count // 5)),
                amount_inr=account.mean_amount * rng.uniform(*parameters["amount"]),
                payment_rail="UPI",
                channel="MOBILE_APP",
                merchant_category="P2P_TRANSFER",
                device_id=(
                    _id("dev_ato", campaign_number * 1000 + index)
                    if rng.random() < parameters["new_device"]
                    else account.primary_device_id
                ),
                ip_prefix=(
                    f"10.250.{campaign_number}.{index % 200}.0/24"
                    if rng.random() < parameters["new_ip"]
                    else account.primary_ip_prefix
                ),
                geo_region=rng.choice(REGIONS) if rng.random() < 0.55 else account.home_region,
                session_age_seconds=rng.randint(*parameters["session"]),
                beneficiary_age_days=rng.randint(*parameters["beneficiary_days"]),
                label_fraud=True,
                attack_family="ATO-01",
                scenario_id=scenario_id,
            )
        )
    return events


def generate_mule_campaign(
    accounts: list[AccountProfile], count: int, start: datetime, rng: random.Random, campaign_number: int = 1
) -> list[dict[str, Any]]:
    """Simulate coordinated sender-to-mule fan-in without real-world entities."""
    scenario_id = f"MULE-01-{campaign_number:03d}"
    senders = rng.sample(accounts, k=min(count, len(accounts)))
    difficulty = ("easy", "medium", "hard")[(campaign_number - 1) % 3]
    parameters = {
        "easy": {"shared_device": 0.85, "shared_ip": 0.90, "spread": 90, "mule_ratio": 8, "amount": (1.8, 5.2), "session": (30, 300), "beneficiary_days": (0, 15)},
        "medium": {"shared_device": 0.55, "shared_ip": 0.60, "spread": 360, "mule_ratio": 5, "amount": (1.1, 3.5), "session": (60, 1200), "beneficiary_days": (0, 180)},
        "hard": {"shared_device": 0.25, "shared_ip": 0.35, "spread": 1440, "mule_ratio": 3, "amount": (0.6, 2.2), "session": (120, 3000), "beneficiary_days": (0, 900)},
    }[difficulty]
    mule_count = max(2, min(16, count // parameters["mule_ratio"]))
    shared_ips = [f"10.240.{campaign_number}.{index}.0/24" for index in range(2)]
    events: list[dict[str, Any]] = []
    burst_start = start + timedelta(days=rng.randint(1, 6), hours=rng.randint(10, 20))
    for index, account in enumerate(senders):
        event_ts = burst_start + timedelta(minutes=rng.randint(0, parameters["spread"]), seconds=rng.randint(0, 59))
        mule_id = _id("mule", campaign_number * 1000 + index % mule_count)
        events.append(
            _base_event(
                rng=rng,
                account=account,
                event_ts=event_ts,
                beneficiary_id=mule_id,
                amount_inr=account.mean_amount * rng.uniform(*parameters["amount"]),
                payment_rail="UPI",
                channel="MOBILE_APP",
                merchant_category="P2P_TRANSFER",
                device_id=(
                    _id("dev_shared", campaign_number * 100 + index % 4)
                    if rng.random() < parameters["shared_device"]
                    else account.primary_device_id
                ),
                ip_prefix=rng.choice(shared_ips) if rng.random() < parameters["shared_ip"] else account.primary_ip_prefix,
                geo_region=account.home_region,
                session_age_seconds=rng.randint(*parameters["session"]),
                beneficiary_age_days=rng.randint(*parameters["beneficiary_days"]),
                label_fraud=True,
                attack_family="MULE-01",
                scenario_id=scenario_id,
            )
        )
    return events


def generate_dataset(
    *, account_count: int = 1000, legitimate_count: int = 10000, ato_count: int = 250,
    mule_count: int = 250, campaign_count: int = 6, seed: int = 20260817,
    start: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return sorted synthetic events and reproducibility metadata."""
    if min(account_count, legitimate_count, campaign_count) < 1 or min(ato_count, mule_count) < 0:
        raise ValueError("Counts must be non-negative; account and legitimate counts must be positive.")
    rng = random.Random(seed)
    start = start or datetime(2026, 8, 1, tzinfo=timezone.utc)
    accounts = make_accounts(account_count, rng)
    control_count = min(legitimate_count // 8, max(20, account_count // 2))
    events = generate_legitimate_events(accounts, legitimate_count - control_count, start, rng)
    events.extend(generate_legitimate_controls(accounts, control_count, start, rng))
    for family_count, generator in ((ato_count, generate_ato_campaign), (mule_count, generate_mule_campaign)):
        base_count, remainder = divmod(family_count, campaign_count)
        for campaign_index in range(campaign_count):
            count = base_count + (1 if campaign_index < remainder else 0)
            if count:
                events.extend(generator(accounts, count, start, rng, campaign_number=campaign_index + 1))
    frame = pd.DataFrame(events).sort_values("event_ts", kind="stable").reset_index(drop=True)
    metadata = {
        "dataset_version": "v0.2",
        "seed": seed,
        "start_timestamp": start.isoformat(),
        "accounts": account_count,
        "events": len(frame),
        "legitimate_events": int((~frame["label_fraud"]).sum()),
        "fraud_events": int(frame["label_fraud"].sum()),
        "legitimate_control_events": int(frame["legitimate_control"].notna().sum()),
        "campaign_count_per_attack_family": campaign_count,
        "attack_counts": frame.loc[frame["label_fraud"], "attack_family"].value_counts().to_dict(),
        "schema_fields": list(frame.columns),
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SentinelLoop synthetic payment dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--accounts", type=int, default=1000)
    parser.add_argument("--legitimate-events", type=int, default=10000)
    parser.add_argument("--ato-events", type=int, default=250)
    parser.add_argument("--mule-events", type=int, default=250)
    parser.add_argument("--campaigns-per-family", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    frame, metadata = generate_dataset(
        account_count=args.accounts,
        legitimate_count=args.legitimate_events,
        ato_count=args.ato_events,
        mule_count=args.mule_events,
        campaign_count=args.campaigns_per_family,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "payment_events.csv"
    metadata_path = args.output_dir / "payment_events.metadata.json"
    frame.to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frame):,} synthetic events to {csv_path}")
    print(f"Fraud events: {metadata['fraud_events']:,}; attack counts: {metadata['attack_counts']}")


if __name__ == "__main__":
    main()
