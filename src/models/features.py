"""Point-in-time feature engineering for synthetic payment events."""

from __future__ import annotations

import math

import pandas as pd


MODEL_FEATURES = [
    "amount_log",
    "session_age_log",
    "account_age_log",
    "beneficiary_age_log",
    "event_hour_sin",
    "event_hour_cos",
    "sender_prior_event_count_log",
    "amount_to_prior_mean",
    "is_new_device",
    "is_new_ip",
    "is_new_beneficiary",
    "beneficiary_prior_sender_count_log",
    "device_prior_account_count_log",
    "ip_prior_account_count_log",
    "payment_rail",
    "channel",
    "auth_method",
    "merchant_category",
]

NUMERIC_FEATURES = MODEL_FEATURES[:14]
CATEGORICAL_FEATURES = MODEL_FEATURES[14:]
FORBIDDEN_MODEL_INPUTS = {
    "label_fraud", "attack_family", "scenario_id", "legitimate_control", "event_id", "transaction_id"
}


def build_point_in_time_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build features using only rows that occurred before each event.

    Raw identifiers are used only for history aggregation and are excluded from
    MODEL_FEATURES, preventing the baseline from memorizing synthetic IDs.
    """
    required = {
        "event_ts",
        "sender_account_id",
        "beneficiary_id",
        "device_id",
        "ip_prefix",
        "amount_inr",
        "session_age_seconds",
        "account_age_days",
        "beneficiary_age_days",
        "payment_rail",
        "channel",
        "auth_method",
        "merchant_category",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required event fields: {sorted(missing)}")

    frame = events.copy()
    frame["_original_order"] = range(len(frame))
    frame["event_ts"] = pd.to_datetime(frame["event_ts"], utc=True)
    frame = frame.sort_values(["event_ts", "_original_order"], kind="stable").reset_index(drop=True)

    sender_group = frame.groupby("sender_account_id", sort=False)
    frame["sender_prior_event_count"] = sender_group.cumcount()
    prior_amount_sum = sender_group["amount_inr"].cumsum() - frame["amount_inr"]
    prior_mean = prior_amount_sum / frame["sender_prior_event_count"].where(
        frame["sender_prior_event_count"].ne(0), float("nan")
    )
    fallback_mean = float(frame["amount_inr"].median())
    frame["amount_to_prior_mean"] = (
        frame["amount_inr"] / prior_mean.fillna(fallback_mean).clip(lower=1.0)
    ).clip(upper=50.0)

    frame["_sender_device_prior"] = frame.groupby(
        ["sender_account_id", "device_id"], sort=False
    ).cumcount()
    frame["_sender_ip_prior"] = frame.groupby(
        ["sender_account_id", "ip_prefix"], sort=False
    ).cumcount()
    frame["_sender_beneficiary_prior"] = frame.groupby(
        ["sender_account_id", "beneficiary_id"], sort=False
    ).cumcount()
    has_history = frame["sender_prior_event_count"] > 0
    frame["is_new_device"] = (has_history & frame["_sender_device_prior"].eq(0)).astype(int)
    frame["is_new_ip"] = (has_history & frame["_sender_ip_prior"].eq(0)).astype(int)
    frame["is_new_beneficiary"] = (has_history & frame["_sender_beneficiary_prior"].eq(0)).astype(int)

    # Cumulative unique relationships are derived without looking at future rows.
    first_beneficiary_sender = ~frame.duplicated(["beneficiary_id", "sender_account_id"])
    frame["beneficiary_prior_sender_count"] = (
        first_beneficiary_sender.astype(int).groupby(frame["beneficiary_id"]).cumsum()
        - first_beneficiary_sender.astype(int)
    )
    first_device_account = ~frame.duplicated(["device_id", "sender_account_id"])
    frame["device_prior_account_count"] = (
        first_device_account.astype(int).groupby(frame["device_id"]).cumsum()
        - first_device_account.astype(int)
    )
    first_ip_account = ~frame.duplicated(["ip_prefix", "sender_account_id"])
    frame["ip_prior_account_count"] = (
        first_ip_account.astype(int).groupby(frame["ip_prefix"]).cumsum()
        - first_ip_account.astype(int)
    )

    frame["amount_log"] = frame["amount_inr"].map(lambda value: math.log1p(max(0.0, value)))
    frame["session_age_log"] = frame["session_age_seconds"].map(lambda value: math.log1p(max(0, value)))
    frame["account_age_log"] = frame["account_age_days"].map(lambda value: math.log1p(max(0, value)))
    frame["beneficiary_age_log"] = frame["beneficiary_age_days"].map(lambda value: math.log1p(max(0, value)))
    frame["sender_prior_event_count_log"] = frame["sender_prior_event_count"].map(math.log1p)
    frame["beneficiary_prior_sender_count_log"] = frame["beneficiary_prior_sender_count"].map(math.log1p)
    frame["device_prior_account_count_log"] = frame["device_prior_account_count"].map(math.log1p)
    frame["ip_prior_account_count_log"] = frame["ip_prior_account_count"].map(math.log1p)
    hour = frame["event_ts"].dt.hour + frame["event_ts"].dt.minute / 60.0
    frame["event_hour_sin"] = (hour * 2.0 * math.pi / 24.0).map(math.sin)
    frame["event_hour_cos"] = (hour * 2.0 * math.pi / 24.0).map(math.cos)

    if FORBIDDEN_MODEL_INPUTS.intersection(MODEL_FEATURES):
        raise AssertionError("A ground-truth or identity field entered MODEL_FEATURES.")
    return frame
