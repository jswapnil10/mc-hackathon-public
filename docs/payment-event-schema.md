# Payment-event data contract

This contract is the shared interface between the simulator, feature pipeline, models, and dashboard. All values are synthetic. UUIDs are preferred for identifiers and all event timestamps are UTC ISO-8601 strings.

## Transaction event

| Field | Type | Required | Description |
|---|---|---:|---|
| `event_id` | string | yes | Unique payment-event ID. |
| `event_ts` | datetime | yes | Time at which the payment was attempted. |
| `transaction_id` | string | yes | Payment attempt ID; may be reused only for lifecycle updates. |
| `sender_account_id` | string | yes | Synthetic payer account ID. |
| `beneficiary_id` | string | yes | Synthetic merchant or recipient ID. |
| `payment_rail` | enum | yes | `UPI`, `CARD`, `BANK_TRANSFER`, or `WALLET`. |
| `channel` | enum | yes | `MOBILE_APP`, `WEB`, `POS`, `API`, or `ATM`. |
| `amount_inr` | decimal | yes | Positive attempted amount in INR. |
| `currency` | string | yes | ISO 4217; prototype uses `INR`. |
| `merchant_category` | string | conditional | Required for merchant payments; `P2P_TRANSFER` for person-to-person payments. |
| `transaction_status` | enum | yes | `APPROVED`, `DECLINED`, `HELD`, `CHALLENGED`, or `REVERSED`. |
| `device_id` | string | yes | Pseudonymous device ID. |
| `ip_prefix` | string | yes | Privacy-preserving network prefix, never a raw IP address. |
| `geo_region` | string | yes | Coarse state/region only. |
| `auth_method` | enum | yes | `BIOMETRIC`, `PIN`, `OTP`, `PASSWORD`, or `CARD_CVV`. |
| `session_age_seconds` | integer | yes | Seconds since the user session began. |
| `account_age_days` | integer | yes | Account age at event time. |
| `beneficiary_age_days` | integer | yes | Sender-beneficiary relationship age. |
| `label_fraud` | boolean | yes | Ground-truth simulator label; unavailable to production inference. |
| `attack_family` | enum/null | yes | Null for legitimate events; otherwise an approved taxonomy ID. |
| `scenario_id` | string/null | yes | Red-team campaign identifier, null for normal behavior. |
| `legitimate_control` | string/null | yes | Ground-truth tag for difficult legitimate look-alikes; excluded from model inputs. |

## Entity tables

| Entity | Key | Essential fields |
|---|---|---|
| Account | `account_id` | customer segment, opening timestamp, home region, historical behavioral profile, risk state |
| Beneficiary | `beneficiary_id` | type (merchant/P2P), category, creation time, settlement group |
| Device | `device_id` | first-seen timestamp, OS family, integrity state, linked-account count |
| Campaign | `scenario_id` | attack family, generation parameters, mutation parent, simulation run, expected outcomes |
| Alert | `alert_id` | event/account/cluster target, risk score, reasons, recommendation, analyst disposition |

## Derived features (not simulator labels)

Models must derive these exclusively from data available before the decision timestamp:

- Transaction amount deviation from the sender's own historical distribution
- Payment count and amount velocity for 5 minutes, 1 hour, and 24 hours
- New-device, new-network, and new-beneficiary indicators
- Time-of-day and channel deviation
- Beneficiary fan-in, device fan-out, and shared-network counts
- Graph features: component size, two-hop fan-in, community risk density, and circular-flow indicator
- ATO composite: credential/session novelty plus recipient/amount/velocity change

## Data-quality and leakage rules

- Split train/test by **campaign and time**, never random rows alone.
- Do not expose `label_fraud`, `attack_family`, `scenario_id`, `legitimate_control`, alert disposition, or post-event status to model features.
- Use only coarse locations and network prefixes; no real personal, payment-card, or bank data.
- Ensure every fraud scenario has legitimate near-neighbors so the model cannot win on a single artificial field.
