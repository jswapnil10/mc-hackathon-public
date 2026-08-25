# Population and defense benchmark methodology

This benchmark answers a narrow question: **can SentinelLoop detect its own synthetic attacks on cases that were sealed before model selection, without using the answer key as an input?** It does not claim production accuracy on live payment traffic.

## Evaluation contract

- **Grain:** one sanitized observable payment-lifecycle event per row.
- **Positive label:** an event emitted by a bounded attack scenario.
- **Negative label:** an ordinary legitimate event or a family-specific legitimate look-alike.
- **Primary metrics:** hidden-test PR-AUC, novel-vector recall and hard false-positive rate.
- **Supporting metrics:** ROC-AUC, precision, recall, F1, value-weighted recall, per-family recall and phase-level recall/customer safety.
- **Fidelity reference:** declared behavioral priors and structural validity checks. No real customer data is used.

## Sealed splits

| Split | Purpose | Model access |
|---|---|---|
| `train` | Fit candidate detectors | Training only |
| `validation` | Select the detector and decision threshold | Selection only |
| `test_known` | Measure unseen cases for represented vectors | Opened after selection |
| `test_novel` | Measure complete vector IDs withheld from train and validation | Opened after selection |

Whole scenarios and sender entities remain inside one split. The quality gate stops the benchmark if either crosses a split. The seven novel holdout vector IDs appear only in `test_novel`.

## Inputs available to the detector

The detector may use sanitized payment and telemetry fields such as event type, lifecycle phase, amount versus customer baseline, rail, channel, authentication method, timing, velocity, beneficiary age, visible trust failures and independent verification.

The following are explicitly forbidden: fraud label, attack family, attack vector, difficulty, scenario ID, case ID, case role, split, event ID, value-at-risk truth and legitimate-control name. This exclusion is enforced in code and tests.

## Candidate selection

SentinelLoop fits two reproducible candidates:

1. balanced logistic regression with numeric scaling and one-hot categorical encoding;
2. histogram gradient boosting with balanced training weights.

The higher validation PR-AUC wins. The operating threshold is also selected on validation, preferring at least 90% precision and then the best F1/recall trade-off. Test labels do not choose the model or threshold.

## Fidelity checks

The generator is scored on:

- rail and channel distribution similarity to declared priors;
- amount calibration relative to each synthetic customer's baseline;
- amount/baseline correlation;
- the expected legitimate edge-case rate;
- attack-vector and family coverage;
- valid event-sequence ordering;
- whether harder scenarios expose fewer obvious risk signals.

The score must always be presented as **synthetic fidelity against declared priors**. A future production pilot must replace or recalibrate those priors with privacy-reviewed aggregate network statistics.

## Data-quality gates

The run checks required schema, event-ID uniqueness, valid splits, scenario isolation, entity isolation, label integrity, amount range, core completeness, class coverage, complete vector coverage and the novel-vector seal. Any critical failure stops training.

## Reproduce the current artifact

```bash
python -m sentinelloop population-benchmark \
  --variants-per-vector 6 \
  --legitimate-events 2400 \
  --seed 20260824
```

Outputs are written to `data/benchmark/`:

- `latest.json` — complete KPI, fidelity, quality and model-selection report;
- `population_metadata.json` — schema and population coverage;
- `data_quality.json` — every gate and its observed result;
- `population_events.csv` — generated event population (ignored by Git);
- `defense_predictions.csv` — sealed-test predictions (ignored by Git).

## Current reproducible result

For seed `20260824`, six variants per vector and 2,400 ordinary legitimate events:

- 4,446 events and 186 attack campaigns;
- 31/31 threat vectors and 9/9 attack families represented;
- all data-quality gates passed;
- fidelity score: 93.17/100 against declared priors;
- hidden-test PR-AUC: 0.999490;
- hidden-test F1: 0.978512;
- novel-vector recall: 0.937500;
- hard false-positive rate: 0.000000.

These high scores are evidence that the detector separates the current synthetic population. They are not a live-network performance guarantee.

## Independent external check

The repository also evaluates the detector methodology on later rows from the public, real, anonymized ULB/Worldline card dataset. That check uses chronological train/validation/test partitions, removes exact duplicates before splitting, freezes model/calibration/threshold choices before opening the future test, and reports uncertainty and temporal drift.

The current external result is PR-AUC 0.7656, precision 0.8727, recall 0.6486, F1 0.7442 and a 0.0124% false-positive rate. It is materially lower than the synthetic result and therefore provides a more credible boundary for claims. It validates the scalable detector method only; the dataset has no lifecycle, entity, channel, action or GenAI-attack semantics.

See [the independent external validation report](external-validation-report.md) for the data-quality reconciliation, confidence intervals, calibration checks, drift findings and next gates. Privacy-approved institution shadow scoring remains the required production validation.
