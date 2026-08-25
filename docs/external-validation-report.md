# Independent external validation report

**Decision: share with caveats.** SentinelLoop's scalable detection method shows meaningful performance on an independent, real, highly imbalanced card-payment dataset. The result supports the detector architecture and leakage controls; it does **not** establish that the Qwen Red/Blue lifecycle agents generalize to live payments.

Generated: 24 August 2026
Validation version: `external-ulb-v1`

## Executive readout

| Future-test result | Observed value |
|---|---:|
| Events / labeled frauds | 56,746 / 74 |
| PR-AUC | 0.7656 (95% bootstrap interval 0.6778–0.8566) |
| Precision | 0.8727 (95% Wilson interval 0.7598–0.9370) |
| Recall | 0.6486 (95% Wilson interval 0.5350–0.7476) |
| F1 | 0.7442 |
| False-positive rate | 0.0124% (95% Wilson interval 0.0060%–0.0255%) |
| Confusion matrix | 56,665 TN · 7 FP · 26 FN · 48 TP |
| Value-weighted recall | 0.4470 |
| Brier score / null Brier | 0.000526 / 0.001302 |
| Brier skill score | 0.5962 |
| Expected calibration error | 0.000163 |

At a review capacity of only 0.1% of future payments—57 cases—the ranking surfaced 50 of 74 frauds, or 67.6%. This is useful operational evidence, but the low count of fraud cases makes the uncertainty intervals essential.

## Source and reconciliation

The test uses the [ULB/Worldline Credit Card Fraud dataset on OpenML](https://www.openml.org/d/1597): 284,807 anonymized European card transactions collected over two days in September 2013, including 492 labeled frauds. V1–V28 are PCA-anonymized; only elapsed time and amount remain interpretable.

The local source matched the published totals exactly and its SHA-256 was recorded as:

`b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d`

The quality audit found 1,081 exact duplicate rows, including 19 duplicate fraud rows. Because the source has no transaction identifier, the duplicates cannot be proven to be distinct payments. They were conservatively removed **before** splitting to prevent optimistic leakage. The model therefore used 283,726 rows and 473 frauds. Raw quality scored 91.89/100; the modeled input passed after this documented remediation.

## Leakage and selection controls

1. Data is ordered by elapsed transaction time.
2. The earliest 60% is used for training, the next 20% for validation, and the latest 20% for the final test.
3. Two candidate detectors are trained: balanced logistic regression and balanced histogram gradient boosting.
4. The detector is selected by validation PR-AUC only. Histogram gradient boosting won 0.7605 versus 0.7532.
5. Platt calibration and the decision threshold are fitted on validation scores only.
6. The threshold maximizes validation F1 while staying under a 0.2% validation false-positive cap.
7. Future test labels are opened only after the model, calibrator, and threshold are frozen.

No label, future row, test statistic, attack-family field, or scenario identifier is used as a detector input.

## Robustness findings

Performance declined within the future window:

| Temporal slice | PR-AUC | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Earlier half | 0.8349 | 0.9730 | 0.6923 | 0.8090 | 0.0035% |
| Later half | 0.6315 | 0.6667 | 0.5455 | 0.6000 | 0.0212% |

This is a material drift signal. Train-to-test population stability was largest for V1 (PSI 1.46), V3 (1.07), V28 (0.72), and V11 (0.45). Because these variables are anonymized, the shift cannot be diagnosed semantically. The production design therefore needs time-window monitoring, recalibration, and a retraining trigger rather than a one-time model release.

## Synthetic-to-real calibration check

Only amount-shape and class prevalence can be compared responsibly. The currency-normalized positive-amount shape score was 0.7037. The synthetic card population's attack prevalence was 20.99% versus 0.1667% after external deduplication—a 125.9× multiplier. That oversampling is appropriate for a stress-test arena, but its risk scores must never be described as production probabilities without institution-specific base-rate calibration.

Synthetic lifecycle events also include many non-value events, producing a 46.3% zero-amount rate versus 0.64% in the transaction-only external data. That difference is expected from the grain mismatch and should not be tuned away blindly.

## Claim-by-claim conclusion

| Claim | Status | Reason |
|---|---|---|
| The detection pipeline can operate under real rare-event imbalance | Supported | Future PR-AUC 0.7656 versus a roughly 0.13% future-test fraud rate |
| A validation-selected threshold can keep future legitimate-payment false positives low | Supported with uncertainty | 7 false positives among 56,672 legitimate future payments; 95% FPR interval reported |
| Synthetic card amounts resemble the external distribution | Partial | Normalized shape is comparable; currencies, period, lifecycle grain, and rails differ |
| Red/Blue GenAI lifecycle reasoning generalizes to live payments | Not testable here | External data has no entity, channel, lifecycle, action, or GenAI-attack semantics |
| The solution is ready for autonomous live blocking | Not supported | Privacy-reviewed institution shadow scoring and operational validation remain mandatory |

## Validation confidence

**Moderate for the narrow detector claim; low for full-system generalization.** The evaluation is reproducible, chronologically separated, independently sourced, uncertainty-bounded, and protected against exact-duplicate leakage. Confidence is reduced by the two-day 2013 collection window, only 74 frauds in the final test, PCA-anonymized variables, lack of entity identifiers, and coverage of only one card dataset.

## Next major gates

1. Add the official [Feedzai Bank Account Fraud dataset](https://github.com/feedzai/bank-account-fraud) for pre-transaction onboarding, temporal drift, and subgroup fairness.
2. Add IEEE-CIS only after accepting its Kaggle competition terms, to test device, identity, and transaction context.
3. Obtain privacy-approved aggregate institution priors for rail, channel, amount, velocity, and lifecycle calibration.
4. Shadow-score live events with customer action disabled; measure delayed-label recall, calibration, latency, drift, and subgroup harm.
5. Run adversarial Qwen-vs-Qwen lifecycle evaluations on an institution-reviewed semantic event schema.

## Reproduce

Install the optional validation dependencies, download the source file from the OpenML record, and run:

```bash
python -m sentinelloop external-validate \
  --input data/external/creditcard.parquet \
  --output-dir data/external_validation \
  --fpr-cap 0.002 \
  --seed 20260824
```

The machine-readable result is `data/external_validation/latest.json`. Raw external data and row-level predictions are deliberately excluded from version control.
