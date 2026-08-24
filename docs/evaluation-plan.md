# Evaluation plan

## Primary metrics

| Dimension | Metric | Initial target |
|---|---|---:|
| Transaction detection | PR-AUC | Reported on withheld campaigns; optimize over baseline |
| Transaction detection | Recall at fixed precision | Recall at 90% precision |
| Operational safety | False-positive rate | Under 2% on legitimate events |
| Ring detection | Fraud-ring account recall | Report on withheld mule campaigns |
| Mitigation | Fraud value prevented | Reported with action-policy assumptions |
| Simulation fidelity | Distributional similarity | Compare normal-event distributions against defined behavioral priors |
| Adaptation | Variant recall lift | Compare before/after retraining on generated hard variants |

## Evaluation protocol

1. Generate distinct simulation runs for train, validation, and test periods.
2. Hold out entire campaign IDs and at least one parameter region for each MVP attack family.
3. Fit preprocessing, feature aggregation, and models on training data only.
4. Choose thresholds on validation data, then report test metrics once.
5. Slice results by rail, channel, new/known device, attack family, and legitimate near-neighbor controls.
6. Document every simulator seed and model configuration for reproducibility.

## Baselines

- Rules: new device + new beneficiary + high amount/velocity.
- Tabular model: gradient-boosted or random-forest classifier over allowed point-in-time features.
- Graph heuristic: beneficiary fan-in / shared-infrastructure threshold.
- Combined model: calibrated transaction score plus ring-risk enrichment.

## Demo-ready result format

Every experiment should save: run ID, random seed, dataset version, feature version, model configuration, metric table, confusion matrix, and threshold. The UI presents concise metrics and a clearly labeled synthetic-data disclaimer.
