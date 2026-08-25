"""Independent validation on the public anonymized ULB/Worldline fraud dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ULB_OPENML_ID = 1597
ULB_SOURCE_URL = "https://www.openml.org/d/1597"
ULB_API_URL = "https://www.openml.org/api/v1/json/data/1597"
ULB_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
ULB_EXPECTED_ROWS = 284_807
ULB_EXPECTED_FRAUDS = 492
PCA_FEATURES = [f"V{index}" for index in range(1, 29)]
EXTERNAL_MODEL_FEATURES = [*PCA_FEATURES, "amount_log", "hour_sin", "hour_cos"]


@dataclass(frozen=True)
class ExternalValidationConfig:
    input_path: Path
    output_dir: Path = Path("data/external_validation")
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    validation_fpr_cap: float = 0.002
    seed: int = 20260824

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"External validation dataset not found: {self.input_path}")
        if self.input_path.suffix.lower() not in {".parquet", ".pq", ".arff", ".csv"}:
            raise ValueError("External dataset must be Parquet, ARFF, or CSV.")
        if not 0.05 <= self.validation_fraction <= 0.30:
            raise ValueError("validation_fraction must be between 0.05 and 0.30.")
        if not 0.05 <= self.test_fraction <= 0.30:
            raise ValueError("test_fraction must be between 0.05 and 0.30.")
        if self.validation_fraction + self.test_fraction >= 0.60:
            raise ValueError("At least 40% of chronological rows must remain for training.")
        if not 0 < self.validation_fpr_cap <= 0.02:
            raise ValueError("validation_fpr_cap must be between 0 and 0.02.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ulb_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path)
        except ImportError as error:
            raise RuntimeError(
                "Parquet validation requires pyarrow. Install requirements-validation.txt."
            ) from error
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        from scipy.io import arff

        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
    frame.columns = [str(column) for column in frame.columns]
    for column in [*PCA_FEATURES, "Time", "Amount", "Class"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def assess_external_data_quality(
    frame: pd.DataFrame, *, source_path: Path | None = None
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: str, severity: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "severity": severity,
            }
        )

    required = {"Time", "Amount", "Class", *PCA_FEATURES}
    missing = sorted(required.difference(frame.columns))
    add("required_schema", not missing, missing, "Time, Amount, Class and V1-V28", "critical")
    if missing:
        return {
            "status": "failed",
            "score": 0.0,
            "checks": checks,
            "failure_count": 1,
            "critical_failure_count": 1,
        }

    exact_duplicates = int(frame.duplicated().sum())
    add("exact_row_duplicates", exact_duplicates == 0, exact_duplicates, "0", "high")
    missing_values = int(frame[list(required)].isna().sum().sum())
    add("required_value_completeness", missing_values == 0, missing_values, "0 null values", "critical")
    finite_failures = int(
        (~np.isfinite(frame[[*PCA_FEATURES, "Time", "Amount"]].to_numpy(dtype=float))).sum()
    )
    add("numeric_finiteness", finite_failures == 0, finite_failures, "0 non-finite values", "critical")
    label_values = sorted(frame["Class"].dropna().astype(int).unique().tolist())
    add("label_domain", label_values == [0, 1], label_values, "[0, 1]", "critical")
    negative_amounts = int(frame["Amount"].lt(0).sum())
    add("amount_domain", negative_amounts == 0, negative_amounts, "0 negative amounts", "high")
    invalid_time = int(frame["Time"].lt(0).sum())
    add("elapsed_time_domain", invalid_time == 0, invalid_time, "0 negative elapsed times", "high")
    row_count = int(len(frame))
    fraud_count = int(frame["Class"].sum())
    add(
        "published_row_count_reconciliation",
        row_count == ULB_EXPECTED_ROWS,
        row_count,
        str(ULB_EXPECTED_ROWS),
        "high",
    )
    add(
        "published_fraud_count_reconciliation",
        fraud_count == ULB_EXPECTED_FRAUDS,
        fraud_count,
        str(ULB_EXPECTED_FRAUDS),
        "high",
    )
    chronological = bool(frame["Time"].is_monotonic_increasing)
    add(
        "source_chronological_order",
        chronological,
        chronological,
        "non-decreasing Time",
        "medium",
    )

    weights = {"critical": 5, "high": 3, "medium": 2, "low": 1}
    total_weight = sum(weights[item["severity"]] for item in checks)
    passed_weight = sum(weights[item["severity"]] for item in checks if item["passed"])
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "score": round(100.0 * passed_weight / total_weight, 2),
        "dataset_grain": "one anonymized card transaction per row",
        "row_count": row_count,
        "column_count": int(len(frame.columns)),
        "fraud_count": fraud_count,
        "fraud_prevalence": round(float(fraud_count / row_count), 8),
        "time_min_seconds": round(float(frame["Time"].min()), 3),
        "time_max_seconds": round(float(frame["Time"].max()), 3),
        "amount_min": round(float(frame["Amount"].min()), 4),
        "amount_max": round(float(frame["Amount"].max()), 4),
        "source_sha256": _sha256(source_path) if source_path else None,
        "checks": checks,
        "failure_count": len(failures),
        "critical_failure_count": sum(
            item["severity"] == "critical" for item in failures
        ),
    }


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["amount_log"] = np.log1p(result["Amount"].clip(lower=0))
    hour = (result["Time"] % 86_400) / 3_600.0
    result["hour_sin"] = np.sin(hour * 2 * math.pi / 24)
    result["hour_cos"] = np.cos(hour * 2 * math.pi / 24)
    return result


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-8, 1 - 1e-8)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def chronological_splits(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * (1.0 - validation_fraction - test_fraction))
    validation_end = int(len(ordered) * (1.0 - test_fraction))
    partitions = {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
    if any(partition["Class"].nunique() < 2 for partition in partitions.values()):
        raise ValueError("Every chronological partition must contain fraud and legitimate rows.")
    return partitions


def _select_threshold_under_fpr_cap(
    labels: np.ndarray, probabilities: np.ndarray, fpr_cap: float
) -> dict[str, float]:
    labels = labels.astype(bool)
    order = np.argsort(-probabilities, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = probabilities[order]
    tp = np.cumsum(ordered_labels)
    fp = np.cumsum(~ordered_labels)
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((~labels).sum()))
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    fpr = fp / negatives
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    eligible = np.flatnonzero(fpr <= fpr_cap)
    candidate_indexes = eligible if len(eligible) else np.arange(len(labels))
    best = int(candidate_indexes[np.argmax(f1[candidate_indexes])])
    return {
        "threshold": float(ordered_scores[best]),
        "validation_precision": float(precision[best]),
        "validation_recall": float(recall[best]),
        "validation_f1": float(f1[best]),
        "validation_false_positive_rate": float(fpr[best]),
    }


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _bootstrap_pr_auc(
    labels: np.ndarray, probabilities: np.ndarray, *, seed: int, repeats: int = 200
) -> list[float]:
    rng = np.random.default_rng(seed)
    positive_scores = probabilities[labels.astype(bool)]
    negative_scores = probabilities[~labels.astype(bool)]
    estimates: list[float] = []
    for _ in range(repeats):
        sampled_positive = rng.choice(positive_scores, size=len(positive_scores), replace=True)
        sampled_negative = rng.choice(negative_scores, size=len(negative_scores), replace=True)
        sampled_scores = np.concatenate([sampled_positive, sampled_negative])
        sampled_labels = np.concatenate(
            [np.ones(len(sampled_positive)), np.zeros(len(sampled_negative))]
        )
        estimates.append(float(average_precision_score(sampled_labels, sampled_scores)))
    return [round(float(value), 6) for value in np.quantile(estimates, [0.025, 0.975])]


def _top_fraction_capture(
    labels: np.ndarray, probabilities: np.ndarray, fractions: tuple[float, ...] = (0.001, 0.005, 0.01)
) -> dict[str, Any]:
    order = np.argsort(-probabilities, kind="stable")
    total_fraud = max(1, int(labels.sum()))
    result: dict[str, Any] = {}
    for fraction in fractions:
        count = max(1, int(math.ceil(len(labels) * fraction)))
        captured = int(labels[order[:count]].sum())
        result[f"top_{fraction * 100:g}_percent"] = {
            "review_count": count,
            "fraud_captured": captured,
            "recall": round(captured / total_fraud, 6),
        }
    return result


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Return equal-frequency ECE so rare-event bins remain populated."""
    order = np.argsort(probabilities, kind="stable")
    total = max(1, len(labels))
    error = 0.0
    for indexes in np.array_split(order, bins):
        if not len(indexes):
            continue
        confidence = float(np.mean(probabilities[indexes]))
        observed_rate = float(np.mean(labels[indexes]))
        error += (len(indexes) / total) * abs(confidence - observed_rate)
    return error


def _external_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    *,
    seed: int,
    include_bootstrap: bool = False,
) -> dict[str, Any]:
    labels = labels.astype(bool)
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    total_fraud_amount = float(amounts[labels].sum())
    detected_fraud_amount = float(amounts[labels & predicted].sum())
    prevalence = float(labels.mean())
    null_brier_score = prevalence * (1.0 - prevalence)
    model_brier_score = float(brier_score_loss(labels, probabilities))
    metrics = {
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 6),
        "false_positive_rate": round(float(fp / max(1, fp + tn)), 6),
        "brier_score": round(model_brier_score, 8),
        "null_brier_score": round(null_brier_score, 8),
        "brier_skill_score": round(
            1.0 - (model_brier_score / null_brier_score)
            if null_brier_score
            else 0.0,
            6,
        ),
        "expected_calibration_error": round(
            _expected_calibration_error(labels, probabilities), 8
        ),
        "value_weighted_recall": round(
            detected_fraud_amount / total_fraud_amount if total_fraud_amount else 0.0,
            6,
        ),
        "event_count": int(len(labels)),
        "fraud_event_count": int(labels.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_95ci": _wilson_interval(int(tp), int(tp + fp)),
        "recall_95ci": _wilson_interval(int(tp), int(tp + fn)),
        "false_positive_rate_95ci": _wilson_interval(int(fp), int(fp + tn)),
        "review_capacity": _top_fraction_capture(labels, probabilities),
    }
    if include_bootstrap:
        metrics["pr_auc_95ci"] = _bootstrap_pr_auc(
            labels, probabilities, seed=seed
        )
    return metrics


def _population_stability_index(reference: pd.Series, observed: pd.Series) -> float:
    reference_values = reference.to_numpy(dtype=float)
    observed_values = observed.to_numpy(dtype=float)
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    observed_counts, _ = np.histogram(observed_values, bins=edges)
    reference_share = np.clip(reference_counts / max(1, reference_counts.sum()), 1e-6, None)
    observed_share = np.clip(observed_counts / max(1, observed_counts.sum()), 1e-6, None)
    return round(float(np.sum((observed_share - reference_share) * np.log(observed_share / reference_share))), 6)


def _amount_shape_profile(values: pd.Series) -> dict[str, Any]:
    positive = values[values.gt(0)].astype(float)
    median = max(float(positive.median()), 1e-9)
    quantiles = positive.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {
        "positive_count": int(len(positive)),
        "zero_rate": round(float(values.eq(0).mean()), 6),
        "median": round(median, 6),
        "normalized_quantiles": {
            str(quantile): round(float(value / median), 6)
            for quantile, value in quantiles.items()
        },
    }


def _synthetic_calibration(
    external: pd.DataFrame, synthetic_path: Path | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_run",
        "comparison_scope": "currency-normalized amount shape and class prevalence only",
        "limitations": [
            "ULB features V1-V28 are anonymized PCA components and cannot be semantically mapped to SentinelLoop lifecycle telemetry.",
            "ULB contains card transactions from two days in 2013; SentinelLoop spans multiple rails and lifecycle phases.",
        ],
    }
    if synthetic_path is None or not synthetic_path.exists():
        return result
    synthetic = pd.read_csv(synthetic_path, low_memory=False)
    card = synthetic[
        synthetic["payment_rail"].eq("CARD") & synthetic["amount_inr"].ge(0)
    ].copy()
    external_profile = _amount_shape_profile(external["Amount"])
    synthetic_profile = _amount_shape_profile(card["amount_inr"])
    external_shape = external_profile["normalized_quantiles"]
    synthetic_shape = synthetic_profile["normalized_quantiles"]
    mean_log_gap = float(
        np.mean(
            [
                abs(math.log(max(float(synthetic_shape[key]), 1e-9) / max(float(value), 1e-9)))
                for key, value in external_shape.items()
            ]
        )
    )
    external_prevalence = float(external["Class"].mean())
    synthetic_prevalence = float(card["label_fraud"].astype(bool).mean())
    result.update(
        {
            "status": "completed",
            "external_amount_profile": external_profile,
            "synthetic_card_amount_profile": synthetic_profile,
            "currency_normalized_amount_shape_score": round(math.exp(-mean_log_gap), 6),
            "external_fraud_prevalence": round(external_prevalence, 8),
            "synthetic_card_attack_prevalence": round(synthetic_prevalence, 8),
            "prevalence_multiplier": round(
                synthetic_prevalence / external_prevalence
                if external_prevalence
                else 0.0,
                2,
            ),
            "interpretation": (
                "The synthetic population intentionally oversamples attacks for stress testing. "
                "Production probability calibration must use institution-specific base rates."
            ),
        }
    )
    return result


def run_external_validation(
    config: ExternalValidationConfig,
    *,
    synthetic_population_path: Path | None = Path("data/benchmark/population_events.csv"),
) -> tuple[dict[str, Any], pd.DataFrame]:
    config.validate()
    raw = load_ulb_dataset(config.input_path)
    quality = assess_external_data_quality(raw, source_path=config.input_path)
    if quality["critical_failure_count"]:
        raise ValueError("External validation stopped on a critical data-quality failure.")
    duplicates_removed = int(raw.duplicated().sum())
    modeled = raw.drop_duplicates(keep="first").reset_index(drop=True)
    quality["model_input_status"] = (
        "passed_after_conservative_deduplication"
        if duplicates_removed
        else "passed_without_remediation"
    )
    quality["remediation"] = {
        "exact_duplicate_rows_removed": duplicates_removed,
        "raw_row_count": int(len(raw)),
        "modeled_row_count": int(len(modeled)),
        "raw_fraud_count": int(raw["Class"].sum()),
        "modeled_fraud_count": int(modeled["Class"].sum()),
        "rationale": (
            "Exact duplicates are removed conservatively before temporal splitting to avoid "
            "optimistic validation. The source has no transaction identifier, so duplicated "
            "observations cannot be proven to be distinct payments."
        ),
    }
    featured = _build_features(modeled)
    partitions = chronological_splits(
        featured,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train, validation, test = (
        partitions["train"],
        partitions["validation"],
        partitions["test"],
    )
    candidates: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1200,
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=config.seed,
        ),
    }
    calibrators: dict[str, LogisticRegression] = {}
    candidate_results: dict[str, Any] = {}
    for name, candidate in candidates.items():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Could not find the number of physical cores.*",
                category=UserWarning,
            )
            if name == "hist_gradient_boosting":
                weights = compute_sample_weight("balanced", train["Class"].astype(bool))
                candidate.fit(
                    train[EXTERNAL_MODEL_FEATURES],
                    train["Class"],
                    sample_weight=weights,
                )
            else:
                candidate.fit(train[EXTERNAL_MODEL_FEATURES], train["Class"])
        raw_validation_scores = candidate.predict_proba(
            validation[EXTERNAL_MODEL_FEATURES]
        )[:, 1]
        calibrator = LogisticRegression(max_iter=1000, random_state=config.seed)
        calibrator.fit(_logit(raw_validation_scores), validation["Class"])
        calibrators[name] = calibrator
        validation_scores = calibrator.predict_proba(
            _logit(raw_validation_scores)
        )[:, 1]
        threshold_policy = _select_threshold_under_fpr_cap(
            validation["Class"].to_numpy(),
            validation_scores,
            config.validation_fpr_cap,
        )
        candidate_results[name] = {
            "validation_pr_auc": round(
                float(average_precision_score(validation["Class"], validation_scores)), 6
            ),
            "validation_roc_auc": round(
                float(roc_auc_score(validation["Class"], validation_scores)), 6
            ),
            "validation_brier_score_after_platt": round(
                float(brier_score_loss(validation["Class"], validation_scores)), 8
            ),
            **{key: round(value, 6) for key, value in threshold_policy.items()},
        }
    selected_name = max(
        candidate_results,
        key=lambda name: candidate_results[name]["validation_pr_auc"],
    )
    selected_model = candidates[selected_name]
    selected_calibrator = calibrators[selected_name]
    threshold = float(candidate_results[selected_name]["threshold"])
    raw_test_scores = selected_model.predict_proba(test[EXTERNAL_MODEL_FEATURES])[:, 1]
    test_scores = selected_calibrator.predict_proba(_logit(raw_test_scores))[:, 1]
    test_labels = test["Class"].to_numpy(dtype=bool)
    test_amounts = test["Amount"].to_numpy(dtype=float)
    test_metrics = _external_metrics(
        test_labels,
        test_scores,
        threshold,
        test_amounts,
        seed=config.seed,
        include_bootstrap=True,
    )

    midpoint = len(test) // 2
    temporal_slices: dict[str, Any] = {}
    for name, indexes in {
        "early_test": np.arange(0, midpoint),
        "late_test": np.arange(midpoint, len(test)),
    }.items():
        temporal_slices[name] = _external_metrics(
            test_labels[indexes],
            test_scores[indexes],
            threshold,
            test_amounts[indexes],
            seed=config.seed,
        )

    drift_features = [*PCA_FEATURES, "Amount"]
    psi_by_feature = {
        feature: _population_stability_index(train[feature], test[feature])
        for feature in drift_features
    }
    worst_drift = sorted(psi_by_feature.items(), key=lambda item: item[1], reverse=True)
    predictions = pd.DataFrame(
        {
            "external_row_id": test.index.astype(int),
            "time_seconds": test["Time"].to_numpy(),
            "amount": test_amounts,
            "label_fraud": test_labels,
            "risk_score": test_scores,
            "predicted_fraud": test_scores >= threshold,
        }
    )
    split_summary = {
        name: {
            "rows": int(len(partition)),
            "frauds": int(partition["Class"].sum()),
            "fraud_prevalence": round(float(partition["Class"].mean()), 8),
            "time_min_seconds": round(float(partition["Time"].min()), 3),
            "time_max_seconds": round(float(partition["Time"].max()), 3),
        }
        for name, partition in partitions.items()
    }
    external_fpr_supported = test_metrics["false_positive_rate"] <= 0.003
    report = {
        "validation_version": "external-ulb-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_assessment": "share_with_caveats",
        "dataset": {
            "name": "ULB/Worldline Credit Card Fraud",
            "openml_dataset_id": ULB_OPENML_ID,
            "source_url": ULB_SOURCE_URL,
            "metadata_api_url": ULB_API_URL,
            "download_url": ULB_PARQUET_URL,
            "source_type": "external_real_anonymized_card_transactions",
            "collection_window": "two days in September 2013",
            "geography": "European cardholders",
            "privacy": "V1-V28 are PCA-anonymized; Time and Amount remain interpretable",
            "license_metadata": "Public (as declared by OpenML dataset 1597)",
            "local_file_name": config.input_path.name,
            "local_file_sha256": quality["source_sha256"],
            "split_summary": split_summary,
        },
        "data_quality": quality,
        "methodology": {
            "split_policy": "chronological 60% train / 20% validation / 20% test; no random shuffling",
            "model_selection": "highest validation PR-AUC only",
            "threshold_selection": (
                "highest validation F1 among thresholds satisfying the validation false-positive cap"
            ),
            "score_calibration": "Platt scaling fit on validation model scores; test labels remain sealed",
            "validation_false_positive_cap": config.validation_fpr_cap,
            "test_opened_after_selection": True,
            "features": EXTERNAL_MODEL_FEATURES,
            "forbidden_features": ["Class", "test labels", "future rows"],
            "currency": "EUR-equivalent source Amount field; no conversion applied",
        },
        "defense": {
            "selected_model": selected_name,
            "risk_score_semantics": "validation-calibrated risk estimate; institution recalibration still required",
            "candidate_models": candidate_results,
            "threshold_selected_on_validation": round(threshold, 8),
            "test_metrics": test_metrics,
            "temporal_slices": temporal_slices,
        },
        "robustness": {
            "population_stability_index_by_feature": psi_by_feature,
            "worst_drift_features": [
                {"feature": feature, "psi": value} for feature, value in worst_drift[:8]
            ],
            "psi_interpretation": {
                "below_0.10": "small shift",
                "0.10_to_0.25": "moderate shift requiring review",
                "above_0.25": "large shift; recalibration or retraining likely required",
            },
        },
        "synthetic_calibration": _synthetic_calibration(modeled, synthetic_population_path),
        "claim_validation": [
            {
                "claim": "The detection pipeline works under real rare-event class imbalance.",
                "status": "supported" if test_metrics["pr_auc"] > quality["fraud_prevalence"] else "not_supported",
                "evidence": f"Test PR-AUC {test_metrics['pr_auc']} versus prevalence {quality['fraud_prevalence']}.",
            },
            {
                "claim": "The threshold controls legitimate-payment false positives on future data.",
                "status": "supported" if external_fpr_supported else "needs_revision",
                "evidence": f"Chronological test FPR {test_metrics['false_positive_rate']} with 95% CI {test_metrics['false_positive_rate_95ci']}.",
            },
            {
                "claim": "Synthetic amount behavior is calibrated to external card-transaction shape.",
                "status": "partial",
                "evidence": "Only currency-normalized amount quantiles are comparable; rail, geography and source period differ.",
            },
            {
                "claim": "Red/Blue lifecycle-agent efficacy generalizes to live payments.",
                "status": "not_testable_with_this_dataset",
                "evidence": "ULB provides no entity, channel, lifecycle, action or GenAI-attack semantics.",
            },
        ],
        "limitations": [
            "This external test validates the scalable detector methodology, not the Qwen agent's lifecycle reasoning.",
            "The data covers one card dataset and two days from 2013; it does not represent UPI, bank transfer, wallets, payouts or agentic commerce.",
            "PCA anonymization prevents semantic mapping of V1-V28 to SentinelLoop evidence tools and mitigations.",
            "No customer, merchant, device or account identifiers are available for entity-isolated splitting or graph validation.",
            "The test set has few fraud cases; confidence intervals must accompany point estimates.",
            "Institution-specific shadow scoring remains required before any live decisioning claim.",
        ],
        "next_validation_gates": [
            "Run the IEEE-CIS adapter after accepting Kaggle terms to validate device, identity and transaction context.",
            "Run the BAF adapter to validate pre-transaction onboarding drift and subgroup fairness.",
            "Obtain privacy-approved aggregate institution priors for rail, channel, amount, velocity and lifecycle calibration.",
            "Shadow-score live events with no customer action, then measure delayed labels, drift, calibration and operational latency.",
        ],
    }
    return report, predictions


def write_external_validation_artifacts(
    config: ExternalValidationConfig,
    report: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (config.output_dir / "data_quality.json").write_text(
        json.dumps(report["data_quality"], indent=2), encoding="utf-8"
    )
    predictions.to_csv(config.output_dir / "test_predictions.csv", index=False)
