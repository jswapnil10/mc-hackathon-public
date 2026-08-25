"""Sealed population benchmark for synthetic fidelity and Blue defense efficacy."""

from __future__ import annotations

import math
import os
import warnings
from datetime import datetime, timezone
from typing import Any

# Some constrained runtimes do not expose the macOS physical-core query used by
# joblib. A logical-core fallback keeps local and container runs quiet and stable.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .population import CHANNEL_PRIORS, RAIL_PRIORS, PopulationConfig, generate_population_dataset
from .threat_atlas import ThreatAtlas


NUMERIC_FEATURES = [
    "amount_log",
    "baseline_amount_log",
    "amount_to_baseline",
    "session_age_log",
    "account_age_log",
    "beneficiary_age_log",
    "event_hour_sin",
    "event_hour_cos",
    "sequence",
    "prior_case_event_count",
    "payment_count_log",
    "urgency_level",
    "channel_switch_count",
    "observable_signal_count",
    "device_is_new",
    "network_is_new",
    "behavior_automation_suspected",
    "identity_consistency_mismatch",
    "evidence_conflict_present",
    "trust_failure_count",
    "independent_verification_count",
    "risk_signal_density",
    "source_reference_count",
]
CATEGORICAL_FEATURES = [
    "event_type",
    "lifecycle_phase",
    "payment_rail",
    "channel",
    "auth_method",
    "merchant_category",
    "source_system",
    "decision_lane",
]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
FORBIDDEN_FEATURES = {
    "label_fraud",
    "attack_family",
    "attack_vector_id",
    "difficulty",
    "scenario_id",
    "case_id",
    "case_role",
    "split",
    "event_id",
    "value_at_risk_inr",
    "legitimate_control",
    "near_neighbor_family",
}
REQUIRED_COLUMNS = {
    "event_id",
    "event_ts",
    "case_id",
    "sequence",
    "split",
    "case_role",
    "label_fraud",
    "attack_family",
    "attack_vector_id",
    "amount_inr",
    "payment_rail",
    "channel",
    "sender_account_id",
    "beneficiary_id",
}


def build_benchmark_features(events: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Population dataset is missing required fields: {sorted(missing)}")
    if FORBIDDEN_FEATURES.intersection(MODEL_FEATURES):
        raise AssertionError("A sealed truth or identifier entered MODEL_FEATURES.")
    frame = events.copy()
    frame["event_ts"] = pd.to_datetime(frame["event_ts"], utc=True)
    frame["amount_log"] = frame["amount_inr"].clip(lower=0).map(math.log1p)
    frame["baseline_amount_log"] = frame["sender_baseline_amount_inr"].clip(lower=0).map(
        math.log1p
    )
    frame["amount_to_baseline"] = frame["amount_to_baseline"].clip(lower=0, upper=50)
    frame["session_age_log"] = frame["session_age_seconds"].clip(lower=0).map(math.log1p)
    frame["account_age_log"] = frame["account_age_days"].clip(lower=0).map(math.log1p)
    frame["beneficiary_age_log"] = frame["beneficiary_age_days"].clip(lower=0).map(
        math.log1p
    )
    frame["payment_count_log"] = frame["payment_count"].clip(lower=0).map(math.log1p)
    hour = frame["event_ts"].dt.hour + frame["event_ts"].dt.minute / 60.0
    frame["event_hour_sin"] = (hour * 2.0 * math.pi / 24.0).map(math.sin)
    frame["event_hour_cos"] = (hour * 2.0 * math.pi / 24.0).map(math.cos)
    frame["prior_case_event_count"] = frame.groupby("case_id", sort=False).cumcount()
    return frame


def _choose_threshold(labels: pd.Series, probabilities: Any) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    preferred: list[tuple[float, float, float]] = []
    fallback: list[tuple[float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        p, r = float(precision[index]), float(recall[index])
        score = 2.0 * p * r / (p + r) if p + r else 0.0
        candidate = (score, r, float(threshold))
        fallback.append(candidate)
        if p >= 0.90:
            preferred.append(candidate)
    return max(preferred or fallback)[2]


def _safe_auc(metric: Any, labels: pd.Series, probabilities: Any) -> float | None:
    if labels.nunique() < 2:
        return None
    return round(float(metric(labels, probabilities)), 6)


def _classification_metrics(
    labels: pd.Series,
    probabilities: Any,
    threshold: float,
    *,
    values: pd.Series | None = None,
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    label_array = labels.to_numpy(dtype=bool)
    predicted_array = predicted.to_numpy(dtype=bool) if hasattr(predicted, "to_numpy") else predicted
    value_array = values.to_numpy(dtype=float) if values is not None else None
    fraud_values = value_array[label_array] if value_array is not None else None
    detected_values = (
        value_array[label_array & predicted_array].sum()
        if value_array is not None and label_array.any()
        else 0.0
    )
    total_values = fraud_values.sum() if fraud_values is not None else 0.0
    return {
        "pr_auc": _safe_auc(average_precision_score, labels, probabilities),
        "roc_auc": _safe_auc(roc_auc_score, labels, probabilities),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 6),
        "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 6),
        "value_weighted_recall": round(
            float(detected_values / total_values) if total_values else 0.0, 6
        ),
        "event_count": int(len(labels)),
        "fraud_event_count": int(labels.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _logistic_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1200, random_state=42
                ),
            ),
        ]
    )


def _gradient_pipeline(seed: int) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.07,
                    max_iter=140,
                    max_leaf_nodes=31,
                    min_samples_leaf=15,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _mitigation_action(score: float, threshold: float) -> str:
    if score < threshold * 0.45:
        return "allow"
    if score < threshold:
        return "monitor"
    if score < min(0.82, threshold + 0.14):
        return "step_up"
    if score < 0.95:
        return "hold"
    return "block"


def _js_similarity(observed: pd.Series, target: dict[str, float]) -> float:
    counts = observed.value_counts(normalize=True).to_dict()
    keys = sorted(set(counts).union(target))
    p = [float(counts.get(key, 0.0)) for key in keys]
    q = [float(target.get(key, 0.0)) for key in keys]
    p_total, q_total = sum(p), sum(q)
    p = [value / p_total for value in p]
    q = [value / q_total for value in q]
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q)]

    def kl(left: list[float], right: list[float]) -> float:
        return sum(
            value * math.log(value / comparison)
            for value, comparison in zip(left, right)
            if value > 0 and comparison > 0
        )

    divergence = 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)
    return round(max(0.0, 1.0 - divergence / math.log(2.0)), 6)


def assess_fidelity(events: pd.DataFrame, atlas: ThreatAtlas) -> dict[str, Any]:
    ordinary = events[events["case_role"].eq("ordinary_legitimate")].copy()
    normalized_channel = ordinary["channel"].replace(
        {"QR": "MOBILE_APP", "ECOMMERCE": "WEB", "ERP": "API"}
    )
    rail_similarity = _js_similarity(ordinary["payment_rail"], RAIL_PRIORS)
    channel_similarity = _js_similarity(normalized_channel, CHANNEL_PRIORS)
    amount_ratio_median = float(
        ordinary.loc[ordinary["amount_inr"].gt(0), "amount_to_baseline"].median()
    )
    amount_calibration = max(
        0.0, 1.0 - abs(math.log(max(amount_ratio_median, 0.01))) / math.log(3.0)
    )
    amount_baseline_correlation = float(
        ordinary[["amount_inr", "sender_baseline_amount_inr"]].corr(method="spearman").iloc[0, 1]
    )
    observed_edge_rate = float(ordinary["legitimate_control"].notna().mean())
    edge_case_calibration = max(0.0, 1.0 - abs(observed_edge_rate - 0.12) / 0.12)
    expected_vectors = {vector.id for vector in atlas.vectors}
    represented_vectors = set(events["attack_vector_id"].dropna().unique())
    vector_coverage = len(represented_vectors.intersection(expected_vectors)) / len(expected_vectors)
    family_coverage = events["attack_family"].dropna().nunique() / len(
        {vector.simulation_family for vector in atlas.vectors}
    )
    sequence_validity = float(
        all(
            sorted(group["sequence"].astype(int).tolist()) == list(range(1, len(group) + 1))
            for _, group in events.groupby("case_id", sort=False)
        )
    )
    difficulty_means = (
        events.loc[events["label_fraud"]]
        .groupby("difficulty")["risk_signal_density"]
        .mean()
        .to_dict()
    )
    difficulty_gradient = float(
        difficulty_means.get("easy", 0.0) >= difficulty_means.get("medium", 0.0)
        and difficulty_means.get("medium", 0.0) >= difficulty_means.get("hard", 0.0)
    )
    fidelity_score = 100.0 * (
        0.16 * rail_similarity
        + 0.12 * channel_similarity
        + 0.12 * amount_calibration
        + 0.10 * max(0.0, min(1.0, amount_baseline_correlation))
        + 0.10 * edge_case_calibration
        + 0.16 * vector_coverage
        + 0.10 * family_coverage
        + 0.08 * sequence_validity
        + 0.06 * difficulty_gradient
    )
    return {
        "score": round(fidelity_score, 2),
        "reference_type": "declared_behavioral_priors_not_real_customer_data",
        "rail_distribution_similarity": rail_similarity,
        "channel_distribution_similarity": channel_similarity,
        "amount_to_customer_baseline_median": round(amount_ratio_median, 4),
        "amount_calibration_score": round(amount_calibration, 6),
        "amount_baseline_spearman": round(amount_baseline_correlation, 6),
        "legitimate_edge_case_rate": round(observed_edge_rate, 6),
        "edge_case_calibration_score": round(edge_case_calibration, 6),
        "attack_vector_coverage": round(vector_coverage, 6),
        "attack_family_coverage": round(family_coverage, 6),
        "sequence_validity_rate": sequence_validity,
        "difficulty_gradient_pass": bool(difficulty_gradient),
        "difficulty_risk_signal_means": {
            str(key): round(float(value), 6) for key, value in difficulty_means.items()
        },
        "limitations": [
            "No real customer or payment data is used.",
            "Distribution similarity is measured against explicit synthetic priors and must not be presented as empirical similarity to a live payment network.",
            "A production pilot should recalibrate priors using privacy-reviewed aggregate network statistics.",
        ],
    }


def assess_data_quality(events: pd.DataFrame, atlas: ThreatAtlas) -> dict[str, Any]:
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

    missing_required = sorted(REQUIRED_COLUMNS.difference(events.columns))
    add("required_schema", not missing_required, missing_required, "no missing fields", "critical")
    duplicate_ids = int(events["event_id"].duplicated().sum())
    add("event_id_uniqueness", duplicate_ids == 0, duplicate_ids, "0 duplicates", "critical")
    invalid_splits = sorted(set(events["split"]).difference({"train", "validation", "test_known", "test_novel"}))
    add("split_domain", not invalid_splits, invalid_splits, "only sealed split values", "critical")
    scenario_leakage = int(
        events.dropna(subset=["scenario_id"]).groupby("scenario_id")["split"].nunique().gt(1).sum()
    )
    add("scenario_split_isolation", scenario_leakage == 0, scenario_leakage, "0 leaking scenarios", "critical")
    entity_leakage = int(
        events.groupby("sender_account_id")["split"].nunique().gt(1).sum()
    )
    add("entity_split_isolation", entity_leakage == 0, entity_leakage, "0 leaking sender entities", "critical")
    attack_label_errors = int(
        events["label_fraud"]
        .loc[events["attack_family"].isna() | events["attack_vector_id"].isna()]
        .sum()
    )
    add("attack_label_integrity", attack_label_errors == 0, attack_label_errors, "0 attack rows without labels", "critical")
    legitimate_label_errors = int(
        ((~events["label_fraud"]) & events["attack_family"].notna()).sum()
    )
    add("legitimate_label_integrity", legitimate_label_errors == 0, legitimate_label_errors, "0 legitimate rows with attack labels", "critical")
    invalid_amounts = int(events["amount_inr"].lt(0).sum() + events["amount_inr"].gt(250_000).sum())
    add("amount_domain", invalid_amounts == 0, invalid_amounts, "INR 0 to 250000", "high")
    missing_core = int(events[["event_id", "event_ts", "case_id", "split"]].isna().any(axis=1).sum())
    add("core_completeness", missing_core == 0, missing_core, "0 incomplete rows", "critical")
    split_class_coverage = {
        split: int(group["label_fraud"].nunique())
        for split, group in events.groupby("split")
    }
    add(
        "class_coverage_by_split",
        all(value == 2 for value in split_class_coverage.values()),
        split_class_coverage,
        "both classes in every split",
        "high",
    )
    represented_vectors = set(events["attack_vector_id"].dropna())
    missing_vectors = sorted({vector.id for vector in atlas.vectors}.difference(represented_vectors))
    add("threat_vector_coverage", not missing_vectors, missing_vectors, "all atlas vectors represented", "high")
    known_vector_leakage = sorted(
        set(events.loc[events["split"].ne("test_novel"), "attack_vector_id"].dropna()).intersection(
            atlas.novel_holdout_vector_ids
        )
    )
    add("novel_vector_seal", not known_vector_leakage, known_vector_leakage, "0 novel vectors outside test_novel", "critical")
    weighted_severity = {"critical": 5, "high": 3, "medium": 2, "low": 1}
    total_weight = sum(weighted_severity[item["severity"]] for item in checks)
    passed_weight = sum(
        weighted_severity[item["severity"]] for item in checks if item["passed"]
    )
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "score": round(100.0 * passed_weight / total_weight, 2) if total_weight else 0.0,
        "grain": "one sanitized observable event per row",
        "row_count": int(len(events)),
        "column_count": int(len(events.columns)),
        "checks": checks,
        "failure_count": len(failures),
        "critical_failure_count": sum(
            item["severity"] == "critical" for item in failures
        ),
    }


def run_population_benchmark(
    config: PopulationConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    active_config = config or PopulationConfig()
    atlas = ThreatAtlas()
    events, metadata = generate_population_dataset(active_config, atlas=atlas)
    quality = assess_data_quality(events, atlas)
    if quality["critical_failure_count"]:
        raise ValueError("Population benchmark stopped because a critical data-quality check failed.")
    fidelity = assess_fidelity(events, atlas)
    featured = build_benchmark_features(events)
    partitions = {
        split: featured[featured["split"].eq(split)].copy()
        for split in ("train", "validation", "test_known", "test_novel")
    }
    train, validation = partitions["train"], partitions["validation"]
    models = {
        "logistic_regression": _logistic_pipeline(),
        "hist_gradient_boosting": _gradient_pipeline(active_config.seed),
    }
    model_results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        if name == "hist_gradient_boosting":
            weights = compute_sample_weight("balanced", train["label_fraud"])
            model.fit(
                train[MODEL_FEATURES],
                train["label_fraud"],
                classifier__sample_weight=weights,
            )
        else:
            model.fit(train[MODEL_FEATURES], train["label_fraud"])
        probabilities = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        threshold = _choose_threshold(validation["label_fraud"], probabilities)
        model_results[name] = {
            "validation_pr_auc": _safe_auc(
                average_precision_score, validation["label_fraud"], probabilities
            ),
            "validation_roc_auc": _safe_auc(
                roc_auc_score, validation["label_fraud"], probabilities
            ),
            "threshold": round(threshold, 6),
        }
    selected_name = max(
        model_results,
        key=lambda name: float(model_results[name]["validation_pr_auc"] or 0.0),
    )
    selected_model = models[selected_name]
    threshold = float(model_results[selected_name]["threshold"])
    scored_parts: list[pd.DataFrame] = []
    split_metrics: dict[str, Any] = {}
    for split in ("test_known", "test_novel"):
        partition = partitions[split]
        probabilities = selected_model.predict_proba(partition[MODEL_FEATURES])[:, 1]
        split_metrics[split] = _classification_metrics(
            partition["label_fraud"],
            probabilities,
            threshold,
            values=partition["value_at_risk_inr"],
        )
        scored = partition[
            [
                "event_id",
                "event_ts",
                "split",
                "case_role",
                "label_fraud",
                "attack_family",
                "attack_vector_id",
                "near_neighbor_family",
                "difficulty",
                "lifecycle_phase",
                "event_type",
                "payment_rail",
                "channel",
                "value_at_risk_inr",
            ]
        ].copy()
        scored["risk_score"] = probabilities
        scored["predicted_fraud"] = probabilities >= threshold
        scored["mitigation_action"] = [
            _mitigation_action(float(score), threshold) for score in probabilities
        ]
        scored_parts.append(scored)
    predictions = pd.concat(scored_parts, ignore_index=True)
    combined = predictions
    combined_metrics = _classification_metrics(
        combined["label_fraud"],
        combined["risk_score"].to_numpy(),
        threshold,
        values=combined["value_at_risk_inr"],
    )
    hard_controls = combined[combined["case_role"].eq("legitimate_control")]
    ordinary_legitimate = combined[
        combined["case_role"].eq("ordinary_legitimate")
    ]
    combined_metrics["hard_false_positive_rate"] = round(
        float(hard_controls["predicted_fraud"].mean()) if len(hard_controls) else 0.0,
        6,
    )
    combined_metrics["ordinary_false_positive_rate"] = round(
        float(ordinary_legitimate["predicted_fraud"].mean())
        if len(ordinary_legitimate)
        else 0.0,
        6,
    )
    combined_metrics["hard_control_event_count"] = int(len(hard_controls))
    family_results: dict[str, Any] = {}
    for family, group in combined[combined["label_fraud"]].groupby("attack_family"):
        family_results[str(family)] = {
            "events": int(len(group)),
            "detected": int(group["predicted_fraud"].sum()),
            "recall": round(float(group["predicted_fraud"].mean()), 6),
            "mean_risk_score": round(float(group["risk_score"].mean()), 6),
            "value_weighted_recall": round(
                float(
                    group.loc[group["predicted_fraud"], "value_at_risk_inr"].sum()
                    / group["value_at_risk_inr"].sum()
                )
                if group["value_at_risk_inr"].sum()
                else 0.0,
                6,
            ),
        }
    vector_results = {
        str(vector): {
            "events": int(len(group)),
            "recall": round(float(group["predicted_fraud"].mean()), 6),
            "mean_risk_score": round(float(group["risk_score"].mean()), 6),
            "split": str(group["split"].iloc[0]),
        }
        for vector, group in combined[combined["label_fraud"]].groupby("attack_vector_id")
    }
    phase_results = {
        str(phase): {
            "events": int(len(group)),
            "precision": round(
                float(precision_score(group["label_fraud"], group["predicted_fraud"], zero_division=0)),
                6,
            ),
            "recall": round(
                float(recall_score(group["label_fraud"], group["predicted_fraud"], zero_division=0)),
                6,
            ),
            "false_positive_rate": round(
                float(
                    ((~group["label_fraud"]) & group["predicted_fraud"]).sum()
                    / max(1, (~group["label_fraud"]).sum())
                ),
                6,
            ),
        }
        for phase, group in combined.groupby("lifecycle_phase")
    }
    benchmark = {
        "benchmark_version": "defense-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": metadata,
        "threat_atlas": atlas.summary(),
        "data_quality": quality,
        "fidelity": fidelity,
        "defense": {
            "architecture": "fast_guard_plus_qwen_agent_plus_measurable_tabular_benchmark",
            "selected_model": selected_name,
            "candidate_models": model_results,
            "threshold_selected_on_validation": round(threshold, 6),
            "prediction_contract": {
                "risk_score": "continuous probability-like score between 0 and 1",
                "flag": "risk_score >= validation-selected threshold",
                "actions": ["allow", "monitor", "step_up", "hold", "block"],
            },
            "metrics": {
                "combined_hidden_test": combined_metrics,
                "known_pattern_test": split_metrics["test_known"],
                "novel_vector_test": split_metrics["test_novel"],
            },
            "family_results": family_results,
            "vector_results": vector_results,
            "lifecycle_results": phase_results,
        },
        "kpi_framework": {
            "primary": [
                {
                    "metric": "hidden_test_pr_auc",
                    "definition": "Area under the precision-recall curve across known and sealed-novel test events.",
                    "value": combined_metrics["pr_auc"],
                },
                {
                    "metric": "novel_vector_recall",
                    "definition": "Recall on threat vectors absent from training and validation.",
                    "value": split_metrics["test_novel"]["recall"],
                },
                {
                    "metric": "hard_false_positive_rate",
                    "definition": "Flagged family-matched legitimate look-alikes divided by all hidden-test look-alikes.",
                    "value": combined_metrics["hard_false_positive_rate"],
                },
            ],
            "drivers": [
                {"metric": "fidelity_score", "value": fidelity["score"]},
                {"metric": "attack_vector_coverage", "value": fidelity["attack_vector_coverage"]},
                {"metric": "value_weighted_recall", "value": combined_metrics["value_weighted_recall"]},
            ],
            "guardrails": [
                {"metric": "critical_data_quality_failures", "target": 0, "value": quality["critical_failure_count"]},
                {"metric": "entity_or_scenario_split_leakage", "target": 0, "value": 0},
                {"metric": "overall_false_positive_rate", "provisional_target": "< 0.02", "value": combined_metrics["false_positive_rate"]},
                {"metric": "hard_false_positive_rate", "provisional_target": "< 0.03", "value": combined_metrics["hard_false_positive_rate"]},
            ],
        },
        "limitations": [
            "All events and entities are synthetic; metrics are stress-test evidence, not a production performance claim.",
            "The tabular detector provides scalable efficacy measurement while Qwen remains the explainable event investigator in the Agent Arena.",
            "Novel-vector evaluation withholds complete vector IDs, but vectors mapped to a known family may still share coarse event semantics.",
        ],
    }
    return benchmark, events, predictions
