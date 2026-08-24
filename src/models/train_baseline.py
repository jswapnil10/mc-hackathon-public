"""Train and evaluate the first leakage-controlled Blue Team baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, build_point_in_time_features


def _chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the earliest 60% for train, next 20% validation, final 20% test."""
    train_end = max(1, int(len(frame) * 0.60))
    validation_end = max(train_end + 1, int(len(frame) * 0.80))
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def _choose_threshold(labels: pd.Series, probabilities: Any) -> float:
    """Choose validation threshold by F1, preferring >=90% precision when available."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    candidates: list[tuple[float, float]] = []
    fallback: list[tuple[float, float]] = []
    for index, threshold in enumerate(thresholds):
        p, r = float(precision[index]), float(recall[index])
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fallback.append((f1, float(threshold)))
        if p >= 0.90:
            candidates.append((f1, float(threshold)))
    return max(candidates or fallback)[1]


def _metrics(labels: pd.Series, probabilities: Any, threshold: float) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    return {
        "average_precision": round(float(average_precision_score(labels, probabilities)), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate(events: pd.DataFrame) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    featured = build_point_in_time_features(events)
    train, validation, test = _chronological_split(featured)
    if min(train["label_fraud"].nunique(), validation["label_fraud"].nunique(), test["label_fraud"].nunique()) < 2:
        raise ValueError("Each chronological split must contain legitimate and fraudulent examples.")

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    model.fit(train[MODEL_FEATURES], train["label_fraud"])
    validation_probabilities = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
    threshold = _choose_threshold(validation["label_fraud"], validation_probabilities)
    test_probabilities = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    results = {
        "model": "logistic_regression_baseline",
        "split": "chronological_60_20_20",
        "threshold_selected_on_validation": round(threshold, 6),
        "row_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "fraud_counts": {
            "train": int(train["label_fraud"].sum()),
            "validation": int(validation["label_fraud"].sum()),
            "test": int(test["label_fraud"].sum()),
        },
        "test_metrics": _metrics(test["label_fraud"], test_probabilities, threshold),
        "limitations": [
            "This is a simulator smoke-test baseline, not a competition-ready efficacy claim.",
            "Current data contains only ATO-01 and MULE-01.",
            "Future evaluation must hold out entire campaigns and harder attack variants.",
        ],
    }
    predictions = test[
        ["event_id", "event_ts", "label_fraud", "attack_family", "scenario_id", "legitimate_control"]
    ].copy()
    predictions["risk_score"] = test_probabilities
    predictions["predicted_fraud"] = test_probabilities >= threshold
    legitimate_predictions = predictions[~predictions["label_fraud"]].copy()
    legitimate_predictions["profile"] = legitimate_predictions["legitimate_control"].fillna("ORDINARY")
    control_summary = legitimate_predictions.groupby("profile")["predicted_fraud"].agg(["count", "sum", "mean"])
    results["legitimate_profile_results"] = {
        str(profile): {
            "events": int(row["count"]),
            "false_positives": int(row["sum"]),
            "false_positive_rate": round(float(row["mean"]), 6),
        }
        for profile, row in control_summary.iterrows()
    }
    fraud_predictions = predictions[predictions["label_fraud"]]
    attack_summary = fraud_predictions.groupby("attack_family")["predicted_fraud"].agg(["count", "sum", "mean"])
    results["attack_family_results"] = {
        str(family): {
            "events": int(row["count"]),
            "detected": int(row["sum"]),
            "recall": round(float(row["mean"]), 6),
        }
        for family, row in attack_summary.iterrows()
    }
    return model, results, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SentinelLoop baseline detector.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/payment_events.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    events = pd.read_csv(args.input)
    _, results, predictions = train_and_evaluate(events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "baseline_metrics.json"
    predictions_path = args.output_dir / "baseline_test_predictions.csv"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    print(json.dumps(results, indent=2))
    print(f"Wrote metrics to {results_path} and test predictions to {predictions_path}")


if __name__ == "__main__":
    main()
