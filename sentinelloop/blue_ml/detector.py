"""HistGBM fraud detector for the Blue team (Phase 1).

Mirrors the sibling repo's proven disciplines: HistGradientBoostingClassifier (native NaN +
categorical splits, no extra dependency), inverse-prevalence sample weights (no SMOTE), a
budget-driven threshold (never 0.5), and a fail-closed leakage-audit gate. Adds the piece the
sibling lacked: joblib persistence so a champion model can be reused at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import CATEGORICAL_FEATURES, FEATURES, audit_leakage

MODEL_PARAMS: dict[str, Any] = dict(
    max_iter=400,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=0,
)
DEFAULT_MODEL_DIR = Path("data/loop/models/champion")
_CAT_INDICES = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    X = X.reindex(columns=FEATURES)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


class FraudDetector:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        leaked = audit_leakage()
        assert not leaked, f"Feature allowlist leaks forbidden columns: {leaked}"
        self.params = dict(params or MODEL_PARAMS)
        self.clf = HistGradientBoostingClassifier(categorical_features=_CAT_INDICES, **self.params)
        self.threshold: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "FraudDetector":
        assert list(X.columns) == FEATURES, "training matrix must equal the FEATURES allowlist"
        X = _prepare(X)
        y = np.asarray(y).astype(int)
        pos = max(int(y.sum()), 1)
        weight = np.where(y == 1, (len(y) - pos) / pos, 1.0)  # inverse-prevalence, no SMOTE
        self.clf.fit(X, y, sample_weight=weight)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(_prepare(X))[:, 1]

    def save(self, directory: str | Path = DEFAULT_MODEL_DIR, *, meta: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.joblib"
        joblib.dump(
            {
                "clf": self.clf,
                "features": FEATURES,
                "params": self.params,
                "threshold": self.threshold,
                "meta": meta or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_MODEL_DIR) -> "FraudDetector":
        bundle = joblib.load(Path(directory) / "model.joblib")
        if bundle.get("features") != FEATURES:
            raise ValueError("Saved model feature list does not match the current FEATURES allowlist.")
        detector = cls(params=bundle.get("params"))
        detector.clf = bundle["clf"]
        detector.threshold = bundle.get("threshold")
        return detector
