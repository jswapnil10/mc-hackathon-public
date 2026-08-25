"""Train + evaluate the Blue-team fraud detector; persist the champion.

Grouped split (StratifiedGroupKFold on `group_key`, stratified by attack_family) prevents
kill-chain stages leaking across train/test. Threshold is chosen at a fixed alert budget, not
0.5. No model is wired into the live Blue agent here (that is Phase 2).

Run:  python scripts/train_detector.py --seed 42 --alert-rate 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sentinelloop importable

from sklearn.model_selection import StratifiedGroupKFold

from sentinelloop.blue_ml.detector import DEFAULT_MODEL_DIR, FraudDetector
from sentinelloop.blue_ml.feature_frame import build_feature_frame_from_seed
from sentinelloop.blue_ml.metrics import summarise, threshold_for_budget


_MEAN_KEYS = [
    "chain_recall", "prevented_share", "chain_precision", "row_recall", "row_precision",
    "hard_false_positive_rate", "ambient_friction_rate", "threshold", "mean_caught_at_stage",
]


def _aggregate(folds: list[dict]) -> dict:
    """Mean +/- std across CV folds for the headline metrics (skipping None)."""
    import numpy as np

    agg: dict = {}
    for key in _MEAN_KEYS:
        vals = [f[key] for f in folds if f.get(key) is not None]
        if vals:
            agg[key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    families: dict[str, list[float]] = {}
    for f in folds:
        for fam, val in f.get("chain_recall_by_family", {}).items():
            families.setdefault(fam, []).append(val)
    agg["chain_recall_by_family_mean"] = {fam: round(float(np.mean(v)), 4) for fam, v in sorted(families.items())}
    agg["chains_caught_total"] = sum(f["chains_caught"] for f in folds)
    agg["chains_total"] = sum(f["chains_total"] for f in folds)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-validate the Blue-team detector (no live wiring).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alert-rate", type=float, default=0.01)
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--seeds-per-cell", type=int, default=8)
    ap.add_argument("--report", type=Path, default=Path("data/loop/detector_report.json"))
    ap.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    X, y, groups, meta = build_feature_frame_from_seed(args.seed, seeds_per_cell=args.seeds_per_cell)
    strat = meta["attack_family"].fillna("none")

    n_splits = max(2, round(1 / args.test_share))  # test_share 0.3 -> 3 folds (~33% test each)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    fold_reports: list[dict] = []
    fold_thresholds: list[float] = []
    print(f"{n_splits}-fold grouped CV over {len(X):,} rows ({int(y.sum())} fraud), test_share~{1/n_splits:.2f}")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, strat, groups), start=1):
        det = FraudDetector().fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = det.score(X.iloc[test_idx])
        budget = max(1, round(args.alert_rate * len(test_idx)))
        thr = threshold_for_budget(scores, budget)
        fold_thresholds.append(thr)
        rep = summarise(meta.iloc[test_idx], y.iloc[test_idx], scores, thr)
        rep["fold"] = fold
        rep["train_rows"] = int(len(train_idx))
        fold_reports.append(rep)
        print(f"  fold {fold}: chain_recall={rep['chain_recall']:.3f} "
              f"chain_prec={rep['chain_precision']:.3f} row_rec/prec={rep['row_recall']:.3f}/{rep['row_precision']:.3f} "
              f"hardFP={rep['hard_false_positive_rate']:.4f} amb={rep['ambient_friction_rate']:.4f}")

    cv = _aggregate(fold_reports)

    # Ship a champion refit on ALL data; threshold = mean of the held-out fold thresholds.
    champion_threshold = float(sum(fold_thresholds) / len(fold_thresholds))
    champion = FraudDetector().fit(X, y)
    champion.threshold = champion_threshold
    model_path = champion.save(
        args.model_out,
        meta={"seed": args.seed, "alert_rate": args.alert_rate, "n_splits": n_splits,
              "trained_on_rows": int(len(X)), "cv": cv},
    )

    report = {
        "seed": args.seed, "alert_rate": args.alert_rate, "test_share": args.test_share,
        "n_splits": n_splits, "total_rows": int(len(X)), "total_fraud": int(y.sum()),
        "cv_summary": cv, "folds": fold_reports, "champion_threshold": round(champion_threshold, 6),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _fmt(k: str) -> str:
        v = cv.get(k)
        return f"{v['mean']:.4f} +/- {v['std']:.4f}" if v else "n/a"

    print("\n=== CV MEAN (across folds) ===")
    print(f"CHAIN recall   = {_fmt('chain_recall')}   ({cv['chains_caught_total']}/{cv['chains_total']} chains)")
    print(f"  by family    : {cv['chain_recall_by_family_mean']}")
    print(f"prevented_share= {_fmt('prevented_share')}")
    print(f"chain precision= {_fmt('chain_precision')}")
    print(f"row rec/prec   = {_fmt('row_recall')} / {_fmt('row_precision')}")
    print(f"hard-FP control= {_fmt('hard_false_positive_rate')}   ambient friction= {_fmt('ambient_friction_rate')}")
    print(f"\nchampion refit on ALL {len(X):,} rows (thr={champion_threshold:.6f}) -> {model_path}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
