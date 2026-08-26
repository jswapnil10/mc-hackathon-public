"""Run one generation's champion/challenger retrain from the battle training log.

Blends the synthetic baseline + accumulated battle log + replay buffer, cross-validates a
challenger, and promotes it over the incumbent champion only if it does not regress chain recall
or increase hard-FP. Run:  python scripts/retrain_detector.py --generation 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinelloop.blue_ml.detector import DEFAULT_MODEL_DIR
from sentinelloop.blue_ml.retrain import retrain


def main() -> None:
    ap = argparse.ArgumentParser(description="Champion/challenger retrain for the Blue detector.")
    ap.add_argument("--log", type=Path, default=Path("data/loop/training_log.jsonl"))
    ap.add_argument("--generation", type=int, default=None)
    ap.add_argument("--baseline-seeds-per-cell", type=int, default=8)
    ap.add_argument("--alert-rate", type=float, default=0.01)
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--min-delta", type=float, default=0.01, help="max allowed chain-recall regression")
    ap.add_argument("--fp-tol", type=float, default=0.002, help="max allowed hard-FP increase (tight)")
    ap.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="minimum real gain required in recall, hard-FP, chain precision, or prevention",
    )
    ap.add_argument("--champion-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--report", type=Path, default=Path("data/loop/retrain_report.json"))
    args = ap.parse_args()

    decision = retrain(
        args.log,
        champion_dir=args.champion_dir,
        baseline_seeds_per_cell=args.baseline_seeds_per_cell,
        alert_rate=args.alert_rate,
        test_share=args.test_share,
        min_delta=args.min_delta,
        fp_tol=args.fp_tol,
        min_improvement=args.min_improvement,
        generation=args.generation,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    verdict = "PROMOTED" if decision["promoted"] else "REJECTED"
    ch = decision["challenger"]
    inc = decision["incumbent"]
    print(f"generation {decision['generation']} | rows={decision['rows']:,} fraud={decision['fraud_rows']}")
    print(f"challenger: chain_recall={ch['chain_recall']:.3f} hard_fp={ch['hard_false_positive_rate']:.4f} "
          f"chain_prec={ch['chain_precision']:.3f}")
    print(f"incumbent : {inc}")
    print(f"=> challenger {verdict}")
    if decision["promoted"]:
        print(f"   new champion threshold = {decision.get('champion_threshold')}  -> {args.champion_dir}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
