"""Command-line entry point for live open-model lab runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from red_team_agent.catalog import AttackCatalog

from .benchmark import run_population_benchmark
from .config import AgentLabConfig
from .external_validation import (
    ExternalValidationConfig,
    run_external_validation,
    write_external_validation_artifacts,
)
from .orchestrator import SentinelLoopOrchestrator
from .population import PopulationConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SentinelLoop's agentic payment-defense lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("config", help="Show the active model configuration without secrets.")
    run = subparsers.add_parser("run", help="Run Red GenAI against Blue GenAI.")
    run.add_argument("--attack-family", default="ATO-01")
    run.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--seed", type=int, default=20260824)
    run.add_argument("--no-controls", action="store_true")
    run.add_argument("--output", type=Path, default=Path("runs/agentic/latest.json"))
    benchmark = subparsers.add_parser(
        "benchmark", help="Evaluate the configured open model across attack families and difficulties."
    )
    benchmark.add_argument("--families", nargs="+", default=None)
    benchmark.add_argument(
        "--difficulties", nargs="+", choices=("easy", "medium", "hard"), default=["medium"]
    )
    benchmark.add_argument("--seed", type=int, default=20260824)
    benchmark.add_argument(
        "--output", type=Path, default=Path("runs/agentic/benchmark.json")
    )
    population = subparsers.add_parser(
        "population-benchmark",
        help="Generate the Threat Atlas population and run a sealed defense benchmark.",
    )
    population.add_argument("--variants-per-vector", type=int, default=6)
    population.add_argument("--legitimate-events", type=int, default=2400)
    population.add_argument("--seed", type=int, default=20260824)
    population.add_argument(
        "--output-dir", type=Path, default=Path("data/benchmark")
    )
    external = subparsers.add_parser(
        "external-validate",
        help="Run chronological validation on the public anonymized ULB/Worldline dataset.",
    )
    external.add_argument(
        "--input", type=Path, default=Path("data/external/creditcard.parquet")
    )
    external.add_argument(
        "--output-dir", type=Path, default=Path("data/external_validation")
    )
    external.add_argument("--fpr-cap", type=float, default=0.002)
    external.add_argument("--seed", type=int, default=20260824)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "population-benchmark":
        population_config = PopulationConfig(
            variants_per_vector=args.variants_per_vector,
            legitimate_event_count=args.legitimate_events,
            seed=args.seed,
        )
        benchmark_payload, events, predictions = run_population_benchmark(
            population_config
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.output_dir / "population_events.csv", index=False)
        predictions.to_csv(args.output_dir / "defense_predictions.csv", index=False)
        (args.output_dir / "population_metadata.json").write_text(
            json.dumps(benchmark_payload["dataset"], indent=2), encoding="utf-8"
        )
        (args.output_dir / "data_quality.json").write_text(
            json.dumps(benchmark_payload["data_quality"], indent=2), encoding="utf-8"
        )
        (args.output_dir / "latest.json").write_text(
            json.dumps(benchmark_payload, indent=2), encoding="utf-8"
        )
        metrics = benchmark_payload["defense"]["metrics"]["combined_hidden_test"]
        print(
            json.dumps(
                {
                    "events": len(events),
                    "vectors": benchmark_payload["threat_atlas"]["vector_count"],
                    "fidelity_score": benchmark_payload["fidelity"]["score"],
                    "data_quality": benchmark_payload["data_quality"]["status"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "hard_false_positive_rate": metrics["hard_false_positive_rate"],
                    "output": str(args.output_dir / "latest.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "external-validate":
        external_config = ExternalValidationConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            validation_fpr_cap=args.fpr_cap,
            seed=args.seed,
        )
        external_report, external_predictions = run_external_validation(
            external_config
        )
        write_external_validation_artifacts(
            external_config, external_report, external_predictions
        )
        external_metrics = external_report["defense"]["test_metrics"]
        print(
            json.dumps(
                {
                    "assessment": external_report["overall_assessment"],
                    "dataset": external_report["dataset"]["name"],
                    "events": external_metrics["event_count"],
                    "fraud_events": external_metrics["fraud_event_count"],
                    "pr_auc": external_metrics["pr_auc"],
                    "pr_auc_95ci": external_metrics["pr_auc_95ci"],
                    "precision": external_metrics["precision"],
                    "recall": external_metrics["recall"],
                    "f1": external_metrics["f1"],
                    "false_positive_rate": external_metrics["false_positive_rate"],
                    "output": str(args.output_dir / "latest.json"),
                },
                indent=2,
            )
        )
        return

    config = AgentLabConfig.from_env()
    if args.command == "config":
        print(
            json.dumps(
                {
                    "model_base_url": config.model_base_url,
                    "red_model_id": config.red_model_id,
                    "blue_model_id": config.blue_model_id,
                    "structured_output_mode": config.structured_output_mode,
                    "case_parallelism": config.case_parallelism,
                    "note": "The API key is intentionally not displayed.",
                },
                indent=2,
            )
        )
        return

    if args.command == "benchmark":
        catalog = AttackCatalog()
        families = args.families or catalog.families
        unknown = sorted(set(families).difference(catalog.families))
        if unknown:
            raise SystemExit(f"Unknown attack families: {unknown}")
        cases: list[dict[str, object]] = []
        for family_index, family in enumerate(families):
            for difficulty_index, difficulty in enumerate(args.difficulties):
                case_seed = args.seed + family_index * 100 + difficulty_index
                result = SentinelLoopOrchestrator(config=config).run(
                    attack_family=family,
                    difficulty=difficulty,
                    rounds=1,
                    seed=case_seed,
                    include_legitimate_controls=True,
                )
                round_result = result.rounds[0]
                report = round_result.referee_report
                cases.append(
                    {
                        "attack_family": family,
                        "difficulty": difficulty,
                        "seed": case_seed,
                        "outcome": report.outcome,
                        "value_prevented_ratio": report.value_prevented_ratio,
                        "hard_false_positive_rate": report.hard_false_positive_rate,
                        "legitimate_friction_rate": report.legitimate_friction_rate,
                        "balanced_lifecycle_defense_score": report.balanced_lifecycle_defense_score,
                        "worst_phase_score": report.worst_phase_score,
                        "time_to_detect_seconds": report.time_to_detect_seconds,
                        "submission_evaluation": round_result.submission_evaluation,
                    }
                )
        count = len(cases)
        summary = {
            "case_count": count,
            "detection_rate": round(
                sum(item["outcome"] != "missed" for item in cases) / count, 4
            ),
            "mean_value_prevented_ratio": round(
                sum(float(item["value_prevented_ratio"]) for item in cases) / count, 4
            ),
            "mean_hard_false_positive_rate": round(
                sum(float(item["hard_false_positive_rate"]) for item in cases) / count, 4
            ),
            "mean_legitimate_friction_rate": round(
                sum(float(item["legitimate_friction_rate"]) for item in cases) / count, 4
            ),
            "mean_balanced_lifecycle_defense_score": round(
                sum(float(item["balanced_lifecycle_defense_score"]) for item in cases) / count,
                2,
            ),
        }
        payload = {
            "benchmark_type": "live_configured_open_model",
            "models": {"red": config.red_model_id, "blue": config.blue_model_id},
            "summary": summary,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({**summary, "output": str(args.output)}, indent=2))
        return

    lab = SentinelLoopOrchestrator(config=config)
    result = lab.run(
        attack_family=args.attack_family,
        difficulty=args.difficulty,
        rounds=args.rounds,
        seed=args.seed,
        include_legitimate_controls=not args.no_controls,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    latest = result.rounds[-1].referee_report
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "rounds": len(result.rounds),
                "final_outcome": latest.outcome,
                "blue_score": latest.blue_score,
                "red_score": latest.red_score,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
