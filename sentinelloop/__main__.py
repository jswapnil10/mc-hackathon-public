"""Command-line entry point for live open-model lab runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AgentLabConfig
from .orchestrator import SentinelLoopOrchestrator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SentinelLoop's agentic payment-defense lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("config", help="Show the active model configuration without secrets.")
    run = subparsers.add_parser("run", help="Run Red GenAI against Blue GenAI.")
    run.add_argument("--attack-family", default="ATO-01")
    run.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--seed", type=int, default=20260824)
    run.add_argument("--no-controls", action="store_true")
    run.add_argument("--output", type=Path, default=Path("runs/agentic/latest.json"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = AgentLabConfig.from_env()
    if args.command == "config":
        print(
            json.dumps(
                {
                    "model_base_url": config.model_base_url,
                    "red_model_id": config.red_model_id,
                    "blue_model_id": config.blue_model_id,
                    "structured_output_mode": config.structured_output_mode,
                    "note": "The API key is intentionally not displayed.",
                },
                indent=2,
            )
        )
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
