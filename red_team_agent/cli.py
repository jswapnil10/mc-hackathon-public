"""Command-line interface for planning, validating, and mutating scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import AttackCatalog
from .models import RefereeFeedback, ScenarioSpec
from .mutation import MutationController
from .openai_backend import OpenAIPlanningBackend
from .planner import OfflinePlanningBackend, RedTeamAgent
from .safety import ScenarioSafetyGate


def _write_or_print(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(rendered, end="")


def _agent(backend_name: str, catalog: AttackCatalog) -> RedTeamAgent:
    if backend_name == "openai":
        backend = OpenAIPlanningBackend(catalog)
    else:
        backend = OfflinePlanningBackend(catalog)
    return RedTeamAgent(catalog=catalog, backend=backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe synthetic Red Team Agent for SentinelLoop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List curated attack cards.")

    show = subparsers.add_parser("show", help="Show one attack card.")
    show.add_argument("--family", required=True)

    plan = subparsers.add_parser("plan", help="Plan and compile a validated scenario.")
    plan.add_argument("--family", choices=None)
    plan.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    plan.add_argument("--objective")
    plan.add_argument("--seed", type=int, default=20260819)
    plan.add_argument("--backend", choices=("offline", "openai"), default="offline")
    plan.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate", help="Validate an existing scenario JSON file.")
    validate.add_argument("scenario", type=Path)

    mutate = subparsers.add_parser("mutate", help="Create the next variant from Referee feedback.")
    mutate.add_argument("--scenario", type=Path, required=True)
    mutate.add_argument("--feedback", type=Path, required=True)
    mutate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = AttackCatalog()
    if args.command == "list":
        for card in catalog.list():
            print(f"{card.attack_family:<9} {card.name}")
        return 0
    if args.command == "show":
        card = catalog.get(args.family)
        _write_or_print(card.__dict__, None)
        return 0
    if args.command == "plan":
        if args.family and args.family not in catalog.families:
            raise SystemExit(f"Unknown family {args.family!r}; choose from {', '.join(catalog.families)}")
        scenario = _agent(args.backend, catalog).plan(
            attack_family=args.family,
            difficulty=args.difficulty,
            objective=args.objective,
            seed=args.seed,
        )
        _write_or_print(scenario.to_dict(), args.output)
        return 0
    if args.command == "validate":
        scenario = ScenarioSpec.from_dict(json.loads(args.scenario.read_text(encoding="utf-8")))
        report = ScenarioSafetyGate(catalog).validate(scenario)
        _write_or_print(report.to_dict(), None)
        return 0 if report.approved else 1
    if args.command == "mutate":
        scenario = ScenarioSpec.from_dict(json.loads(args.scenario.read_text(encoding="utf-8")))
        feedback = RefereeFeedback.from_dict(json.loads(args.feedback.read_text(encoding="utf-8")))
        mutated = MutationController(catalog).mutate(scenario, feedback)
        _write_or_print(mutated.to_dict(), args.output)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
