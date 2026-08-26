"""Generate a resumable catalog of recorded Qwen Agent Arena replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from red_team_agent.catalog import AttackCatalog
from sentinelloop.config import AgentLabConfig
from sentinelloop.orchestrator import SentinelLoopOrchestrator


DIFFICULTIES = ("easy", "medium", "hard")


def _valid_existing(path: Path, family: str, difficulty: str, rounds: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload["rounds"]
        scenario = recorded[0]["red"]["scenario"]
        return (
            len(recorded) == rounds
            and scenario["attack_family"] == family
            and scenario["difficulty"] == difficulty
            and all(item.get("submission_evaluation") for item in recorded)
        )
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo_runs"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    catalog = AttackCatalog()
    families = args.families or list(catalog.families)
    unknown = sorted(set(families) - set(catalog.families))
    if unknown:
        parser.error(f"Unknown attack families: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AgentLabConfig.from_env()
    failures: list[str] = []
    total = len(families) * len(DIFFICULTIES)
    position = 0
    for family_index, family in enumerate(families):
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            position += 1
            slug = family.lower().replace("-", "_")
            destination = args.output_dir / f"{slug}-{difficulty}-{args.rounds}r-qwen35-9b.json"
            if _valid_existing(destination, family, difficulty, args.rounds):
                print(f"[{position}/{total}] SKIP {family} / {difficulty}", flush=True)
                continue
            run_seed = args.seed + family_index * 100 + difficulty_index
            print(f"[{position}/{total}] RUN  {family} / {difficulty} / seed {run_seed}", flush=True)
            try:
                result = SentinelLoopOrchestrator(config=config).run(
                    attack_family=family,
                    difficulty=difficulty,
                    rounds=args.rounds,
                    seed=run_seed,
                    include_legitimate_controls=True,
                    include_ambient=False,
                    training_log_path=None,
                    retrain_every=None,
                )
                payload = result.to_dict()
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                temporary.replace(destination)
                if not _valid_existing(destination, family, difficulty, args.rounds):
                    raise ValueError("written artifact failed replay validation")
                print(f"[{position}/{total}] DONE {destination}", flush=True)
            except Exception as error:  # continue so one model hiccup does not lose the batch
                failures.append(f"{family}/{difficulty}: {error}")
                print(f"[{position}/{total}] FAIL {family} / {difficulty}: {error}", flush=True)

    if failures:
        print("Incomplete catalog:", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    print(f"Catalog complete: {total} validated replay artifacts.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
