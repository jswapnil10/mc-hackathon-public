"""Generate a resumable catalog of recorded Agent Arena replays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from red_team_agent.catalog import AttackCatalog
from sentinelloop.config import AgentLabConfig
from sentinelloop.model_gateway import OpenAICompatibleGateway
from sentinelloop.orchestrator import SentinelLoopOrchestrator


DIFFICULTIES = ("easy", "medium", "hard")


def _generation_gateway(
    config: AgentLabConfig,
    *,
    source_repo: Path | None,
    session_id: str,
) -> OpenAICompatibleGateway | None:
    """Optionally reuse the MARS/Kong authentication path from another local repo."""
    if source_repo is None:
        return None
    app_root = source_repo / "bos-cc-stream"
    source_dir = app_root / "src"
    if not (source_dir / "proxy_utils.py").is_file():
        raise ValueError(
            f"{source_repo} does not contain bos-cc-stream/src/proxy_utils.py"
        )
    for path in (app_root, source_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from proxy_utils import get_proxy_auth_token

    return OpenAICompatibleGateway(
        config,
        api_key_provider=get_proxy_auth_token,
        extra_body={"trace_data": {"session_id": session_id}},
    )


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
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--difficulties", nargs="*", choices=DIFFICULTIES, default=None)
    parser.add_argument("--artifact-label", default="qwen35-9b")
    parser.add_argument(
        "--prompt-profile",
        choices=("generic", "claude"),
        default=None,
        help="Provider-specific structured-response guidance (defaults to MODEL_PROMPT_PROFILE).",
    )
    parser.add_argument(
        "--mars-auth-source",
        type=Path,
        default=None,
        help="Local bos-cc-stream repository used only to mint cached Kong tokens.",
    )
    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be at least 1")

    catalog = AttackCatalog()
    families = args.families or list(catalog.families)
    difficulties = args.difficulties or list(DIFFICULTIES)
    unknown = sorted(set(families) - set(catalog.families))
    if unknown:
        parser.error(f"Unknown attack families: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AgentLabConfig.from_env()
    if args.prompt_profile is not None:
        config = replace(config, prompt_profile=args.prompt_profile)
        config.validate()
    failures: list[str] = []
    total = len(families) * len(difficulties)
    position = 0
    for family_index, family in enumerate(families):
        for difficulty_index, difficulty in enumerate(difficulties):
            position += 1
            slug = family.lower().replace("-", "_")
            destination = args.output_dir / (
                f"{slug}-{difficulty}-{args.rounds}r-{args.artifact_label}.json"
            )
            if _valid_existing(destination, family, difficulty, args.rounds):
                print(f"[{position}/{total}] SKIP {family} / {difficulty}", flush=True)
                continue
            base_seed = args.seed + family_index * 100 + difficulty_index
            for attempt in range(1, args.attempts + 1):
                run_seed = base_seed + (attempt - 1) * 10_000
                print(
                    f"[{position}/{total}] RUN  {family} / {difficulty} / "
                    f"attempt {attempt}/{args.attempts} / seed {run_seed}",
                    flush=True,
                )

                def report_progress(update: dict[str, object]) -> None:
                    if update.get("stage") == "round_complete":
                        print(
                            f"[{position}/{total}] ROUND "
                            f"{update.get('round_number')}/{args.rounds} complete",
                            flush=True,
                        )

                try:
                    gateway = _generation_gateway(
                        config,
                        source_repo=args.mars_auth_source,
                        session_id=f"masterguard-{family.lower()}-{difficulty}-{args.rounds}r",
                    )
                    result = SentinelLoopOrchestrator(
                        config=config,
                        gateway=gateway,
                        progress_callback=report_progress,
                    ).run(
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
                    break
                except Exception as error:  # retry transient/model-contract failures
                    if attempt < args.attempts:
                        print(
                            f"[{position}/{total}] RETRY {family} / {difficulty}: {error}",
                            flush=True,
                        )
                        continue
                    failures.append(f"{family}/{difficulty}: {error}")
                    print(
                        f"[{position}/{total}] FAIL {family} / {difficulty}: {error}",
                        flush=True,
                    )

    if failures:
        print("Incomplete catalog:", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    print(f"Catalog complete: {total} validated replay artifacts.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
