"""Validate and copy a recorded Agent Arena run into the public replay catalog."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/curate_demo_run.py SOURCE.json DESTINATION.json")
        return 2
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    if not source.is_file() or destination.suffix != ".json":
        print("Source must exist and destination must end in .json.")
        return 2
    payload = json.loads(source.read_text(encoding="utf-8"))
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        print("Source is missing Agent Arena rounds.")
        return 2
    scenario = rounds[0].get("red", {}).get("scenario", {})
    if not scenario.get("attack_family") or not scenario.get("difficulty"):
        print("Source is missing scenario family or difficulty metadata.")
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"Curated {scenario['attack_family']} / {scenario['difficulty']} / {len(rounds)} round(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
