"""Flask web prototype for the agentic lab, with the v1 dashboard preserved."""

from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from red_team_agent.catalog import AttackCatalog
from sentinelloop.config import AgentLabConfig
from sentinelloop.evaluation import catalog_submission_profile
from sentinelloop.orchestrator import SentinelLoopOrchestrator
from sentinelloop.threat_atlas import ThreatAtlas
from sentinelloop.trace import trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVENTS_PATH = DATA_DIR / "payment_events.csv"
PREDICTIONS_PATH = DATA_DIR / "baseline_test_predictions.csv"
METRICS_PATH = DATA_DIR / "baseline_metrics.json"
LEGACY_DEMO_DIR = PROJECT_ROOT / "app" / "legacy_demo"
AGENT_RUNS_DIR = PROJECT_ROOT / "runs" / "agentic"
LATEST_AGENT_RUN_PATH = AGENT_RUNS_DIR / "latest.json"
PRECOMPUTED_DEMO_DIR = PROJECT_ROOT / "data" / "demo_runs"
WEB_MAX_ROUNDS = 5
AGENT_RUN_LOCK = threading.Lock()
RUN_PROGRESS_LOCK = threading.Lock()
RUN_PROGRESS: dict[str, dict[str, Any]] = {}
RUN_PROGRESS_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{8,80}$")
BATTLE_COUNT_PATH = PROJECT_ROOT / "data" / "loop" / "battle_count.txt"
BATTLE_COUNT_LOCK = threading.Lock()
BACKGROUND_RETRAIN_LOCK = threading.Lock()
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
LATEST_BENCHMARK_PATH = BENCHMARK_DIR / "latest.json"
BENCHMARK_RUN_LOCK = threading.Lock()
EXTERNAL_VALIDATION_DIR = PROJECT_ROOT / "data" / "external_validation"
LATEST_EXTERNAL_VALIDATION_PATH = EXTERNAL_VALIDATION_DIR / "latest.json"


def _precomputed_demo_enabled() -> bool:
    return os.environ.get("DEMO_MODE", "").strip().lower() in {
        "precomputed", "replay", "offline"
    }


def _load_precomputed_runs() -> list[dict[str, Any]]:
    """Load complete, curated Agent Arena run artifacts for the no-LLM demo."""
    runs: list[dict[str, Any]] = []
    for path in sorted(PRECOMPUTED_DEMO_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("rounds"), list) and payload["rounds"]:
            payload["_demo_source"] = path.name
            runs.append(payload)
    return runs


def _select_precomputed_run(
    *, attack_family: str, difficulty: str, rounds: int, seed: int
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for payload in _load_precomputed_runs():
        recorded_rounds = payload["rounds"]
        scenario = recorded_rounds[0].get("red", {}).get("scenario", {})
        if (
            scenario.get("attack_family") == attack_family
            and scenario.get("difficulty") == difficulty
        ):
            matches.append(payload)
    if not matches:
        return None

    rng = random.Random(seed)
    longest_recording = max(len(payload["rounds"]) for payload in matches)
    complete_matches = [
        payload
        for payload in matches
        if len(payload["rounds"]) >= rounds
        and len(payload["rounds"]) == longest_recording
    ]
    sources: list[str]
    if complete_matches:
        selected = copy.deepcopy(rng.choice(complete_matches))
        recorded_rounds = selected["rounds"]
        sources = [selected.pop("_demo_source", "curated artifact")]
        selected["rounds"] = recorded_rounds[:rounds]
        selected["duration_ms"] = sum(
            int(item.get("duration_ms") or 0) for item in selected["rounds"]
        )
        if len(recorded_rounds) > rounds:
            next_playbook = (
                recorded_rounds[rounds].get("blue", {}).get("active_playbook")
            )
            if next_playbook:
                selected["final_defense_playbook"] = copy.deepcopy(next_playbook)
        sequence_kind = "recorded_run" if rounds == 1 else "adaptive_recorded_run"
    else:
        single_round_matches = [
            payload for payload in matches if len(payload["rounds"]) == 1
        ]
        if not single_round_matches:
            return None
        rng.shuffle(single_round_matches)
        chosen = [
            single_round_matches[index % len(single_round_matches)]
            for index in range(rounds)
        ]
        selected = copy.deepcopy(chosen[0])
        selected["rounds"] = []
        sources = []
        for round_number, payload in enumerate(chosen, start=1):
            replay_round = copy.deepcopy(payload["rounds"][0])
            replay_round["round_number"] = round_number
            replay_round["blue_adaptation"] = None
            replay_round["recorded_replay_source"] = payload.get(
                "_demo_source", "curated artifact"
            )
            selected["rounds"].append(replay_round)
            sources.append(payload.get("_demo_source", "curated artifact"))
        selected["final_defense_playbook"] = copy.deepcopy(
            chosen[-1].get(
                "final_defense_playbook", selected["final_defense_playbook"]
            )
        )
        selected["duration_ms"] = sum(
            int(item.get("duration_ms") or 0) for item in selected["rounds"]
        )
        selected.pop("_demo_source", None)
        sequence_kind = "independent_recorded_replays"

    selected["run_id"] = f"{selected.get('run_id', 'LAB-RECORDED')}-REPLAY-{rounds}R"
    selected["demo_mode"] = "precomputed_replay"
    selected["demo_provenance"] = {
        "label": "Precomputed replay of recorded, bounded agent output",
        "sequence_kind": sequence_kind,
        "requested_rounds": rounds,
        "unique_recordings_used": len(set(sources)),
        "sources": list(dict.fromkeys(sources)),
        "note": (
            "This is a recorded adaptive multi-round run."
            if sequence_kind == "adaptive_recorded_run"
            else "Replay cycles are recorded snapshots; no live cross-round model learning occurs."
            if rounds > 1
            else "No live model call occurs."
        ),
    }
    return selected


def _precomputed_demo_catalog() -> list[dict[str, Any]]:
    """Return only selectable replay combinations, preventing dead-end UI choices."""
    scenario_capacity: dict[tuple[str, str], int] = {}
    for payload in _load_precomputed_runs():
        recorded_rounds = payload["rounds"]
        scenario = recorded_rounds[0].get("red", {}).get("scenario", {})
        family = scenario.get("attack_family")
        difficulty = scenario.get("difficulty")
        if not family or not difficulty:
            continue
        capacity = WEB_MAX_ROUNDS if len(recorded_rounds) == 1 else len(recorded_rounds)
        key = (family, difficulty)
        scenario_capacity[key] = max(
            scenario_capacity.get(key, 0), min(WEB_MAX_ROUNDS, capacity)
        )
    return [
        {"attack_family": family, "difficulty": difficulty, "rounds": rounds}
        for (family, difficulty), capacity in sorted(scenario_capacity.items())
        for rounds in range(1, capacity + 1)
    ]


def _set_run_progress(progress_id: str, payload: dict[str, Any]) -> None:
    """Keep a small, sanitized in-memory progress record for browser polling."""
    with RUN_PROGRESS_LOCK:
        RUN_PROGRESS[progress_id] = {
            **payload,
            "progress_id": progress_id,
            "updated_at_epoch_ms": round(time.time() * 1000),
        }
        while len(RUN_PROGRESS) > 8:
            removable = [key for key in RUN_PROGRESS if key != progress_id]
            oldest = min(
                removable or list(RUN_PROGRESS),
                key=lambda key: RUN_PROGRESS[key].get("updated_at_epoch_ms", 0),
            )
            RUN_PROGRESS.pop(oldest, None)


def _read_battle_count() -> int:
    try:
        return int(BATTLE_COUNT_PATH.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def _bump_battle_count() -> int:
    """Atomically record one successful web battle for the retraining cadence."""
    with BATTLE_COUNT_LOCK:
        battle_number = _read_battle_count() + 1
        BATTLE_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = BATTLE_COUNT_PATH.with_suffix(".tmp")
        temporary_path.write_text(str(battle_number), encoding="utf-8")
        temporary_path.replace(BATTLE_COUNT_PATH)
        return battle_number


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _background_retrain(
    *,
    training_log_path: str,
    champion_dir: str,
    generation: int,
) -> None:
    """Retrain outside the request path; single-flight and failure-isolated."""
    if not BACKGROUND_RETRAIN_LOCK.acquire(blocking=False):
        trace(
            "app.retrain.skipped",
            "A background retrain is already in flight; the current champion remains active.",
            generation=generation,
        )
        return
    try:
        from sentinelloop.blue_ml.retrain import retrain

        decision = retrain(
            _project_path(training_log_path),
            champion_dir=_project_path(champion_dir),
            generation=generation,
        )
        trace(
            "app.retrain.done",
            "Background champion/challenger evaluation finished.",
            generation=generation,
            promoted=decision.get("promoted"),
            challenger=decision.get("challenger"),
        )
    except Exception as error:  # noqa: BLE001 - retraining must not crash the web process
        trace(
            "app.retrain.failed",
            "Background retraining failed; the current champion remains active.",
            generation=generation,
            error=str(error),
        )
    finally:
        BACKGROUND_RETRAIN_LOCK.release()


def _record_learning_loop(config: AgentLabConfig) -> dict[str, Any]:
    """Count a completed battle and schedule a due retrain without delaying the response."""
    every = config.retrain_every_battles
    enabled = bool(config.ml_detector_enabled and config.training_log_path and every)
    try:
        battle_number = _bump_battle_count()
    except OSError as error:
        trace(
            "app.battle_count.failed",
            "The battle completed, but its local learning-loop counter could not be persisted.",
            error=str(error),
        )
        return {
            "battle_number": None,
            "auto_retrain_every_battles": every,
            "enabled": enabled,
            "retrain_scheduled": False,
        }

    due = bool(enabled and battle_number % every == 0)
    if due:
        generation = battle_number // every
        trace(
            "app.retrain.scheduled",
            "The completed-battle threshold was reached; retraining continues in the background.",
            battle_number=battle_number,
            generation=generation,
            every=every,
        )
        threading.Thread(
            target=_background_retrain,
            kwargs={
                "training_log_path": config.training_log_path,
                "champion_dir": config.ml_model_dir,
                "generation": generation,
            },
            daemon=True,
            name=f"masterguard-retrain-{generation}",
        ).start()
    return {
        "battle_number": battle_number,
        "auto_retrain_every_battles": every,
        "enabled": enabled,
        "retrain_scheduled": due,
    }


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    primary_paths = (EVENTS_PATH, PREDICTIONS_PATH, METRICS_PATH)
    fallback_paths = (
        LEGACY_DEMO_DIR / "payment_events.csv",
        LEGACY_DEMO_DIR / "baseline_test_predictions.csv",
        LEGACY_DEMO_DIR / "baseline_metrics.json",
    )
    selected_paths = primary_paths if all(path.exists() for path in primary_paths) else fallback_paths
    missing = [path for path in selected_paths if not path.exists()]
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing dashboard data: {names}. Restore the bundled legacy demo or run the "
            "simulator and baseline commands."
        )
    events_path, predictions_path, metrics_path = selected_paths
    events = pd.read_csv(events_path)
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
    predictions = pd.read_csv(predictions_path)
    predictions["event_ts"] = pd.to_datetime(predictions["event_ts"], utc=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return events, predictions, metrics


def _timeline(events: pd.DataFrame) -> list[dict[str, Any]]:
    daily = (
        events.assign(day=events["event_ts"].dt.strftime("%d %b"))
        .groupby(["day", "label_fraud"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    return [
        {
            "day": str(index),
            "legitimate": int(row.get(False, 0)),
            "fraud": int(row.get(True, 0)),
        }
        for index, row in daily.iterrows()
    ]


def _risk_histogram(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    buckets = pd.cut(predictions["risk_score"], bins=boundaries, labels=labels, include_lowest=True, right=False)
    grouped = predictions.assign(bucket=buckets).groupby(["bucket", "label_fraud"], observed=False).size().unstack(fill_value=0)
    return [
        {
            "bucket": str(index),
            "legitimate": int(row.get(False, 0)),
            "fraud": int(row.get(True, 0)),
        }
        for index, row in grouped.iterrows()
    ]


def _campaigns(events: pd.DataFrame) -> list[dict[str, Any]]:
    fraud = events[events["label_fraud"]].copy()
    grouped = fraud.groupby(["attack_family", "scenario_id"], dropna=False)
    result: list[dict[str, Any]] = []
    for (family, scenario), group in grouped:
        campaign_number = int(str(scenario).rsplit("-", 1)[-1])
        result.append(
            {
                "family": str(family),
                "scenario": str(scenario),
                "difficulty": ("Easy", "Medium", "Hard")[(campaign_number - 1) % 3],
                "events": len(group),
                "average_amount": round(float(group["amount_inr"].mean()), 2),
                "start": group["event_ts"].min().strftime("%d %b, %H:%M"),
            }
        )
    return sorted(result, key=lambda item: item["scenario"])


def _network(events: pd.DataFrame) -> dict[str, Any]:
    mule_events = events[events["attack_family"].eq("MULE-01")]
    if mule_events.empty:
        return {"scenario": None, "nodes": [], "edges": []}
    scenario = str(mule_events.groupby("scenario_id").size().idxmax())
    campaign = mule_events[mule_events["scenario_id"].eq(scenario)].head(100)
    account_ids = list(dict.fromkeys(campaign["sender_account_id"].astype(str)))
    beneficiary_ids = list(dict.fromkeys(campaign["beneficiary_id"].astype(str)))
    nodes = [
        {"id": node, "kind": "account", "label": node.replace("acct_", "A-")}
        for node in account_ids
    ] + [
        {"id": node, "kind": "beneficiary", "label": node.replace("mule_", "M-")}
        for node in beneficiary_ids
    ]
    edges = [
        {
            "source": str(row.sender_account_id),
            "target": str(row.beneficiary_id),
            "amount": round(float(row.amount_inr), 2),
        }
        for row in campaign.itertuples()
    ]
    return {"scenario": scenario, "nodes": nodes, "edges": edges}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    # Keep the HTML shell synchronized with its CSS and JavaScript during local demos.
    # This is intentionally enabled even when Flask debug mode is off.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.after_request
    def security_headers(response: Any) -> Any:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("lab.html")

    @app.get("/healthz")
    def healthz() -> Any:
        """Small dependency-free health check for managed web hosts."""
        return jsonify({"status": "ok", "service": "masterguard-ai"})

    @app.get("/legacy")
    def legacy_dashboard() -> str:
        return render_template("index.html")

    @app.get("/api/v2/status")
    def agent_status() -> Any:
        config = AgentLabConfig.from_env()
        detector_dir = Path(config.ml_model_dir)
        if not detector_dir.is_absolute():
            detector_dir = PROJECT_ROOT / detector_dir
        detector_available = config.ml_detector_enabled and (detector_dir / "model.joblib").is_file()
        catalog = AttackCatalog()
        atlas = ThreatAtlas()
        return jsonify(
            {
                "system": "MasterGuard AI - Attack. Adapt. Defend.",
                "mode": "precomputed_replay" if _precomputed_demo_enabled() else "live_open_model",
                "architecture": (
                    "Red GenAI → Synthetic Arena → Blue gradient-boosting detector + sequence guard + "
                    "GenAI Investigator → Deterministic Referee → separated Red/Blue feedback loops"
                ),
                "models": {
                    "red": config.red_model_id,
                    "blue": config.blue_model_id,
                    "blue_detector": "HistGradientBoostingClassifier",
                    "blue_detector_enabled": config.ml_detector_enabled,
                    "blue_detector_active": detector_available,
                    "blue_strategist": config.blue_model_id,
                    "referee": "deterministic-policy-v2",
                },
                "model_endpoint": "server-side and private",
                "structured_output_mode": config.structured_output_mode,
                "latency_profile": {
                    "blue_model_calls_per_event": 1,
                    "case_parallelism": config.case_parallelism,
                    "recommended_demo_rounds": 1,
                    "reasoning_profiles": {
                        "red_planner": config.reasoning_effort_for("red_planner"),
                        "blue_live_event": config.reasoning_effort_for("blue_event_agent"),
                        "blue_between_rounds": config.reasoning_effort_for("blue_strategist"),
                    },
                    "model_call_timeout_seconds": config.request_timeout_seconds,
                },
                "learning_loop": {
                    "blue_post_referee_logging": bool(config.training_log_path),
                    "auto_retrain_enabled": bool(
                        config.ml_detector_enabled
                        and config.training_log_path
                        and config.retrain_every_battles
                    ),
                    "auto_retrain_every_battles": config.retrain_every_battles,
                    "completed_battle_count": _read_battle_count(),
                    "promotion_gate": "grouped_k_fold_champion_challenger",
                },
                "attack_families": [
                    {
                        "id": card.attack_family,
                        "name": card.name,
                        "genai_role": card.genai_role,
                    }
                    for card in catalog.list()
                ],
                "latest_run_available": LATEST_AGENT_RUN_PATH.exists(),
                "latest_benchmark_available": LATEST_BENCHMARK_PATH.exists(),
                "latest_external_validation_available": LATEST_EXTERNAL_VALIDATION_PATH.exists(),
                "precomputed_demo": {
                    "enabled": _precomputed_demo_enabled(),
                    "run_count": len(_load_precomputed_runs()),
                    "available_scenarios": _precomputed_demo_catalog(),
                    "disclosure": "Recorded, bounded agent runs; no model is called during replay.",
                },
                "threat_atlas": atlas.summary(),
                "truth_boundary": "Blue receives sanitized observables only; Referee owns sealed labels.",
                "submission_profile": catalog_submission_profile(catalog),
            }
        )

    @app.get("/api/v2/threat-atlas")
    def threat_atlas() -> Any:
        return jsonify(ThreatAtlas().to_dict())

    @app.get("/api/v2/benchmark")
    def latest_population_benchmark() -> Any:
        if not LATEST_BENCHMARK_PATH.exists():
            return jsonify(
                {
                    "error": "No population benchmark has been generated yet.",
                    "action": "Run the Scenario Foundry from this page.",
                }
            ), 404
        return jsonify(json.loads(LATEST_BENCHMARK_PATH.read_text(encoding="utf-8")))

    @app.get("/api/v2/external-validation")
    def latest_external_validation() -> Any:
        if not LATEST_EXTERNAL_VALIDATION_PATH.exists():
            return jsonify(
                {
                    "error": "No external validation has been generated yet.",
                    "action": "Run python -m sentinelloop external-validate after downloading the public dataset.",
                }
            ), 404
        return jsonify(
            json.loads(LATEST_EXTERNAL_VALIDATION_PATH.read_text(encoding="utf-8"))
        )

    @app.post("/api/v2/benchmark/run")
    def run_population_defense_benchmark() -> Any:
        if not BENCHMARK_RUN_LOCK.acquire(blocking=False):
            return jsonify({"error": "The Scenario Foundry is already running."}), 409
        try:
            from sentinelloop.benchmark import run_population_benchmark
            from sentinelloop.population import PopulationConfig

            body = request.get_json(silent=True) or {}
            config = PopulationConfig(
                variants_per_vector=int(body.get("variants_per_vector", 6)),
                legitimate_event_count=int(body.get("legitimate_events", 2400)),
                seed=int(body.get("seed", 20260824)),
            )
            config.validate()
            benchmark, events, predictions = run_population_benchmark(config)
            BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
            temporary_json = BENCHMARK_DIR / ".latest.json.tmp"
            temporary_json.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
            temporary_json.replace(LATEST_BENCHMARK_PATH)
            events.to_csv(BENCHMARK_DIR / "population_events.csv", index=False)
            predictions.to_csv(BENCHMARK_DIR / "defense_predictions.csv", index=False)
            (BENCHMARK_DIR / "population_metadata.json").write_text(
                json.dumps(benchmark["dataset"], indent=2), encoding="utf-8"
            )
            (BENCHMARK_DIR / "data_quality.json").write_text(
                json.dumps(benchmark["data_quality"], indent=2), encoding="utf-8"
            )
            return jsonify(benchmark)
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 422
        finally:
            BENCHMARK_RUN_LOCK.release()

    @app.get("/api/v2/latest")
    def latest_agent_run() -> Any:
        if not LATEST_AGENT_RUN_PATH.exists():
            return jsonify({"error": "No agentic run has been completed yet."}), 404
        return jsonify(json.loads(LATEST_AGENT_RUN_PATH.read_text(encoding="utf-8")))

    @app.get("/api/v2/run-progress/<progress_id>")
    def agent_run_progress(progress_id: str) -> Any:
        if not RUN_PROGRESS_ID_PATTERN.fullmatch(progress_id):
            return jsonify({"error": "Invalid progress identifier."}), 400
        with RUN_PROGRESS_LOCK:
            progress = RUN_PROGRESS.get(progress_id)
            snapshot = dict(progress) if progress is not None else None
        if snapshot is None:
            return jsonify({"error": "This battle has not started reporting progress yet."}), 404
        return jsonify(snapshot)

    @app.post("/api/v2/run")
    def run_agent_lab() -> Any:
        if not AGENT_RUN_LOCK.acquire(blocking=False):
            return jsonify({"error": "Another Agent Arena run is already in progress."}), 409
        progress_id: str | None = None
        try:
            body = request.get_json(silent=True) or {}
            family = str(body.get("attack_family", "ATO-01"))
            difficulty = str(body.get("difficulty", "medium"))
            rounds = int(body.get("rounds", 1))
            seed = int(body.get("seed", 20260824))
            requested_progress_id = str(body.get("progress_id", "")).strip()
            if requested_progress_id:
                if not RUN_PROGRESS_ID_PATTERN.fullmatch(requested_progress_id):
                    return jsonify({"error": "Invalid progress identifier."}), 400
                progress_id = requested_progress_id
            if family not in AttackCatalog().families:
                return jsonify({"error": f"Unknown attack family {family!r}."}), 400
            if difficulty not in {"easy", "medium", "hard"}:
                return jsonify({"error": "Difficulty must be easy, medium, or hard."}), 400
            if not 1 <= rounds <= WEB_MAX_ROUNDS:
                return jsonify(
                    {"error": f"The web demo supports between 1 and {WEB_MAX_ROUNDS} rounds."}
                ), 400

            if _precomputed_demo_enabled():
                result = _select_precomputed_run(
                    attack_family=family,
                    difficulty=difficulty,
                    rounds=rounds,
                    seed=seed,
                )
                if result is None:
                    return jsonify(
                        {
                            "error": "No recorded replay matches this selection.",
                            "hint": "Add a curated artifact for this family, difficulty, and round count.",
                        }
                    ), 404
                if progress_id:
                    _set_run_progress(
                        progress_id,
                        {
                            "status": "completed",
                            "stage": "completed",
                            "headline": "Recorded replay loaded",
                            "detail": "No model was called. This is a precomputed bounded agent run.",
                            "round_number": rounds,
                            "total_rounds": rounds,
                            "run_id": result.get("run_id"),
                            "duration_ms": 0,
                        },
                    )
                return jsonify(result)

            if progress_id:
                _set_run_progress(
                    progress_id,
                    {
                        "status": "running",
                        "stage": "preparing",
                        "headline": "Preparing the synthetic payment arena",
                        "detail": "The server accepted the battle and is validating its boundaries.",
                        "round_number": 1,
                        "total_rounds": rounds,
                    },
                )

            trace(
                "web.run.accepted",
                "The web API validated the requested battle settings.",
                attack_family=family,
                difficulty=difficulty,
                rounds=rounds,
                seed=seed,
            )
            config = AgentLabConfig.from_env()
            progress_callback = (
                (lambda update: _set_run_progress(progress_id, update))
                if progress_id is not None
                else None
            )
            result = SentinelLoopOrchestrator(
                config=config,
                progress_callback=progress_callback,
            ).run(
                attack_family=family,
                difficulty=difficulty,
                rounds=rounds,
                seed=seed,
                include_legitimate_controls=True,
                include_ambient=config.include_ambient_evaluation,
                ambient_sample=config.ambient_sample,
                trap_sample_each=config.trap_sample_each,
                training_log_path=config.training_log_path or None,
                retrain_every=config.retrain_every or None,
            )
            payload = result.to_dict()
            payload["learning_loop"] = _record_learning_loop(config)
            AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
            temporary_path = AGENT_RUNS_DIR / f".{result.run_id}.tmp"
            temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary_path.replace(LATEST_AGENT_RUN_PATH)
            trace(
                "web.run.saved",
                "The completed audit record was saved for the Agent Arena.",
                run_id=result.run_id,
                output_path=str(LATEST_AGENT_RUN_PATH.relative_to(PROJECT_ROOT)),
            )
            return jsonify(payload)
        except RuntimeError as error:
            if progress_id:
                _set_run_progress(
                    progress_id,
                    {
                        "status": "error",
                        "stage": "error",
                        "headline": "The battle stopped safely",
                        "detail": str(error),
                    },
                )
            return jsonify(
                {
                    "error": str(error),
                    "hint": (
                        "Verify endpoint reachability and structured-output compatibility. The "
                        "default profile already uses direct structured generation with reasoning "
                        "effort set to none. If a local model still times out, increase "
                        "MODEL_TIMEOUT_SECONDS or use a smaller model."
                    ),
                }
            ), 502
        except ValueError as error:
            if progress_id:
                _set_run_progress(
                    progress_id,
                    {
                        "status": "error",
                        "stage": "error",
                        "headline": "A safety or policy contract stopped the battle",
                        "detail": str(error),
                    },
                )
            return jsonify(
                {
                    "error": str(error),
                    "hint": (
                        "An agent output did not satisfy a bounded safety or decision contract; "
                        "the episode was stopped before an unsafe action could be applied."
                    ),
                }
            ), 422
        finally:
            AGENT_RUN_LOCK.release()

    @app.get("/api/dashboard")
    def dashboard() -> Any:
        events, predictions, metrics = _load_data()
        test_metrics = metrics["test_metrics"]
        confusion = test_metrics["confusion_matrix"]
        payload = {
            "summary": {
                "events": len(events),
                "fraud_events": int(events["label_fraud"].sum()),
                "campaigns": int(events["scenario_id"].nunique()),
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "false_positive_rate": test_metrics["false_positive_rate"],
                "threshold": metrics["threshold_selected_on_validation"],
            },
            "timeline": _timeline(events),
            "risk_histogram": _risk_histogram(predictions),
            "confusion": confusion,
            "attack_results": metrics.get("attack_family_results", {}),
            "legitimate_results": metrics.get("legitimate_profile_results", {}),
            "campaigns": _campaigns(events),
            "network": _network(events),
            "limitations": metrics.get("limitations", []),
        }
        return jsonify(payload)

    @app.get("/api/transactions")
    def transactions() -> Any:
        _, predictions, _ = _load_data()
        kind = request.args.get("kind", "all")
        family = request.args.get("family", "all")
        if kind == "false_positive":
            filtered = predictions[(~predictions["label_fraud"]) & predictions["predicted_fraud"]]
        elif kind == "missed_fraud":
            filtered = predictions[predictions["label_fraud"] & (~predictions["predicted_fraud"])]
        elif kind == "detected_fraud":
            filtered = predictions[predictions["label_fraud"] & predictions["predicted_fraud"]]
        else:
            filtered = predictions
        if family != "all":
            filtered = filtered[filtered["attack_family"].eq(family)]
        filtered = filtered.sort_values("risk_score", ascending=False).head(100).copy()
        filtered["event_ts"] = filtered["event_ts"].dt.strftime("%d %b %Y, %H:%M:%S")
        columns = [
            "event_id", "event_ts", "attack_family", "scenario_id", "legitimate_control",
            "risk_score", "label_fraud", "predicted_fraud",
        ]
        filtered = filtered.where(pd.notna(filtered), None)
        return jsonify({"rows": filtered[columns].to_dict(orient="records"), "returned": len(filtered)})

    @app.errorhandler(FileNotFoundError)
    def missing_data(error: FileNotFoundError) -> Any:
        return jsonify({"error": str(error)}), 503

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8501")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
