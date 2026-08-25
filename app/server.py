"""Flask web prototype for the agentic lab, with the v1 dashboard preserved."""

from __future__ import annotations

import json
import os
import threading
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
AGENT_RUNS_DIR = PROJECT_ROOT / "runs" / "agentic"
LATEST_AGENT_RUN_PATH = AGENT_RUNS_DIR / "latest.json"
AGENT_RUN_LOCK = threading.Lock()
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
LATEST_BENCHMARK_PATH = BENCHMARK_DIR / "latest.json"
BENCHMARK_RUN_LOCK = threading.Lock()
EXTERNAL_VALIDATION_DIR = PROJECT_ROOT / "data" / "external_validation"
LATEST_EXTERNAL_VALIDATION_PATH = EXTERNAL_VALIDATION_DIR / "latest.json"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = [path for path in (EVENTS_PATH, PREDICTIONS_PATH, METRICS_PATH) if not path.exists()]
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing dashboard data: {names}. Run the simulator and baseline commands first."
        )
    events = pd.read_csv(EVENTS_PATH)
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["event_ts"] = pd.to_datetime(predictions["event_ts"], utc=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
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

    @app.get("/legacy")
    def legacy_dashboard() -> str:
        return render_template("index.html")

    @app.get("/api/v2/status")
    def agent_status() -> Any:
        config = AgentLabConfig.from_env()
        catalog = AttackCatalog()
        atlas = ThreatAtlas()
        return jsonify(
            {
                "system": "MasterGuard AI - Attack. Adapt. Defend.",
                "mode": "live_open_model",
                "architecture": (
                    "Red GenAI → Synthetic Arena → Fast Sequence Guard + Blue GenAI → Deterministic Referee "
                    "→ guarded Red/Blue feedback loops"
                ),
                "models": {
                    "red": config.red_model_id,
                    "blue": config.blue_model_id,
                    "blue_strategist": config.blue_model_id,
                    "referee": "deterministic-policy-v2",
                },
                "model_endpoint": "server-side and private",
                "structured_output_mode": config.structured_output_mode,
                "latency_profile": {
                    "blue_model_calls_per_event": 1,
                    "case_parallelism": config.case_parallelism,
                    "recommended_demo_rounds": 1,
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

    @app.post("/api/v2/run")
    def run_agent_lab() -> Any:
        if not AGENT_RUN_LOCK.acquire(blocking=False):
            return jsonify({"error": "Another Agent Arena run is already in progress."}), 409
        try:
            body = request.get_json(silent=True) or {}
            family = str(body.get("attack_family", "ATO-01"))
            difficulty = str(body.get("difficulty", "medium"))
            rounds = int(body.get("rounds", 1))
            seed = int(body.get("seed", 20260824))
            if family not in AttackCatalog().families:
                return jsonify({"error": f"Unknown attack family {family!r}."}), 400
            if difficulty not in {"easy", "medium", "hard"}:
                return jsonify({"error": "Difficulty must be easy, medium, or hard."}), 400
            if not 1 <= rounds <= 3:
                return jsonify({"error": "The web demo supports between 1 and 3 rounds."}), 400

            trace(
                "web.run.accepted",
                "The web API validated the requested battle settings.",
                attack_family=family,
                difficulty=difficulty,
                rounds=rounds,
                seed=seed,
            )
            result = SentinelLoopOrchestrator().run(
                attack_family=family,
                difficulty=difficulty,
                rounds=rounds,
                seed=seed,
                include_legitimate_controls=True,
            )
            payload = result.to_dict()
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
            return jsonify(
                {
                    "error": str(error),
                    "hint": "Verify that the configured Qwen/vLLM endpoint is reachable from the web service.",
                }
            ), 502
        except ValueError as error:
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
