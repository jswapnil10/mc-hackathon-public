import unittest

from app.server import create_app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_template_auto_reload_keeps_ui_assets_in_sync(self):
        self.assertTrue(self.app.config["TEMPLATES_AUTO_RELOAD"])
        self.assertTrue(self.app.jinja_env.auto_reload)

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MasterGuard AI", response.data)
        self.assertIn(b"Attack. Adapt. Defend.", response.data)
        self.assertIn(b"OVERALL DEFENSE SCORE", response.data)
        self.assertIn(b"What happens when you start a battle?", response.data)
        self.assertIn(b"How to read this report", response.data)
        self.assertIn(b"ATTACK STRENGTH \xc3\x97 MONEY LOST", response.data)
        self.assertIn(b"Threat Atlas", response.data)
        self.assertIn(b"Scenario Foundry", response.data)
        self.assertIn(b"Defense Benchmark", response.data)
        self.assertIn(b"LATEST SAVED SUCCESS", response.data)
        self.assertIn(b"Reality-check the defense beyond our simulator", response.data)
        self.assertIn(b"Mastercard Innovation Challenge 2026", response.data)
        self.assertIn(b"Run guide", response.data)
        script = self.client.get("/static/lab.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"CURRENT ATTEMPT FAILED", script.data)

    def test_run_state_guide_explains_every_output_boundary(self):
        with self.client.get("/static/run-guide.html") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"What happens after you click", response.data)
            self.assertIn(b"Red designs a bounded attack plan", response.data)
            self.assertIn(b"The arena materializes payment events", response.data)
            self.assertIn(b"Blue investigates and acts on each event", response.data)
            self.assertIn(b"The Referee opens sealed truth", response.data)
            self.assertIn(b"Blue tests an update", response.data)
            self.assertIn(b"The report is assembled and saved", response.data)
            self.assertIn(b"SIMULATED DETECTION TIME", response.data)

    def test_dashboard_api_has_required_sections(self):
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue({"summary", "timeline", "attack_results", "network"}.issubset(payload))
        self.assertGreater(payload["summary"]["events"], 0)

    def test_transaction_filters(self):
        response = self.client.get("/api/transactions?kind=missed_fraud")
        self.assertEqual(response.status_code, 200)
        rows = response.get_json()["rows"]
        self.assertTrue(all(row["label_fraud"] and not row["predicted_fraud"] for row in rows))

    def test_agent_status_does_not_expose_private_model_endpoint(self):
        response = self.client.get("/api/v2/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertNotIn("model_base_url", payload)
        self.assertEqual(payload["model_endpoint"], "server-side and private")
        self.assertIn("blue_detector_active", payload["models"])
        self.assertEqual(
            payload["latency_profile"]["reasoning_profiles"],
            {
                "red_planner": "none",
                "blue_live_event": "none",
                "blue_between_rounds": "none",
            },
        )
        self.assertEqual(payload["latency_profile"]["model_call_timeout_seconds"], 120)
        self.assertEqual(len(payload["attack_families"]), 9)
        self.assertEqual(payload["submission_profile"]["diversity"]["attack_family_count"], 9)
        self.assertEqual(payload["submission_profile"]["fidelity"]["lab_only_event_types_allowed"], [])
        self.assertGreaterEqual(payload["threat_atlas"]["vector_count"], 30)

    def test_threat_atlas_api_exposes_sources_and_holdouts(self):
        response = self.client.get("/api/v2/threat-atlas")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        payload = response.get_json()
        self.assertGreaterEqual(len(payload["vectors"]), 30)
        self.assertTrue(all(vector["sources"] for vector in payload["vectors"]))
        self.assertTrue(any(vector["novel_holdout"] for vector in payload["vectors"]))

    def test_latest_population_benchmark_is_available(self):
        response = self.client.get("/api/v2/benchmark")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["data_quality"]["status"], "passed")
        self.assertIn("combined_hidden_test", payload["defense"]["metrics"])

    def test_external_validation_is_available_with_evidence_boundaries(self):
        response = self.client.get("/api/v2/external-validation")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        payload = response.get_json()
        self.assertTrue(payload["methodology"]["test_opened_after_selection"])
        self.assertEqual(payload["overall_assessment"], "share_with_caveats")
        self.assertIn("pr_auc_95ci", payload["defense"]["test_metrics"])
        self.assertTrue(payload["limitations"])

    def test_agent_run_rejects_unbounded_web_inputs_before_model_call(self):
        response = self.client.post(
            "/api/v2/run",
            json={"attack_family": "UNBOUNDED-99", "difficulty": "medium", "rounds": 2},
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_run_progress_is_explicit(self):
        response = self.client.get("/api/v2/run-progress/browser-test-0001")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not started", response.get_json()["error"])

    def test_invalid_run_progress_identifier_is_rejected(self):
        response = self.client.get("/api/v2/run-progress/not_valid!")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
