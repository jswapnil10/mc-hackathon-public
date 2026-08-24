import unittest

from app.server import create_app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SentinelLoop", response.data)

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
        self.assertEqual(len(payload["attack_families"]), 6)

    def test_agent_run_rejects_unbounded_web_inputs_before_model_call(self):
        response = self.client.post(
            "/api/v2/run",
            json={"attack_family": "UNBOUNDED-99", "difficulty": "medium", "rounds": 2},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
