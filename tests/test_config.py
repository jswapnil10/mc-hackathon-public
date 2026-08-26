from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sentinelloop.config import AgentLabConfig, LOCAL_OLLAMA_BASE_URL


class AgentLabConfigTests(unittest.TestCase):
    @patch("sentinelloop.config._detect_local_ollama_qwen")
    def test_uses_detected_local_qwen_when_environment_is_empty(self, detect):
        detect.return_value = (LOCAL_OLLAMA_BASE_URL, "qwen3-coder:latest")
        with patch.dict(os.environ, {}, clear=True):
            config = AgentLabConfig.from_env()
        self.assertEqual(config.model_base_url, LOCAL_OLLAMA_BASE_URL)
        self.assertEqual(config.red_model_id, "qwen3-coder:latest")
        self.assertEqual(config.blue_model_id, "qwen3-coder:latest")
        self.assertEqual(config.reasoning_effort, "auto")
        self.assertEqual(config.reasoning_effort_for("red_planner"), "medium")
        self.assertEqual(config.reasoning_effort_for("blue_event_agent"), "none")
        self.assertEqual(config.reasoning_effort_for("blue_strategist"), "medium")
        self.assertTrue(config.ml_detector_enabled)
        self.assertEqual(config.retrain_every, 0)

    @patch("sentinelloop.config._detect_local_ollama_qwen")
    def test_explicit_environment_overrides_local_detection(self, detect):
        with patch.dict(
            os.environ,
            {
                "MODEL_BASE_URL": "https://private-model.example/v1",
                "RED_MODEL_ID": "custom-red",
                "BLUE_MODEL_ID": "custom-blue",
                "MODEL_REASONING_EFFORT": "low",
            },
            clear=True,
        ):
            config = AgentLabConfig.from_env()
        detect.assert_not_called()
        self.assertEqual(config.model_base_url, "https://private-model.example/v1")
        self.assertEqual(config.red_model_id, "custom-red")
        self.assertEqual(config.blue_model_id, "custom-blue")
        self.assertEqual(config.reasoning_effort, "low")

    @patch("sentinelloop.config._detect_local_ollama_qwen")
    def test_blue_hybrid_settings_are_explicit_and_do_not_change_red(self, detect):
        with patch.dict(
            os.environ,
            {
                "MODEL_BASE_URL": "https://private-model.example/v1",
                "RED_MODEL_ID": "red-qwen",
                "BLUE_MODEL_ID": "blue-qwen",
                "ML_DETECTOR_ENABLED": "false",
                "BATTLE_RETRAIN_EVERY": "3",
                "BATTLE_INCLUDE_AMBIENT": "true",
            },
            clear=True,
        ):
            config = AgentLabConfig.from_env()
        detect.assert_not_called()
        self.assertEqual(config.red_model_id, "red-qwen")
        self.assertFalse(config.ml_detector_enabled)
        self.assertEqual(config.retrain_every, 3)
        self.assertTrue(config.include_ambient_evaluation)


if __name__ == "__main__":
    unittest.main()
