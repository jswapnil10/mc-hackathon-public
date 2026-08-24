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

    @patch("sentinelloop.config._detect_local_ollama_qwen")
    def test_explicit_environment_overrides_local_detection(self, detect):
        with patch.dict(
            os.environ,
            {
                "MODEL_BASE_URL": "https://private-model.example/v1",
                "RED_MODEL_ID": "custom-red",
                "BLUE_MODEL_ID": "custom-blue",
            },
            clear=True,
        ):
            config = AgentLabConfig.from_env()
        detect.assert_not_called()
        self.assertEqual(config.model_base_url, "https://private-model.example/v1")
        self.assertEqual(config.red_model_id, "custom-red")
        self.assertEqual(config.blue_model_id, "custom-blue")


if __name__ == "__main__":
    unittest.main()
