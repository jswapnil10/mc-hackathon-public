from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from sentinelloop.config import AgentLabConfig
from sentinelloop.model_gateway import OpenAICompatibleGateway


class _FakeResponse:
    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class ModelGatewayCompatibilityTests(unittest.TestCase):
    def _call(self, config: AgentLabConfig):
        return OpenAICompatibleGateway(config).generate_json(
            agent_name="compatibility_test",
            model="qwen3.5:9b",
            system_prompt="Return JSON.",
            user_payload={"request": "test"},
            schema_name="compatibility",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            temperature=0.0,
            seed=7,
        )

    @patch("sentinelloop.model_gateway.urllib.request.urlopen")
    def test_disables_reasoning_for_contract_bound_json(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        result, _ = self._call(AgentLabConfig(reasoning_effort="none"))
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(result, {"ok": True})

    @patch("sentinelloop.model_gateway.urllib.request.urlopen")
    def test_reports_reasoning_budget_exhaustion_precisely(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "The unfinished private reasoning trace.",
                        },
                        "finish_reason": "length",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(RuntimeError, "MODEL_REASONING_EFFORT=none"):
            self._call(AgentLabConfig(reasoning_effort="high"))


if __name__ == "__main__":
    unittest.main()
