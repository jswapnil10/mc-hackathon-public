from __future__ import annotations

import io
import json
import unittest
import urllib.error
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
    def _call(self, config: AgentLabConfig, agent_name: str = "blue_event_agent"):
        return OpenAICompatibleGateway(config).generate_json(
            agent_name=agent_name,
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
    def test_global_override_can_disable_reasoning(self, urlopen) -> None:
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
    def test_role_aware_reasoning_profiles(self, urlopen) -> None:
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
        config = AgentLabConfig(reasoning_effort="auto")
        self._call(config, "red_planner")
        red_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(red_payload["reasoning_effort"], "none")
        self._call(config, "blue_event_agent")
        blue_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(blue_payload["reasoning_effort"], "none")
        self._call(config, "blue_strategist")
        strategy_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(strategy_payload["reasoning_effort"], "none")

    @patch("sentinelloop.model_gateway.urllib.request.urlopen")
    def test_retries_without_thinking_when_reasoning_exhausts_output(self, urlopen) -> None:
        reasoning_only = _FakeResponse(
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
        final_content = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        urlopen.side_effect = [reasoning_only, final_content]
        result, trace = self._call(
            AgentLabConfig(reasoning_effort="auto", red_reasoning_effort="medium"),
            "red_planner",
        )
        first = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        second = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(first["reasoning_effort"], "medium")
        self.assertEqual(second["reasoning_effort"], "none")
        self.assertEqual(result, {"ok": True})
        self.assertIn("reasoning_exhausted_retry_without_thinking", trace.compatibility_fallbacks)

    @patch("sentinelloop.model_gateway.urllib.request.urlopen")
    def test_reasoning_timeout_retries_direct_generation_with_time_remaining(self, urlopen) -> None:
        final_content = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        urlopen.side_effect = [TimeoutError("timed out"), final_content]

        result, trace = self._call(
            AgentLabConfig(reasoning_effort="auto", red_reasoning_effort="medium"),
            "red_planner",
        )

        first = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        second = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], 45)
        self.assertEqual(first["reasoning_effort"], "medium")
        self.assertEqual(second["reasoning_effort"], "none")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(trace.reasoning_effort, "none")
        self.assertIn(
            "reasoning_timeout_retry_without_thinking",
            trace.compatibility_fallbacks,
        )

    @patch("sentinelloop.model_gateway.urllib.request.urlopen")
    def test_omits_reasoning_for_endpoints_that_do_not_support_it(self, urlopen) -> None:
        unsupported = urllib.error.HTTPError(
            url="https://model.example/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"reasoning_effort is not supported by this model"}'),
        )
        final_content = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        urlopen.side_effect = [unsupported, final_content]
        result, trace = self._call(AgentLabConfig(reasoning_effort="auto"))
        first = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        second = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(first["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", second)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(trace.reasoning_effort, "omitted")
        self.assertIn("reasoning_parameter_omitted", trace.compatibility_fallbacks)


if __name__ == "__main__":
    unittest.main()
