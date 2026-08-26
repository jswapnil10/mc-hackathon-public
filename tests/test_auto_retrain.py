from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import server
from sentinelloop.config import AgentLabConfig


class AutoRetrainTests(unittest.TestCase):
    def test_successful_battles_increment_an_atomic_local_counter(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            server, "BATTLE_COUNT_PATH", Path(tmp) / "battle_count.txt"
        ):
            self.assertEqual(server._bump_battle_count(), 1)
            self.assertEqual(server._bump_battle_count(), 2)
            self.assertEqual(server._read_battle_count(), 2)
            self.assertFalse((Path(tmp) / "battle_count.tmp").exists())

    def test_fifth_battle_schedules_one_non_blocking_retrain(self):
        config = AgentLabConfig(retrain_every_battles=5)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            server, "BATTLE_COUNT_PATH", Path(tmp) / "battle_count.txt"
        ), patch("app.server.threading.Thread") as thread:
            outcomes = [server._record_learning_loop(config) for _ in range(5)]
        self.assertTrue(all(not item["retrain_scheduled"] for item in outcomes[:4]))
        self.assertTrue(outcomes[4]["retrain_scheduled"])
        self.assertEqual(outcomes[4]["battle_number"], 5)
        thread.assert_called_once()
        self.assertTrue(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once_with()

    def test_disabled_ml_counts_battles_without_scheduling_retrain(self):
        config = AgentLabConfig(ml_detector_enabled=False, retrain_every_battles=5)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            server, "BATTLE_COUNT_PATH", Path(tmp) / "battle_count.txt"
        ), patch("app.server.threading.Thread") as thread:
            outcomes = [server._record_learning_loop(config) for _ in range(5)]
        self.assertFalse(outcomes[-1]["enabled"])
        self.assertFalse(outcomes[-1]["retrain_scheduled"])
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
