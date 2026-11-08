"""Persistence tests do not need RLinf, a GPU, or a simulator."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate.nominal_storage import read_complete, summarize_run, write_episode


class NominalStorageTests(unittest.TestCase):
    def test_complete_episode_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metadata = {
                "complete": True,
                "episode_id": "libero_goal-t00-i00",
                "protocol_hash": "p1",
                "suite": "libero_goal",
                "task_id": 0,
                "trial_id": 0,
                "success": True,
                "episode_length": 2,
            }
            path = write_episode(
                run_dir,
                metadata,
                {"robot0_eef_pos": np.zeros((3, 3))},
                {"executed_actions": np.zeros((2, 7))},
                {"main_image_policy": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            )
            self.assertEqual(read_complete(path, protocol_hash="p1"), metadata)
            summary = summarize_run(run_dir, {metadata["episode_id"]}, "p1")
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(json.loads((run_dir / "summary.json").read_text()), summary)
            with self.assertRaises(FileExistsError):
                write_episode(run_dir, metadata, {"x": np.zeros(1)},
                              {"x": np.zeros(1)}, {"x": np.zeros(1)})

    def test_partial_is_not_counted_and_wrong_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            partial = run_dir / "rollouts" / ".partial-test"
            partial.mkdir(parents=True)
            expected = {"libero_goal-t00-i00"}
            summary = summarize_run(run_dir, expected, "p1")
            self.assertEqual(summary["completed"], 0)
            self.assertEqual(summary["missing_episode_ids"], sorted(expected))
            metadata = {"complete": True, "episode_id": next(iter(expected)),
                        "protocol_hash": "p2", "suite": "libero_goal",
                        "success": False, "episode_length": 1}
            write_episode(run_dir, metadata, {"x": np.zeros(1)},
                          {"x": np.zeros(1)}, {"x": np.zeros(1)})
            with self.assertRaisesRegex(ValueError, "Protocol hash mismatch"):
                summarize_run(run_dir, expected, "p1")


if __name__ == "__main__":
    unittest.main()
