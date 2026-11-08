"""Recompute and validate a completed or interrupted nominal run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nominal_storage import episode_id, read_complete, summarize_run

SUITES = ("libero_goal", "libero_spatial", "libero_10")


def validate_episode(path: Path, protocol_hash: str) -> dict:
    metadata = read_complete(path, protocol_hash=protocol_hash)
    with np.load(path / "states.npz", allow_pickle=False) as states, \
         np.load(path / "actions.npz", allow_pickle=False) as actions, \
         np.load(path / "observations.npz", allow_pickle=False) as observations:
        length = int(metadata["episode_length"])
        queries = int(metadata["num_queries"])
        assert states["robot0_eef_pos"].shape == (length + 1, 3)
        assert states["robot0_eef_quat"].shape == (length + 1, 4)
        assert actions["executed_actions"].shape == (length, 7)
        assert actions["decoded_action_chunks"].shape == (queries, 8, 7)
        assert actions["raw_action_tokens"].shape == (queries, 8, 7)
        assert observations["main_image_policy"].shape[0] == queries
        assert observations["main_image_raw"].shape[0] == queries
        assert observations["pixel_values"].shape[0] == queries
        assert observations["input_ids"].shape[0] == queries
        assert observations["attention_mask"].shape[0] == queries
        query_ids = actions["executed_query_index"]
        slots = actions["executed_action_slot"]
        assert len(query_ids) == len(slots) == length
        assert np.array_equal(actions["simulator_step"], np.arange(length))
        assert np.all((query_ids >= 0) & (query_ids < queries))
        assert np.all((slots >= 0) & (slots < 8))
        assert np.array_equal(observations["query_step"],
                              np.r_[0, np.flatnonzero(np.diff(query_ids)) + 1])
        for step in range(length):
            assert np.allclose(actions["executed_actions"][step, :6],
                               actions["decoded_action_chunks"][query_ids[step], slots[step], :6])
            decoded_gripper = actions["decoded_action_chunks"][query_ids[step], slots[step], 6]
            assert np.allclose(actions["executed_actions"][step, 6],
                               -np.sign(2 * decoded_gripper - 1))
        for key in states.files:
            assert len(states[key]) == length + 1
            assert np.isfinite(states[key]).all()
        assert np.isfinite(actions["executed_actions"]).all()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    protocol_hash = manifest["protocol_hash"]
    trials = int(manifest["protocol"]["trials_per_task"])
    task_count = 1 if trials == 1 else 10
    expected = {episode_id(s, t, i) for s in SUITES for t in range(task_count) for i in range(trials)}
    summary = summarize_run(args.run_dir, expected, protocol_hash)
    for path in sorted((args.run_dir / "rollouts").iterdir()):
        if path.is_dir() and not path.name.startswith(".partial-"):
            validate_episode(path, protocol_hash)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_episode_ids"]:
        raise SystemExit(f"Incomplete run: {len(summary['missing_episode_ids'])} episodes missing")


if __name__ == "__main__":
    main()
