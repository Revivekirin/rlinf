"""Explicit, simulator-only PLUS reset reproducibility smoke test.

This script is never imported or run by sampling/collector unit tests.
Run it on the evaluation host only after explicitly authorizing a smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fingerprint(raw: dict, sim_flat) -> dict:
    fields = {}
    for name, value in raw.items():
        if name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos") or (
            not name.startswith("robot") and
            ("pos" in name or "quat" in name or "object-state" in name)
        ):
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number):
                fields[name] = {"shape": list(array.shape),
                                "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                                "values": array.tolist()}
    image = np.ascontiguousarray(raw["agentview_image"])
    sim_flat = np.ascontiguousarray(np.asarray(sim_flat))
    return {"sim_state_flat": {"shape": list(sim_flat.shape),
                               "sha256": hashlib.sha256(sim_flat.tobytes()).hexdigest(),
                               "values": sim_flat.tolist()},
            "camera_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "camera_shape": list(image.shape),
            "observed_physical_fields": fields}


def differences(a: dict, b: dict) -> list[str]:
    changed = []
    if a["sim_state_flat"]["sha256"] != b["sim_state_flat"]["sha256"]:
        changed.append("sim_state_flat")
    if a["camera_sha256"] != b["camera_sha256"]:
        changed.append("agentview_image")
    for key in sorted(set(a["observed_physical_fields"]) | set(b["observed_physical_fields"])):
        if (a["observed_physical_fields"].get(key, {}).get("sha256") !=
            b["observed_physical_fields"].get(key, {}).get("sha256")):
            changed.append(key)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2:
        parser.error("At least two repeated resets are required")
    root = args.research_root.resolve()
    import os
    if os.environ.get("LIBERO_TYPE") != "plus" or os.environ.get("LIBERO_SUFFIX") != "none":
        parser.error("Set LIBERO_TYPE=plus and LIBERO_SUFFIX=none")
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from scripts.benchmark.plus_evaluation import load_selection
    from scripts.benchmark.plus_inventory import SUITE_COUNTS
    from scripts.evaluate.plus_collect import CONDITIONS
    classification_path = root / "upstream/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    map_path = root / "upstream/LIBERO-plus/libero/libero/benchmark/libero_suite_task_map.py"
    classification = json.loads(classification_path.read_text())
    manifest, selection = load_selection(args.task_manifest, classification,
                                         classification_path, map_path)
    report = {"manifest_content_sha256": manifest["manifest_content_sha256"],
              "repeats": args.repeats,
              "qpos_qvel_individually_available": False,
              "state_access_note": "RLinf worker exposes get_sim_state() as a full flattened simulator state; qpos and qvel are not separately exposed. Object poses are reported only when named numeric observation fields exist.",
              "model_loading_executed": False,
              "condition_path_note": "All four condition labels execute the collector's identical environment configuration and reset call. Policy loading/inference is not part of this smoke test.",
              "suites": {}}
    for suite in SUITE_COUNTS:
        task_id = selection[suite][0]
        config = OmegaConf.load(root / "upstream/RLinf/examples/embodiment/config/env/libero_10.yaml")
        config.task_suite_name = suite
        config.total_num_envs = 1
        config.is_eval = True
        config.auto_reset = False
        config.ignore_terminations = False
        config.reset_gripper_open = False
        config.use_ordered_reset_state_ids = True
        config.max_episode_steps = 520
        config.max_steps_per_rollout_epoch = 520
        config.video_cfg.save_video = False
        env = LiberoEnv(config, num_envs=1, seed_offset=0,
                        total_num_processes=1, worker_info=None)
        samples = []
        saved = None
        try:
            reset_id = int(env.cumsum_trial_id_bins[task_id - 1]) if task_id else 0
            for condition in CONDITIONS:
                for repeat in range(args.repeats):
                    env._is_start = False
                    env.reset(env_idx=np.array([0]), reset_state_ids=np.array([reset_id]))
                    if int(env.task_ids[0]) != task_id or int(env.trial_ids[0]) != 0:
                        raise ValueError("Reset selected a different task/trial")
                    raw = env.current_raw_obs[0]
                    state = fingerprint(raw, env.env.get_sim_state()[0])
                    samples.append({"condition": condition, "repeat": repeat,
                                    "fingerprint": state})
                    # The collector seeds policy RNG after the environment reset.
                    policy_seed = 100000 * list(SUITE_COUNTS).index(suite) + task_id
                    random.seed(policy_seed)
                    np.random.seed(policy_seed)
            # Separate validation path: apply the saved init state only here,
            # then perform the same 15 zero-action settling steps.
            saved_state = env.task_suite.get_task_init_states(task_id)[0]
            env._is_start = False
            env.reset(env_idx=np.array([0]), reset_state_ids=np.array([reset_id]))
            env.env.workers[0].set_init_state(saved_state)
            for _ in range(15):
                raw_batch, _reward, _done, _info = env.env.step(np.zeros((1, 7)))
            saved = fingerprint(raw_batch[0], env.env.get_sim_state()[0])
        finally:
            env.env.close()
        reference = samples[0]["fingerprint"]
        comparisons = [{"condition": s["condition"], "repeat": s["repeat"],
                        "changed_fields": differences(reference, s["fingerprint"])}
                       for s in samples[1:]]
        saved_diff = differences(reference, saved)
        report["suites"][suite] = {
            "task_id_zero_based": task_id,
            "task_id_official": task_id + 1,
            "repeated_reset_comparisons": comparisons,
            "repeated_physical_state_equal": all("sim_state_flat" not in x["changed_fields"]
                                                 and not any(f not in ("agentview_image",)
                                                             for f in x["changed_fields"])
                                                 for x in comparisons),
            "repeated_camera_equal": all("agentview_image" not in x["changed_fields"]
                                         for x in comparisons),
            "saved_init_vs_current_changed_fields": saved_diff,
            "saved_init_physical_equal": "sim_state_flat" not in saved_diff,
            "saved_init_camera_equal": "agentview_image" not in saved_diff,
            "accessible_observation_fields": sorted(reference["observed_physical_fields"]),
        }
    report["paired_reset_reproducibility"] = all(
        r["repeated_physical_state_equal"] and r["repeated_camera_equal"]
        for r in report["suites"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
