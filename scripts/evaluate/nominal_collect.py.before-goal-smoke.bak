"""Single-GPU, evaluation-only LIBERO nominal trajectory collector.

Run from the research checkout with EMBODIED_PATH and PYTHONPATH pointing at
the pinned RLinf checkout. The user-facing protocol is in docs/NOMINAL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from nominal_storage import episode_id, summarize_run, write_episode

UPSTREAM_COMMIT = "e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d"
NORM_KEY = "libero_130_no_noops_trajall"
SUITES = ("libero_goal", "libero_spatial", "libero_10")
CONDITIONS = {
    "sft_deterministic": ("sft", False, 1.0, True),
    "sft_base_only_deterministic": ("sft", False, 1.0, False),
    "grpo_deterministic": ("grpo", False, 1.0, False),
    "grpo_stochastic": ("grpo", True, 1.6, False),
    "sft_stochastic": ("sft", True, 1.6, True),
}
MODELS = {
    "sft": ("RLinf-OpenVLAOFT-LIBERO-130-Base-Lora", "RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora", "a4ea91b3897648f3c1395e5221edb40b8f53d1a4"),
    "grpo": ("RLinf-OpenVLAOFT-LIBERO-130", "RLinf/RLinf-OpenVLAOFT-LIBERO-130", "473954241c8148bc9a1bf9ed547a74523e4a7d41"),
}


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_upstream(upstream: Path) -> None:
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f"RLinf commit mismatch: {actual}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("RLinf upstream checkout is dirty")


def check_mapping(mapping: dict, suites: tuple[str, ...], trials: int) -> dict:
    if mapping.get("schema_version") != 1 or mapping.get("trials_compared", 0) < trials:
        raise ValueError("Mapping audit is absent or compares too few init states")
    rows = {(r["suite"], r["subset_task_id"]): r for r in mapping["tasks"]}
    for suite in suites:
        for task_id in range(10):
            row = rows.get((suite, task_id))
            if row is None or len(row["trial_sha256"]) < trials:
                raise ValueError(f"Missing audited mapping: {suite}/{task_id}")
    return rows


def check_checkpoint_inventory(root: Path, checkpoint: Path, repo: str, revision: str) -> None:
    inventory = json.loads((root / "manifests" / "models_130.json").read_text())
    entries = [item for item in inventory["models"]
               if item["repo_id"] == repo and item["revision"] == revision]
    if len(entries) != 1:
        raise ValueError(f"Checkpoint inventory identity mismatch: {repo}@{revision}")
    for item in entries[0]["files"]:
        path = checkpoint / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Missing or size-mismatched checkpoint file: {path}")


def array_from_tensor(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.to(dtype=__import__("torch").float32)
        return value.numpy()
    return np.asarray(value)


def finite_array(value, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"Invalid numeric array: {label}")
    return array.copy()


def capture_state(raw: dict) -> dict[str, np.ndarray]:
    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    result = {key: finite_array(raw[key], key) for key in required}
    if result["robot0_eef_pos"].shape != (3,) or result["robot0_eef_quat"].shape != (4,):
        raise ValueError("Unexpected EEF pose shape")
    for key, value in raw.items():
        if key in result or "image" in key or "depth" in key:
            continue
        array = np.asarray(value)
        if array.dtype.kind in "biuf" and array.size < 100000:
            result[key] = finite_array(array, key)
    return result


def stack_states(states: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(states[0])
    if any(set(item) != keys for item in states):
        raise ValueError("Simulator observation keys changed within an episode")
    return {key: np.stack([item[key] for item in states]) for key in sorted(keys)}


def collect_episode(env, model, *, suite: str, task_id: int, trial_id: int,
                    sampling: bool, temperature: float, policy_seed: int,
                    metadata: dict, run_dir: Path) -> Path:
    import torch

    reset_id = (int(env.cumsum_trial_id_bins[task_id - 1]) if task_id else 0) + trial_id
    env._is_start = False  # Keep the explicit audited initial-state ID.
    obs, _ = env.reset(env_idx=np.array([0]), reset_state_ids=np.array([reset_id]))
    if int(env.task_ids[0]) != task_id or int(env.trial_ids[0]) != trial_id:
        raise ValueError("LIBERO selected a different task or init-state ID")
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)

    states = [capture_state(env.current_raw_obs[0])]
    queries = []
    tokens = []
    decoded = []
    executed = []
    query_index = []
    action_slot = []
    simulator_step = []
    done = False
    reason = "horizon"
    while not done and len(executed) < int(env.cfg.max_episode_steps):
        step_at_query = len(executed)
        raw = env.current_raw_obs[0]
        query = {
            "step": step_at_query,
            "main_image_raw": np.asarray(raw["agentview_image"]).copy(),
            "main_image_policy": array_from_tensor(obs["main_images"][0]).copy(),
            "proprio_policy": array_from_tensor(obs["states"][0]).copy(),
        }
        if int(model.vision_backbone.get_num_images_in_input()) > 1:
            query["wrist_image_policy"] = array_from_tensor(obs["wrist_images"][0]).copy()
        with torch.inference_mode():
            actions, result = model.predict_action_batch(
                env_obs={k: v.clone() if hasattr(v, "clone") else list(v)
                         for k, v in obs.items()},
                do_sample=sampling, temperature=temperature, top_k=-1, top_p=1.0,
                calculate_logprobs=False, calculate_values=False,
            )
        chunk = finite_array(array_from_tensor(actions)[0], "decoded actions")
        action_tokens = array_from_tensor(result["forward_inputs"]["action_tokens"])[0].copy()
        if chunk.shape != (8, 7) or action_tokens.shape != (8, 7):
            raise ValueError("Unexpected action chunk/token shape")
        query["pixel_values"] = array_from_tensor(result["forward_inputs"]["pixel_values"])[0].copy()
        query["input_ids"] = array_from_tensor(result["forward_inputs"]["input_ids"])[0].copy()
        query["attention_mask"] = array_from_tensor(result["forward_inputs"]["attention_mask"])[0].copy()
        queries.append(query)
        tokens.append(action_tokens)
        decoded.append(chunk)
        # prepare_actions_for_libero mutates its input, hence the explicit copy.
        from rlinf.envs.action_utils import prepare_actions_for_libero
        prepared = prepare_actions_for_libero(chunk[None].copy(), "openvla_oft")[0]
        for slot, action in enumerate(prepared):
            if len(executed) >= int(env.cfg.max_episode_steps):
                break
            next_obs, _, termination, truncation, info = env.step(action[None].copy(), auto_reset=False)
            executed.append(action.copy())
            query_index.append(len(queries) - 1)
            action_slot.append(slot)
            simulator_step.append(len(executed) - 1)
            states.append(capture_state(env.current_raw_obs[0]))
            done = bool(array_from_tensor(termination)[0]) or bool(array_from_tensor(truncation)[0])
            if done:
                reason = "success" if bool(array_from_tensor(termination)[0]) else "horizon"
                break
        if not done:
            obs = next_obs
    if len(states) != len(executed) + 1 or len(executed) > 520:
        raise ValueError("State/action timestep alignment failure")
    if not queries or not executed:
        raise ValueError("Episode contains no policy action")
    success = bool(env.success_once[0])
    if success != (reason == "success"):
        raise ValueError("Official LIBERO success metric disagrees with terminal signal")
    metadata.update(success=success, episode_length=len(executed),
                    termination_reason=reason, num_queries=len(queries),
                    language=str(env.task_descriptions[0]))
    action_arrays = {
        "raw_action_tokens": np.stack(tokens),
        "decoded_action_chunks": np.stack(decoded),
        "executed_actions": np.stack(executed),
        "executed_query_index": np.asarray(query_index, dtype=np.int32),
        "executed_action_slot": np.asarray(action_slot, dtype=np.int16),
        "simulator_step": np.asarray(simulator_step, dtype=np.int32),
    }
    observation_arrays = {"query_step": np.asarray([q["step"] for q in queries], dtype=np.int32)}
    for key in queries[0]:
        if key != "step":
            observation_arrays[key] = np.stack([q[key] for q in queries])
    return write_episode(run_dir, metadata, stack_states(states), action_arrays, observation_arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("LIBERO_TYPE", "standard") != "standard":
        parser.error("Only original LIBERO is supported")
    root = args.research_root.resolve()
    upstream = root / "upstream" / "RLinf"
    check_upstream(upstream)
    if not __file__.startswith(str(root)):
        raise ValueError("Collector must run from the research checkout")
    if not args.run_id or "/" in args.run_id or args.run_id.startswith("."):
        parser.error("Invalid run ID")
    mapping = json.loads(args.mapping.read_text())
    trials = 1 if args.smoke else 5
    rows = check_mapping(mapping, SUITES, trials)
    checkpoint_kind, sampling, temperature, load_lora = CONDITIONS[args.condition]
    checkpoint_dir, checkpoint_repo, checkpoint_revision = MODELS[checkpoint_kind]
    checkpoint = root / "checkpoints" / checkpoint_dir
    check_checkpoint_inventory(root, checkpoint, checkpoint_repo, checkpoint_revision)
    for required in ("config.json", "dataset_statistics.json", "model.safetensors.index.json"):
        if not (checkpoint / required).is_file():
            raise FileNotFoundError(checkpoint / required)
    if load_lora and not (checkpoint / "lora_adapter" / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "lora_adapter" / "adapter_model.safetensors")
    norm_stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    if NORM_KEY not in norm_stats:
        raise ValueError(f"Missing normalization statistics: {NORM_KEY}")
    if not Path(os.environ.get("EMBODIED_PATH", "")).resolve() == upstream / "examples" / "embodiment":
        raise ValueError("EMBODIED_PATH does not point to pinned RLinf examples/embodiment")
    if shutil_disk_free(root) < 10 * 2**30:
        raise OSError("Less than 10 GiB free disk space")
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.models import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model_cfg = OmegaConf.load(upstream / "examples/embodiment/config/model/openvla_oft.yaml")
    model_cfg.model_path = str(checkpoint)
    model_cfg.unnorm_key = NORM_KEY
    model_cfg.max_prompt_length = 128
    model_cfg.is_lora = load_lora
    model_cfg.lora_path = str(checkpoint / "lora_adapter") if load_lora else None
    protocol = {
        "schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_repo": checkpoint_repo, "checkpoint_revision": checkpoint_revision,
        "checkpoint_path": str(checkpoint), "condition": args.condition,
        "sampling_mode": "stochastic" if sampling else "deterministic",
        "do_sample": sampling, "temperature": temperature,
        "lora_applied": load_lora,
        "top_k": -1, "top_p": 1.0, "unnorm_key": NORM_KEY,
        "seed": 0, "policy_seed_formula": "100000*subset_suite_index+100*task_id+trial_id",
        "num_steps_wait": 15, "action_chunks": 8, "horizon": 520,
        "trials_per_task": trials, "mapping_sha256": sha256_json(mapping),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
    }
    protocol_hash = sha256_json(protocol)
    run_dir = root / "outputs" / "nominal" / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists; use --resume only after inspecting it: {run_dir}")
    if not run_dir.exists() and args.resume:
        raise FileNotFoundError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = run_dir / "manifest.json"
    manifest = {"protocol_hash": protocol_hash, "protocol": protocol}
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume protocol differs from existing manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    expected = {episode_id(s, t, i) for s in SUITES for t in range(1 if args.smoke else 10) for i in range(trials)}
    summarize_run(run_dir, expected, protocol_hash)
    pending = [item for item in sorted(expected) if not (run_dir / "rollouts" / item).exists()]
    if not pending:
        print("Run already complete")
        return
    model = get_model(model_cfg).to("cuda").eval()
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if core.unnorm_key != NORM_KEY or core.vision_backbone.get_num_images_in_input() != 1:
        raise ValueError("Loaded model normalization key or camera count differs")
    for suite_index, suite in enumerate(SUITES):
        suite_pending = [item for item in pending if item.startswith(suite + "-")]
        if not suite_pending:
            continue
        env_cfg = OmegaConf.load(upstream / "examples/embodiment/config/env/libero_10.yaml")
        env_cfg.task_suite_name = suite
        env_cfg.total_num_envs = 1
        env_cfg.is_eval = True
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = False
        env_cfg.reset_gripper_open = False
        env_cfg.use_ordered_reset_state_ids = True
        env_cfg.max_episode_steps = 520
        env_cfg.max_steps_per_rollout_epoch = 520
        env_cfg.video_cfg.save_video = False
        env = LiberoEnv(env_cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
        try:
            for task_id in range(1 if args.smoke else 10):
                for trial_id in range(trials):
                    eid = episode_id(suite, task_id, trial_id)
                    if eid not in suite_pending:
                        continue
                    row = rows[(suite, task_id)]
                    actual_task = env.task_suite.get_task(task_id)
                    from scripts.diagnostics.audit_libero_mapping import state_hash, task_identity
                    if tuple(row[k] for k in ("task_name", "problem_folder", "bddl_file", "init_states_file")) != task_identity(actual_task):
                        raise ValueError(f"Task identity changed: {eid}")
                    if state_hash(env.task_suite.get_task_init_states(task_id)[trial_id]) != row["trial_sha256"][trial_id]:
                        raise ValueError(f"Init-state hash changed: {eid}")
                    policy_seed = 100000 * suite_index + 100 * task_id + trial_id
                    metadata = {
                        "complete": True, "episode_id": eid, "run_id": args.run_id,
                        "protocol_hash": protocol_hash, "suite": suite, "task_id": task_id,
                        "libero_130_task_id": row["libero_130_task_id"],
                        "task_name": row["task_name"], "trial_id": trial_id,
                        "initial_state_sha256": row["trial_sha256"][trial_id],
                        "simulator_seed": 0, "policy_seed": policy_seed,
                        "checkpoint_repo": checkpoint_repo, "checkpoint_revision": checkpoint_revision,
                        "lora_applied": load_lora,
                        "sampling_mode": protocol["sampling_mode"], "temperature": temperature,
                        "horizon": 520, "wait_steps": 15, "action_chunks": 8,
                        "unnorm_key": NORM_KEY, "upstream_commit": UPSTREAM_COMMIT,
                        "collector_sha256": protocol["collector_sha256"],
                    }
                    path = collect_episode(env, model, suite=suite, task_id=task_id,
                        trial_id=trial_id, sampling=sampling, temperature=temperature,
                        policy_seed=policy_seed, metadata=metadata, run_dir=run_dir)
                    summary = summarize_run(run_dir, expected, protocol_hash)
                    print(f"{path.name}: success={metadata['success']} length={metadata['episode_length']} ({summary['completed']}/{summary['expected']})", flush=True)
        finally:
            env.env.close()
    summary = summarize_run(run_dir, expected, protocol_hash)
    if summary["completed"] != summary["expected"]:
        raise RuntimeError(f"Missing {len(summary['missing_episode_ids'])} episodes")


def shutil_disk_free(path: Path) -> int:
    import shutil
    return shutil.disk_usage(path).free


if __name__ == "__main__":
    main()
