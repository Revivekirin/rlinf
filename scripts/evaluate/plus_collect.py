"""Single-GPU LIBERO-PLUS evaluation using the pinned nominal policy path.

The default artifact is metadata only: 40,120 raw trajectories would exceed
the available evaluation volume. ``--smoke`` retains the nominal raw format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import nominal_collect as nominal
from nominal_storage import write_episode as write_raw_episode

PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CONDITIONS = {
    "sft_stochastic": ("sft", True, 1.6),
    "grpo_stochastic": ("grpo", True, 1.6),
    "sft_deterministic": ("sft", False, 1.0),
    "grpo_deterministic": ("grpo", False, 1.0),
}


def eid(suite: str, task_id: int) -> str:
    return f"{suite}-t{task_id:04d}-i00"


def atomic_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def initial_hash(states: dict, observations: dict) -> str:
    digest = hashlib.sha256()
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        value = np.ascontiguousarray(states[key][0])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    image = np.ascontiguousarray(observations["main_image_raw"][0])
    digest.update(image.tobytes())
    return digest.hexdigest()


def write_plus_episode(run_dir, metadata, states, actions, observations):
    metadata["initial_state_sha256"] = initial_hash(states, observations)
    if metadata["storage_mode"] == "raw":
        return write_raw_episode(run_dir, metadata, states, actions, observations)
    root = run_dir / "rollouts"
    root.mkdir(parents=True, exist_ok=True)
    target = root / metadata["episode_id"]
    if target.exists():
        raise FileExistsError(target)
    temp = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    atomic_json(temp / "metadata.json", metadata)
    if target.exists():
        raise FileExistsError(target)
    os.rename(temp, target)
    return target


def read_episode(path: Path, protocol_hash: str, *,
                 expected_horizon: int | None = None,
                 expected_protocol_version: str | None = None,
                 expected_manifest_hash: str | None = None) -> dict:
    metadata = json.loads((path / "metadata.json").read_text())
    if (not path.is_dir() or metadata.get("complete") is not True
            or metadata.get("episode_id") != path.name
            or metadata.get("protocol_hash") != protocol_hash
            or len(metadata.get("initial_state_sha256", "")) != 64
            or not isinstance(metadata.get("success"), bool)
            or not 1 <= metadata.get("episode_length", 0) <= 520
            or (expected_horizon is not None and
                (metadata.get("horizon") != expected_horizon or
                 metadata["episode_length"] > expected_horizon))
            or (expected_protocol_version is not None and
                metadata.get("protocol_version") != expected_protocol_version)
            or (expected_manifest_hash is not None and
                metadata.get("task_manifest_content_sha256") != expected_manifest_hash)):
        raise ValueError(f"Invalid episode: {path}")
    if metadata["storage_mode"] == "raw":
        from nominal_validate import validate_episode
        validate_episode(path, protocol_hash)
    return metadata


def summarize(run_dir: Path, expected: set[str], protocol_hash: str, *,
              suite_horizons: dict[str, int] | None = None,
              protocol_version: str | None = None,
              task_manifest_hash: str | None = None) -> dict:
    rows = []
    for path in sorted((run_dir / "rollouts").glob("*")):
        if path.name.startswith(".partial-"):
            continue
        if path.name not in expected:
            raise ValueError(f"Unexpected episode: {path}")
        suite = path.name.split("-t", 1)[0]
        rows.append(read_episode(
            path, protocol_hash,
            expected_horizon=suite_horizons[suite] if suite_horizons else None,
            expected_protocol_version=protocol_version,
            expected_manifest_hash=task_manifest_hash))
    seen = {row["episode_id"] for row in rows}
    if len(seen) != len(rows):
        raise ValueError("Duplicate episodes")
    summary = {
        "expected": len(expected), "completed": len(rows),
        "missing": len(expected - seen),
        "successes": sum(row["success"] for row in rows),
        "failures": sum(not row["success"] for row in rows),
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-manifest", type=Path,
                        help="Validated pilot manifest; never changes the legacy default")
    parser.add_argument("--horizon-protocol", choices=("rlinf-520", "lerobot-suite"),
                        default="rlinf-520")
    args = parser.parse_args()
    if args.smoke and args.task_manifest:
        parser.error("--smoke and --task-manifest cannot be combined")
    new_protocol = args.task_manifest is not None or args.horizon_protocol != "rlinf-520"
    if not args.run_id or "/" in args.run_id or args.run_id.startswith("."):
        parser.error("Invalid run ID")
    if os.environ.get("LIBERO_TYPE") != "plus" or os.environ.get("LIBERO_SUFFIX") != "none":
        parser.error("Set LIBERO_TYPE=plus and LIBERO_SUFFIX=none")
    root = args.research_root.resolve()
    if not Path(__file__).resolve().is_relative_to(root):
        raise ValueError("Collector must run from the research checkout")
    upstream = root / "upstream/RLinf"
    nominal.check_upstream(upstream)
    official = root / "upstream/LIBERO-plus"
    from scripts.diagnostics.check_libero_plus_source import inspect
    source = inspect(official, require_assets=False)
    if source["revision"] != PLUS_COMMIT:
        raise ValueError("Official LIBERO-PLUS revision changed")
    kind, sampling, temperature = CONDITIONS[args.condition]
    checkpoint_dir, checkpoint_repo, checkpoint_revision = nominal.MODELS[kind]
    checkpoint = root / "checkpoints" / checkpoint_dir
    nominal.check_checkpoint_inventory(root, checkpoint, checkpoint_repo, checkpoint_revision)
    load_lora = new_protocol and kind == "sft"
    if load_lora and not (checkpoint / "lora_adapter/adapter_model.safetensors").is_file():
        raise FileNotFoundError("SFT LoRA adapter missing")
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    if nominal.NORM_KEY not in stats:
        raise ValueError("LIBERO-130 normalization statistics missing")
    if Path(os.environ.get("EMBODIED_PATH", "")).resolve() != upstream / "examples/embodiment":
        raise ValueError("EMBODIED_PATH mismatch")
    free = shutil.disk_usage(root).free
    if free < 10 * 2**30:
        raise OSError("Less than 10 GiB free disk space")

    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.models import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = OmegaConf.load(upstream / "examples/embodiment/config/model/openvla_oft.yaml")
    cfg.model_path = str(checkpoint)
    cfg.unnorm_key = nominal.NORM_KEY
    cfg.max_prompt_length = 128
    cfg.is_lora = load_lora
    cfg.lora_path = str(checkpoint / "lora_adapter") if load_lora else None
    classification = json.loads((official / "libero/libero/benchmark/task_classification.json").read_text())
    from scripts.benchmark.plus_evaluation import (
        horizon_for, load_selection, prepare_run_directory, resume_protocol)
    selected = (SUITES[0],) if args.smoke else SUITES
    task_manifest = None
    if args.task_manifest:
        task_manifest, task_indices = load_selection(
            args.task_manifest, classification,
            official / "libero/libero/benchmark/task_classification.json",
            official / "libero/libero/benchmark/libero_suite_task_map.py")
    else:
        task_indices = {suite: list(range(1 if args.smoke else len(classification[suite])))
                        for suite in selected}
    expected = {eid(suite, task_id) for suite in selected
                for task_id in task_indices[suite]}
    protocol = {
        "schema_version": 1, "upstream_commit": nominal.UPSTREAM_COMMIT,
        "plus_commit": PLUS_COMMIT, "condition": args.condition,
        "checkpoint_repo": checkpoint_repo, "checkpoint_revision": checkpoint_revision,
        "sampling_mode": "stochastic" if sampling else "deterministic",
        "do_sample": sampling, "temperature": temperature,
        "top_p": 1.0, "top_k": -1, "action_chunks": 8, "action_dim": 7,
        "unnorm_key": nominal.NORM_KEY, "horizon": 520,
        "wait_steps": 15, "lora_applied": load_lora,
        "simulator_seed": 0, "policy_seed": "100000*suite_index+task_id",
        "task_counts": {s: len(classification[s]) if not new_protocol else len(task_indices[s])
                        for s in selected},
        "selected_task_count": len(expected), "storage_mode": "raw" if args.smoke else "metadata",
        "classification_sha256": hashlib.sha256(
            (official / "libero/libero/benchmark/task_classification.json").read_bytes()
        ).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "nominal_collector_sha256": hashlib.sha256(Path(nominal.__file__).read_bytes()).hexdigest(),
        "model_config": OmegaConf.to_container(cfg, resolve=True),
    }
    if new_protocol:
        protocol.update({
            "protocol_version": "plus_pilot_suite_v1" if args.horizon_protocol == "lerobot-suite" else "plus_pilot_520_v1",
            "horizon_protocol": args.horizon_protocol,
            "suite_horizons": {s: horizon_for(args.horizon_protocol, s) for s in selected},
            "selected_task_indices": {s: task_indices[s] for s in selected},
            "selected_episode_ids": sorted(expected),
            "task_manifest_content_sha256": task_manifest["manifest_content_sha256"] if task_manifest else None,
            "initial_state_protocol": "plus_seed0_reset_no_saved_init_state_v1",
        })
        protocol["horizon"] = None
    run_dir = root / "outputs/libero_plus" / args.run_id / args.condition
    prepare_run_directory(run_dir, resume=args.resume)
    if args.resume:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        protocol = resume_protocol(manifest["protocol"], protocol, legacy=not new_protocol)
        protocol_hash = manifest["protocol_hash"]
        if protocol_hash != nominal.sha256_json(protocol):
            raise ValueError("Stored protocol hash mismatch")
    else:
        protocol_hash = nominal.sha256_json(protocol)
        manifest = {"protocol_hash": protocol_hash, "protocol": protocol}
        atomic_json(run_dir / "manifest.json", manifest)
    print(f"preflight: condition={args.condition} expected={len(expected)} "
          f"by_suite={protocol['task_counts']} free_gib={free / 2**30:.1f}", flush=True)
    summary_kwargs = {
        "suite_horizons": protocol.get("suite_horizons"),
        "protocol_version": protocol.get("protocol_version"),
        "task_manifest_hash": protocol.get("task_manifest_content_sha256"),
    }
    summary = summarize(run_dir, expected, protocol_hash, **summary_kwargs)
    if summary["missing"] == 0:
        print("Run already complete", flush=True)
        return
    completed_count = summary["completed"]

    model = get_model(cfg).to("cuda").eval()
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if core.unnorm_key != nominal.NORM_KEY or core.vision_backbone.get_num_images_in_input() != 1:
        raise ValueError("Model normalization key or camera count differs")
    nominal.write_episode = write_plus_episode
    for suite_index, suite in enumerate(selected):
        suite_tasks = task_indices[suite]
        if all((run_dir / "rollouts" / eid(suite, t)).exists()
               for t in suite_tasks):
            continue
        env_cfg = OmegaConf.load(upstream / "examples/embodiment/config/env/libero_10.yaml")
        env_cfg.task_suite_name = suite
        env_cfg.total_num_envs = 1
        env_cfg.is_eval = True
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = False
        env_cfg.reset_gripper_open = False
        env_cfg.use_ordered_reset_state_ids = True
        actual_horizon = horizon_for(args.horizon_protocol, suite)
        env_cfg.max_episode_steps = actual_horizon
        env_cfg.max_steps_per_rollout_epoch = actual_horizon
        env_cfg.video_cfg.save_video = False
        env = LiberoEnv(env_cfg, num_envs=1, seed_offset=0,
                        total_num_processes=1, worker_info=None)
        try:
            if env.task_suite.get_num_tasks() != len(classification[suite]):
                raise ValueError(f"PLUS task count mismatch: {suite}")
            for task_id in suite_tasks:
                episode = eid(suite, task_id)
                if (run_dir / "rollouts" / episode).exists():
                    read_episode(run_dir / "rollouts" / episode, protocol_hash,
                                 expected_horizon=actual_horizon if new_protocol else None,
                                 expected_protocol_version=protocol.get("protocol_version"),
                                 expected_manifest_hash=protocol.get("task_manifest_content_sha256"))
                    continue
                row = classification[suite][task_id]
                task = env.task_suite.get_task(task_id)
                if task.name != row["name"]:
                    raise ValueError(f"PLUS task mapping mismatch: {episode}")
                policy_seed = 100000 * suite_index + task_id
                metadata = {
                    "complete": True, "execution_status": "complete",
                    "episode_id": episode, "run_id": args.run_id,
                    "protocol_hash": protocol_hash, "suite": suite,
                    "task_id": task_id, "plus_classification_id": row["id"],
                    "task_name": task.name, "shift_category": row["category"],
                    "difficulty": row["difficulty_level"], "trial_id": 0,
                    "simulator_seed": 0, "policy_seed": policy_seed,
                    "checkpoint_repo": checkpoint_repo,
                    "checkpoint_revision": checkpoint_revision,
                    "sampling_mode": protocol["sampling_mode"],
                    "temperature": temperature, "lora_applied": load_lora,
                    "horizon": actual_horizon, "wait_steps": 15,
                    "action_chunks": 8, "unnorm_key": nominal.NORM_KEY,
                    "upstream_commit": nominal.UPSTREAM_COMMIT,
                    "plus_commit": PLUS_COMMIT, "storage_mode": protocol["storage_mode"],
                }
                if new_protocol:
                    metadata.update(protocol_version=protocol["protocol_version"],
                                    task_manifest_content_sha256=protocol["task_manifest_content_sha256"],
                                    horizon_protocol=args.horizon_protocol,
                                    initial_state_protocol=protocol["initial_state_protocol"])
                nominal.collect_episode(env, model, suite=suite, task_id=task_id,
                    trial_id=0, sampling=sampling, temperature=temperature,
                    policy_seed=policy_seed, metadata=metadata, run_dir=run_dir)
                completed_count += 1
                print(f"{episode}: success={metadata['success']} "
                      f"steps={metadata['episode_length']} "
                      f"({completed_count}/{len(expected)})", flush=True)
                if completed_count % 100 == 0:
                    summarize(run_dir, expected, protocol_hash, **summary_kwargs)
                if shutil.disk_usage(root).free < 10 * 2**30:
                    raise OSError("Less than 10 GiB free disk space")
        finally:
            env.env.close()
    summary = summarize(run_dir, expected, protocol_hash, **summary_kwargs)
    if summary["missing"]:
        raise RuntimeError(f"Missing {summary['missing']} episodes")


if __name__ == "__main__":
    main()
