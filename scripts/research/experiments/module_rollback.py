"""Plan, run and verify exhaustive R2-A interventions on the sealed replay."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from scripts.common.integrity import array_hash, sha
from scripts.research.capture_contract import SELECTORS
from scripts.research.core.artifacts import write_json
from scripts.research.core.r1 import prior_evidence, sealed_files, validate_run
from scripts.research.data.camera_pairs import load_arrays, save_arrays
from scripts.research.data.simulator_state import revision
from scripts.research.interventions.parameter import groups_from_audit
from scripts.research.metrics.action import differences
from scripts.research.metrics.intervention import activation_difference, pair_change
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint, read_json
from scripts.research.models.tensor_shards import indexed_tensors
from scripts.research.models.runtime_evidence import load_r0_1

ALPHAS = (0.0, 0.5, 1.0)
SCHEMA = 1


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def plan(root, replay):
    validate_run(root, replay)
    r0 = root / "outputs/research/R0" / read_json(replay / "provenance.json")["r0_3"]
    inventory, checkpoints, details, provenance, references = prior_evidence(root, r0)
    comparison_path = r0.parent / Path(provenance["r0_2"]).name / "comparison.json"
    _, _, runtime, _ = load_r0_1(root, r0.parent / Path(provenance["r0_1"]).name)
    comparison = read_json(comparison_path)
    groups = groups_from_audit(comparison, runtime["grpo"])
    identity = {role: {"repo_id": row["repo_id"], "revision": row["revision"],
                      "file_hashes": {item["path"]: item["lfs_sha256"] or item["blob_id"]
                                      for entry in inventory["models"]
                                      if entry["repo_id"] == row["repo_id"] and entry["revision"] == row["revision"]
                                      for item in entry["files"]}}
                for role, row in checkpoints.items()}
    experiment = {
        "schema": SCHEMA, "checkpoints": identity, "r1_run": replay.name,
        "r1_status_sha256": sha(replay / "status.json"), "dataset_sha256": sha(replay / "dataset.json"),
        "r0_2_comparison_sha256": sha(comparison_path), "r0_3_status_sha256": sha(r0 / "status.json"),
        "inference": {"seed": 3101, "do_sample": False, "temperature": 1.0, "top_k": -1,
                      "top_p": 1.0, "selectors": list(SELECTORS)},
        "inventory": groups,
    }
    return experiment, (inventory, checkpoints, runtime["grpo"], references["grpo"])


def units(experiment, dataset, mode):
    inventory = experiment["inventory"]
    leaves = inventory["leaves"]
    if mode == "coarse":
        groups = inventory["parents"]
    else:
        groups = {key: [key] for key in leaves}
    if mode == "smoke":
        selected, seen = {}, set()
        for key, group in sorted(leaves.items(), key=lambda item: (item[1]["elements"], item[0])):
            region = "transformer" if group["region"].startswith("transformer.") else group["region"]
            category = group["status"] if group["status"] == "NO_UPDATE" else region
            if group["status"] != "UNSAFE" and category not in seen:
                selected[key] = [key]
                seen.add(category)
        groups = selected
    obs_ids = sorted(dataset["observations"])
    if mode == "smoke":
        first_state = dataset["observations"][obs_ids[0]]["state_id"]
        obs_ids = [key for key in obs_ids if dataset["observations"][key]["state_id"] == first_state]
    signature = hash_object({key: value for key, value in experiment.items() if key != "inventory"})
    for group, owners in sorted(groups.items()):
        values = [leaves[key] for key in owners]
        if any(row["status"] == "UNSAFE" for row in values):
            continue
        for alpha in ALPHAS:
            unit = {"group": group, "alpha": alpha, "mode": mode, "observations": obs_ids,
                    "no_update": all(row["status"] == "NO_UPDATE" for row in values),
                    "parameters": sorted(name for row in values for name in row["parameters"]),
                    "elements": sum(row["elements"] for row in values), "experiment": signature}
            yield hash_object(unit), unit


def selected_pairs(dataset, obs_ids):
    return [pair for pair in dataset["pairs"] if pair["id"] in obs_ids and pair["ood"] in obs_ids]


def unit_result(directory, unit, dataset, baseline, capture_records):
    status = read_json(directory / "status.json")
    if status["unit"] != unit or status["status"] != "PASS":
        raise ValueError("Unexpected intervention unit identity")
    sealed_files(directory, status)
    result = read_json(directory / "measurements.json")
    if (result["r0_grpo_regression"] != "PASS" or result["r1_grpo_regression"] != "PASS"
            or result["sft_reference"] != "sealed R1 deterministic captures"
            or not np.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0
            or result["cuda_peak_bytes"] < 0):
        raise ValueError("Invalid control or resource evidence")
    weights = result["weights"]
    if (not weights["restored"] or not weights["non_selected_unchanged"]
            or weights["elements"] != unit["elements"]
            or not 0 <= weights["effective_changed"] <= unit["elements"]
            or not np.isfinite(weights["effective_l2"]) or weights["effective_l2"] < 0
            or (weights["effective_relative_l2"] is not None and (
                not np.isfinite(weights["effective_relative_l2"]) or weights["effective_relative_l2"] < 0
            ))):
        raise ValueError("Invalid restoration/effective perturbation evidence")
    invariant = unit["alpha"] == 1.0 or unit["no_update"]
    if invariant and (weights["effective_changed"] or weights["effective_l2"] != 0):
        raise ValueError("Control changed model weights")
    if set(result["observations"]) != set(unit["observations"]):
        raise ValueError("Incomplete intervention observations")
    actions, refs = {}, {}
    for obs_id, row in result["observations"].items():
        observation = dataset["observations"][obs_id]
        capture = capture_records[obs_id]
        if (row["input_sha256"] != dataset["observations"][obs_id]["data"]["sha256"]
                or row["camera_id"] != observation["camera_id"] or row["pair_ids"] != capture["pair_ids"]
                or row["hooks"] != capture["hooks"]
                or not row["hook_registrations_restored"]
                or set(row["functional"]) != set(SELECTORS)):
            raise ValueError("Invalid observation provenance")
        arrays = load_arrays(directory, row["data"])
        base, base_hashes = baseline(obs_id)
        if (set(arrays) != {"actions", "action_tokens", "action.logits"}
                or row["tensor_sha256"].keys() != base_hashes.keys()
                or any(array_hash(value) != row["tensor_sha256"][key] for key, value in arrays.items())):
            raise ValueError("Invalid intervention tensor identity")
        lo = row["vocab_start"]
        if (lo != 31744 or arrays["actions"].shape != (8, 7)
                or arrays["action.logits"].shape != (56, 256)
                or arrays["action_tokens"].shape != (8, 7) or arrays["action_tokens"].dtype.kind not in "iu"
                or not np.array_equal((arrays["action.logits"].argmax(-1) + lo).reshape(8, 7),
                                      arrays["action_tokens"])
                or activation_difference(base["action.logits"], arrays["action.logits"])
                   != row["functional"]["action.logits"]
                or differences(base["actions"], arrays["actions"], base["action_tokens"],
                               arrays["action_tokens"]) != row["direct"]):
            raise ValueError("Invalid token decode or direct effect")
        for selector, metric in row["functional"].items():
            if (not np.isfinite([metric["l2"], metric["max_abs"]]).all()
                    or min(metric["l2"], metric["max_abs"]) < 0
                    or (invariant and (metric["l2"] or metric["max_abs"]))
                    or metric["elements"] != int(np.prod(base_hashes[selector]["shape"]))):
                raise ValueError("Invalid activation summary")
        if invariant and any(row["tensor_sha256"][name] != record["sha256"] for name, record in base_hashes.items()):
            raise ValueError("Control diverged from archived GRPO")
        actions[obs_id], refs[obs_id] = arrays["actions"], base["actions"]
    expected = [{"pair_id": p["pair_id"], "id": p["id"], "ood": p["ood"],
                 "task": dataset["observations"][p["ood"]]["task"],
                 "variant": dataset["observations"][p["ood"]]["variant"],
                 **pair_change(refs[p["id"]], refs[p["ood"]], actions[p["id"]], actions[p["ood"]])}
                for p in selected_pairs(dataset, actions)]
    if expected != result["pairs"]:
        raise ValueError("Invalid paired action response evidence")
    return result


def r1_baseline(replay, captures, obs_id):
    record = captures["policies"]["grpo"][obs_id]["data"]
    with np.load(replay / record["file"], allow_pickle=False) as arrays:
        value = {key: arrays[key] for key in ("actions", "action_tokens", "action.logits")}
    return value, record["arrays"]


def validate(root, out, replay):
    experiment, _ = plan(root, replay)
    manifest = read_json(out / "manifest.json")
    if manifest["experiment"] != experiment:
        raise ValueError("Experiment provenance or inventory mismatch")
    mode = manifest["mode"]
    dataset = read_json(replay / "dataset.json")
    captures = read_json(replay / "captures.json")
    expected = dict(units(experiment, dataset, mode))
    done = set()
    for directory in sorted((out / "units").iterdir()):
        if not directory.is_dir() or directory.name not in expected:
            raise ValueError(f"Unexpected unit: {directory}")
        unit_result(directory, expected[directory.name], dataset,
                    lambda obs_id: r1_baseline(replay, captures, obs_id), captures["policies"]["grpo"])
        done.add(directory.name)
    status = "PARTIAL"
    if done == expected.keys():
        status = {"smoke": "SMOKE_PASS", "coarse": "COARSE_PASS", "full": "FULL_PASS"}[mode]
        if mode == "full" and any(g["status"] == "UNSAFE" for g in experiment["inventory"]["leaves"].values()):
            status = "PARTIAL"
    return {"status": status, "verified_units": len(done), "expected_units": len(expected),
            "verified_groups": len({expected[key]["group"] for key in done}),
            "remaining_units": sorted(expected.keys() - done)}


def execute(root, out, replay, mode, checkpoint_root, preflight=False, max_units=None):
    experiment, (inventory, checkpoints, runtime, measured) = plan(root, replay)
    manifest = {"experiment": experiment, "mode": mode}
    if not (out / "manifest.json").exists():
        write_json(out / "manifest.json", manifest)
    elif read_json(out / "manifest.json") != manifest:
        raise ValueError("Cannot resume with different experiment provenance or coverage")
    unit_root = out / "units"
    unit_root.mkdir(exist_ok=True)
    if preflight:
        return {"status": "PREFLIGHT", "inventory": dict(Counter(
            row["status"] for row in experiment["inventory"]["leaves"].values()))}

    import torch
    if not torch.cuda.is_available():
        raise AuditBlocked("CUDA unavailable for audited intervention")
    from scripts.analysis import phase3b
    from scripts.analysis.phase3b_distribution import capture_native_logits
    from scripts.research.capture.activations import collect_observation
    from scripts.research.capture.equivalence import evaluate
    from scripts.research.cli.audit import UPSTREAM_PIN
    from scripts.research.cli.check_hook_equivalence import load_observation
    from scripts.research.interventions.parameter import edited, live_parameters
    from scripts.research.models.rlinf_oft import RLinfOFTAdapter, baseline_config

    dataset = read_json(replay / "dataset.json")
    captures = read_json(replay / "captures.json")
    condition_units = list(units(experiment, dataset, mode))
    completed = validate(root, out, replay)
    remaining = set(completed["remaining_units"])
    pending = [item for item in condition_units if item[0] in remaining]
    if not pending:
        return completed
    cfg = read_json(root / "configs/research/checkpoint_audit.json")
    protocols = read_json(root / cfg["phase3a_manifest"])["protocols"]
    revision(root / "upstream/RLinf", UPSTREAM_PIN)
    shard_index, paths = {}, {}
    for role, reference in checkpoints.items():
        checkpoint = checkpoint_root / reference["repo_id"].rsplit("/", 1)[-1]
        paths[role] = checkpoint
        verified = inspect_checkpoint(checkpoint, cfg[role], inventory)
        if verified != dict(reference, path=str(checkpoint)):
            raise ValueError(f"Authenticated checkpoint differs from R0: {role}")
        shard_index[role] = indexed_tensors(checkpoint, verified["shards"])
    adapter = RLinfOFTAdapter.load(root / "upstream/RLinf",
                                  baseline_config(protocols["grpo"], cfg["grpo"], paths["grpo"]))
    parameters = live_parameters(adapter.model, runtime)
    r0_run = root / "outputs/research/R0" / read_json(replay / "provenance.json")["r0_3"]
    r0_data = read_json(r0_run / "provenance.json")
    entry, bank = load_observation((root / cfg["phase3a_manifest"]).parent, r0_data["observation_id"])
    phase3b.require(entry["file_sha256"] == r0_data["observation_sha256"], "R0 replay observation changed")
    report, logits = evaluate(adapter, bank, entry["source_metadata"]["language"],
                              entry["observation_id"], capture_native_logits)
    with np.load(r0_run / "grpo-action-logits.npz", allow_pickle=False) as reference:
        if report != measured or not np.array_equal(reference["phase3b"], logits["phase3b"]):
            raise ValueError("Live GRPO failed the sealed R0-3 regression")

    def infer(obs_id):
        obs = dataset["observations"][obs_id]
        arrays = load_arrays(replay, obs["data"])
        return collect_observation(adapter, arrays["main_image_policy"], arrays["proprio_policy"],
                                   obs["language"], obs_id, measured)

    # One live GRPO baseline per selected observation; SFT controls are the sealed R1 captures.
    owner_calls = set()
    handles = [adapter.model.get_submodule(owner).register_forward_pre_hook(
        lambda module, inputs, owner=owner: owner_calls.add(owner)
    ) for owner in experiment["inventory"]["leaves"]]
    try:
        for obs_id in condition_units[0][1]["observations"]:
            arrays, record = infer(obs_id)
            _, hashes = r1_baseline(replay, captures, obs_id)
            if (any(array_hash(value) != hashes[name]["sha256"] for name, value in arrays.items())
                    or record["unhooked_hashes"] != captures["policies"]["grpo"][obs_id]["unhooked_hashes"]):
                raise ValueError(f"Live GRPO differs from sealed R1 controls: {obs_id}")
    finally:
        for handle in handles:
            handle.remove()

    for unit_id, unit in pending[:max_units]:
        print(f"Running {unit['group']} alpha={unit['alpha']}", flush=True)
        start = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        observations, base_actions, new_actions = {}, {}, {}
        with TemporaryDirectory(prefix="pending-", dir=out) as staging:
            directory = Path(staging) / unit_id
            directory.mkdir()
            with edited(parameters, unit["parameters"], shard_index["sft"], shard_index["grpo"], unit["alpha"]) as stats:
                for obs_id in unit["observations"]:
                    obs = dataset["observations"][obs_id]
                    values, evidence = infer(obs_id)
                    r1_record = captures["policies"]["grpo"][obs_id]
                    if any(evidence["unhooked_hashes"][key] != r1_record["unhooked_hashes"][key]
                           for key in ("input_ids", "attention_mask", "pixel_values")):
                        raise ValueError(f"Preprocessing changed for {obs_id}")
                    with np.load(replay / r1_record["data"]["file"], allow_pickle=False) as archived:
                        functional = {key: activation_difference(archived[key], values[key]) for key in SELECTORS}
                        direct = differences(archived["actions"], values["actions"], archived["action_tokens"],
                                             values["action_tokens"])
                        base_actions[obs_id] = archived["actions"]
                    new_actions[obs_id] = values["actions"]
                    observations[obs_id] = {
                        "input_sha256": obs["data"]["sha256"], "camera_id": obs["camera_id"],
                        "pair_ids": r1_record["pair_ids"], "hooks": evidence["hooks"],
                        "hook_registrations_restored": evidence["hook_registrations_restored"],
                        "vocab_start": next(h for h in evidence["hooks"] if h["selector"] == "action.logits")
                                       ["token_boundaries"]["vocabulary"][0],
                        "functional": functional, "direct": direct,
                        "tensor_sha256": {key: array_hash(value) for key, value in values.items()},
                        "data": save_arrays(directory / f"{obs_id}.npz", {
                            key: values[key] for key in ("actions", "action_tokens", "action.logits")
                        }),
                    }
            pairs = [{"pair_id": pair["pair_id"], "id": pair["id"], "ood": pair["ood"],
                      "task": dataset["observations"][pair["ood"]]["task"],
                      "variant": dataset["observations"][pair["ood"]]["variant"],
                      **pair_change(base_actions[pair["id"]], base_actions[pair["ood"]],
                                    new_actions[pair["id"]], new_actions[pair["ood"]])}
                     for pair in selected_pairs(dataset, observations)]
            write_json(directory / "measurements.json", {
                "weights": stats, "observations": observations, "pairs": pairs,
                "owner_forward_observed": sorted(owner_calls),
                "r0_grpo_regression": "PASS", "r1_grpo_regression": "PASS",
                "sft_reference": "sealed R1 deterministic captures",
                "elapsed_seconds": perf_counter() - start, "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            })
            write_json(directory / "status.json", {"status": "PASS", "unit": unit,
                       "evidence_sha256": {p.name: sha(p) for p in directory.iterdir()}})
            unit_result(directory, unit, dataset, lambda obs_id: r1_baseline(replay, captures, obs_id),
                        captures["policies"]["grpo"])
            directory.rename(unit_root / unit_id)
    return validate(root, out, replay)
