"""Offline R1 dataset and capture checks."""

from collections import Counter

import numpy as np

from scripts.common.integrity import array_hash
from scripts.research.capture_contract import SELECTORS
from scripts.research.metrics.action import differences
from scripts.research.models.checkpoint import read_json
from .camera_pairs import OBSERVATIONS, PAIRS, SCHEMA, TASKS, TRIALS, VIEWS, load_arrays, validate_pair


def verify_dataset(path):
    dataset = read_json(path / "dataset.json")
    spec = read_json(path / "plan.json")
    if (dataset["schema"] != SCHEMA or spec["schema"] != SCHEMA
            or spec["expected_observations"] != OBSERVATIONS or spec["expected_pairs"] != PAIRS
            or any(spec[key] != dataset[key] for key in ("plus_commit", "rlinf_commit"))):
        raise ValueError("Unsupported dataset schema")
    variants = {
        (case["suite"], case["task_id"], case["task"]): {
            f"plus-{variant['id']}": variant for variant in case["variants"]
        }
        for case in spec["cases"]
    }
    if set(variants) != set(TASKS) or any(len(views) != VIEWS - 1 for views in variants.values()):
        raise ValueError("Incomplete camera plan")
    observations = dataset["observations"]
    pairs = dataset["pairs"]
    if (len(observations), len(pairs)) != (OBSERVATIONS, PAIRS):
        raise ValueError("Missing target observations or pairs")
    arrays = {}
    views = Counter()
    for key, record in observations.items():
        task = record["task"]
        task_key = (task["suite"], task["original_id"], task["name"])
        if task_key not in TASKS or key != record["observation_id"]:
            raise ValueError("Invalid observation identity")
        trial = record["trial"]
        state_id = f"{task['suite']}-t{task['original_id']:02d}-s{trial:02d}"
        label = record["camera_id"]
        variant = record["variant"]
        if not 0 <= trial < TRIALS or record["state_id"] != state_id or key != f"{state_id}-{label}":
            raise ValueError("Invalid state identity")
        if ((label != "canonical" and label not in variants[task_key])
                or variant != variants[task_key].get(label)):
            raise ValueError("Invalid camera identity")
        camera = record["camera"]
        rot = np.asarray(camera["rotation_world_from_camera"])
        pos = np.asarray(camera["position_m"])
        fov = camera["fovy_deg"]
        if (rot.shape != (3, 3) or pos.shape != (3,) or not np.isfinite(pos).all()
                or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(rot), 1, atol=1e-7, rtol=0)
                or not 0 < fov < 180 or not np.isclose(camera["focal_px"], 128 / np.tan(np.radians(fov) / 2))
                or camera["principal_point_px"] != [128, 128]):
            raise ValueError("Invalid camera calibration")
        if not record["language"] or record["controller"]["config"]["control_delta"] is not True:
            raise ValueError("Missing instruction or delta action convention")
        arrays[key] = load_arrays(path, record["data"])
        views[state_id] += 1
    if len(views) != len(TASKS) * TRIALS or set(views.values()) != {VIEWS}:
        raise ValueError("Unexpected number of views per state")
    seen = set()
    for pair in pairs:
        left, right = pair["id"], pair["ood"]
        if (pair["pair_id"] != right or right in seen or pair["physical_atol"] or pair["physical_rtol"]
                or pair["kind"] != "same_state_replay_of_plus_camera"):
            raise ValueError("Invalid pair identity or alignment contract")
        validate_pair(observations[left], observations[right], arrays[left], arrays[right])
        seen.add(right)
    if seen != {key for key, record in observations.items() if record["camera_id"] != "canonical"}:
        raise ValueError("Incomplete pair coverage")
    return dataset


def comparisons(dataset, captured, identities):
    rows = []
    for role, records in captured.items():
        for pair in dataset["pairs"]:
            left, right = (records[pair[key]] for key in ("id", "ood"))
            obs = dataset["observations"][pair["ood"]]
            rows.append({
                "pair_id": pair["pair_id"], "task": obs["task"], "variant": obs["variant"],
                "model": role, "identity": identities[role], "id": pair["id"], "ood": pair["ood"],
                "inputs": [dataset["observations"][pair[key]]["data"]["sha256"] for key in ("id", "ood")],
                "outputs": [left["hash"], right["hash"]],
                **differences(left["actions"], right["actions"], left["action_tokens"], right["action_tokens"]),
            })
    return rows


def verify_capture(path, dataset, identities, references):
    manifest = read_json(path / "captures.json")
    if (manifest["schema"] != SCHEMA or manifest["checkpoints"] != identities
            or set(manifest["policies"]) != set(identities)):
        raise ValueError("Invalid capture schema or checkpoint identities")
    captured = {}
    processed = {}
    for role, records in manifest["policies"].items():
        if set(records) != dataset["observations"].keys():
            raise ValueError(f"Incomplete {role} captures")
        reference = references[role]
        regression = read_json(path / f"{role}-r0-regression.json")
        if manifest["r0_regression"][role] != "PASS" or not regression["passed"] or any(
            regression[key] != reference[key] for key in (
                "decoded_actions", "selected_token_ids", "processed_input_sha256", "hook_calls"
            )
        ):
            raise ValueError(f"Missing native regression for {role}")
        baseline = {row["selector"]: row for row in reference["hook_calls"][0]}
        captured[role] = {}
        for obs_id, record in records.items():
            if (record["observation_id"] != obs_id
                    or record["input_sha256"] != dataset["observations"][obs_id]["data"]["sha256"]
                    or record["hook_registrations_restored"] is not True):
                raise ValueError("Capture input or lifecycle mismatch")
            pairs = sorted(p["pair_id"] for p in dataset["pairs"] if obs_id in (p["id"], p["ood"]))
            if pairs != record["pair_ids"]:
                raise ValueError("Capture pair association mismatch")
            hashes = {key: record["unhooked_hashes"][key]
                      for key in ("pixel_values", "input_ids", "attention_mask")}
            if processed.setdefault(obs_id, hashes) != hashes:
                raise ValueError("SFT and GRPO preprocessing differs for the saved observation")
            values = load_arrays(path, record["data"])
            hooks = {row["selector"]: row for row in record["hooks"]}
            if (set(hooks) != set(SELECTORS) or len(record["hooks"]) != len(SELECTORS)
                    or set(values) != set(SELECTORS) | {"actions", "action_tokens"}):
                raise ValueError("Incorrect capture selector coverage")
            for selector, row in hooks.items():
                expected = baseline[selector]
                if any(row[key] != expected[key] for key in (
                    "module_path", "module_type", "shape", "dtype", "forward_call_index", "token_boundaries"
                )) or row["input_identity"] != obs_id:
                    raise ValueError(f"Hook provenance mismatch: {selector}")
                shape = (56, 256) if selector == "action.logits" else tuple(row["shape"])
                dtype = "float32" if row["dtype"] == "torch.bfloat16" else row["dtype"].removeprefix("torch.")
                if values[selector].shape != shape or str(values[selector].dtype) != dtype or row["stored_dtype"] != dtype:
                    raise ValueError(f"Stored hook array mismatch: {selector}")
            lo = hooks["action.logits"]["token_boundaries"]["vocabulary"][0]
            tokens, actions = values["action_tokens"], values["actions"]
            if (tokens.dtype.kind not in "iu" or tokens.shape != (8, 7) or actions.shape != (8, 7)
                    or not np.array_equal(tokens, (values["action.logits"].argmax(-1) + lo).reshape(8, 7))
                    or any(array_hash(values[key]) != record["unhooked_hashes"][key]
                           for key in ("actions", "action_tokens"))):
                raise ValueError("Action/logit validation failed")
            captured[role][obs_id] = {"actions": actions, "action_tokens": tokens,
                                       "hash": record["data"]["sha256"]}
    if read_json(path / "comparisons.json") != comparisons(dataset, captured, identities):
        raise ValueError("Pair comparison metrics disagree with saved actions")
