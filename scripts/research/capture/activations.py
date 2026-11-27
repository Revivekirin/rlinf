"""Targeted R1 capture at the measured R0 module boundaries."""

import numpy as np
import torch

from scripts.common.integrity import array_hash
from scripts.research.capture.equivalence import to_array
from scripts.research.capture.hooks import HookManager

from scripts.research.capture_contract import SELECTORS


def collect_observation(adapter, rgb, states, language, observation_id, measured):
    def predict():
        torch.manual_seed(3101)
        torch.cuda.manual_seed_all(3101)
        actions, info = adapter.predict({
            "main_images": torch.from_numpy(rgb.copy()).unsqueeze(0),
            "states": torch.from_numpy(states.copy()).unsqueeze(0),
            "task_descriptions": [language],
        })
        return {"actions": to_array(actions)[0],
                **{key: to_array(value)[0] for key, value in info["forward_inputs"].items()}}

    def registered():
        return {name: tuple(module._forward_hooks) for name, module in adapter.model.named_modules()}

    rows = {row["selector"]: row for row in measured["hook_calls"][0]}
    tokens = rows["action.logits"]["token_boundaries"]
    start, end = tokens["action_sequence"]
    lo, hi = tokens["vocabulary"]

    def bins(tensor):
        return tensor[0, start:end, lo:hi]

    reducers = dict.fromkeys(SELECTORS, lambda tensor: tensor)
    reducers["action.logits"] = bins
    manager = HookManager(adapter.registry, SELECTORS, input_identity=observation_id,
                          token_boundaries={key: rows[key]["token_boundaries"] for key in SELECTORS},
                          capture=reducers)
    initial_hooks = registered()
    plain = predict()
    with manager:
        hooked = predict()
    if registered() != initial_hooks:
        raise ValueError("Capture left residual hooks")
    if plain.keys() != hooked.keys() or any(
        not np.array_equal(plain[key], hooked[key]) or plain[key].dtype != hooked[key].dtype
        for key in plain
    ):
        raise ValueError("Capture changed native inference")
    if len(manager.records) != len(SELECTORS) or set(manager.values) != set(SELECTORS):
        raise ValueError("Unexpected capture count")
    arrays = {"actions": plain["actions"], "action_tokens": plain["action_tokens"]}
    for row in manager.records:
        key = row["selector"]
        reference = rows[key]
        for field in ("module_path", "module_type", "shape", "dtype", "forward_call_index", "token_boundaries"):
            if row[field] != reference[field]:
                raise ValueError(f"R0 hook contract mismatch for {key}: {field}")
        arrays[key] = to_array(manager.values[key][0])
        row["stored_dtype"] = str(arrays[key].dtype)
        row["representation"] = "action_bin_slice" if key == "action.logits" else "unpooled_module_output"

    logits = arrays["action.logits"]
    if (logits.shape != (56, 256) or arrays["actions"].shape != (8, 7)
            or arrays["action_tokens"].shape != (8, 7)
            or not np.array_equal((logits.argmax(-1) + lo).reshape(8, 7), arrays["action_tokens"])
            or not all(np.isfinite(value).all() for value in arrays.values())):
        raise ValueError("Invalid action tensor or logits/greedy-token disagreement")
    return arrays, {
        "hooks": manager.records, "hook_registrations_restored": True, "greedy_seed": 3101,
        "unhooked_hashes": {key: array_hash(value) for key, value in plain.items()},
    }
