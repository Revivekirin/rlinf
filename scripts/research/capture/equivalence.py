"""Compare native Phase 3B inference with registry-based capture."""

import random

import numpy as np
import torch

from scripts.common.integrity import array_hash
from scripts.research.capture.hooks import HookManager


def to_array(tensor):
    tensor = tensor.detach().cpu()
    return (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()


def exact_difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"equal": False, "max_absolute_delta": None,
                "left_shape": list(left.shape), "right_shape": list(right.shape)}
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return {"equal": False, "max_absolute_delta": None}
    max_delta = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), initial=0))
    return {"equal": np.array_equal(left, right), "max_absolute_delta": max_delta}


def evaluate(adapter, bank, language, observation_id, reference, seed=3101):
    core, registry = adapter.model, adapter.registry
    blocks = registry.block_count
    vision = ["vision", "vision.primary", "vision.fused", "projector"]
    selectors = [key for key in vision if key in registry.modules]
    selectors += [f"transformer.{i}" for i in sorted({0, blocks // 2, blocks - 1})]
    selectors.append("action.logits")
    lo, hi = core.vocab_size - core.config.n_action_bins, core.vocab_size
    slots = core.action_dim * core.num_action_chunks
    patches = core.vision_backbone.get_num_patches() * core.vision_backbone.get_num_images_in_input()
    start = patches + bank["input_ids"].shape[-1] - 1
    end = start + slots

    def action_bins(tensor):
        if tensor.ndim != 3 or tensor.shape[:2] != (1, end + 1) or tensor.shape[-1] < hi:
            raise ValueError("LM output shape disagrees with the pinned native action slice")
        return tensor[:, start:end, lo:hi].float()

    boundaries = {
        key: {"action_sequence": [start, end]}
        for key in selectors if key.startswith("transformer.")
    }
    boundaries["action.logits"] = {"action_sequence": [start, end], "vocabulary": [lo, hi]}
    for key in ("vision", "projector"):
        if key in selectors:
            boundaries[key] = {"patch_axis": 1, "local_patches": [0, patches]}
    # Substream outputs can include architecture-specific token prefixes. Their
    # shape is recorded independently of positions in the fused LM sequence.
    manager = HookManager(registry, selectors, input_identity=observation_id,
                          token_boundaries=boundaries, capture={"action.logits": action_bins})

    def predict():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        observations = {
            "main_images": torch.from_numpy(bank["main_image_policy"].copy()).unsqueeze(0),
            "states": torch.from_numpy(bank["proprio_policy"].copy()).unsqueeze(0),
            "task_descriptions": [language],
        }
        if "wrist_image_policy" in bank:
            observations["wrist_images"] = torch.from_numpy(bank["wrist_image_policy"].copy()).unsqueeze(0)
        actions, result = adapter.predict(observations)
        return {"actions": to_array(actions)[0],
                **{key: to_array(value)[0] for key, value in result["forward_inputs"].items()}}

    def registrations():
        return {name: tuple(module._forward_hooks) for name, module in core.named_modules()}

    existing = registrations()
    plain = predict()
    ref_logits, ref_actions, ref_inputs, ref_lo, ref_hi = reference(
        core, core, bank, language, seed, check_equivalence=False,
    )
    if registrations() != existing or (lo, hi) != (ref_lo, ref_hi):
        raise ValueError("Phase 3B reference changed hook registration or action vocabulary")
    captures = []
    hooked_outputs = []
    for _ in range(2):
        with manager:
            hooked_outputs.append(predict())
        if registrations() != existing:
            raise ValueError("Residual hook registration after a capture")
        if {row["selector"] for row in manager.records} != set(selectors) or len(manager.records) != len(selectors):
            raise ValueError("Expected one forward call per selected boundary")
        captures.append((manager.records.copy(), manager.values["action.logits"][0].numpy()[0]))

    ref_output = {"actions": ref_actions, **ref_inputs}
    if (plain["actions"].shape != (8, 7) or plain["action_tokens"].shape != (8, 7)
            or plain["input_ids"].shape != bank["input_ids"].shape
            or ref_logits.shape != (slots, hi - lo)):
        raise ValueError("Unexpected native inference output or prompt shape")
    checks = {}
    for name, candidate in (("phase3b", ref_output), ("hooked", hooked_outputs[0]),
                            ("hooked_repeat", hooked_outputs[1])):
        if candidate.keys() != plain.keys():
            raise ValueError(f"Native forward-input keys differ in {name}")
        checks[name] = {key: exact_difference(value, candidate[key]) for key, value in plain.items()}
    for index, (_, logits) in enumerate(captures):
        checks[f"action_logits_{index}"] = {"phase3b": exact_difference(ref_logits, logits)}
        expected_tokens = (logits.argmax(-1) + lo).reshape(8, 7)
        checks[f"action_logits_{index}"]["greedy_tokens"] = exact_difference(
            plain["action_tokens"], expected_tokens.astype(plain["action_tokens"].dtype),
        )
    stable_hooks = captures[0][0] == captures[1][0]
    return {
        "passed": stable_hooks and all(row["equal"] for group in checks.values() for row in group.values()),
        "checks": checks, "hook_calls": [records for records, _ in captures],
        "repeated_hook_metadata_equal": stable_hooks,
        "unavailable_selectors": registry.unavailable,
        "processed_input_sha256": {key: array_hash(plain[key])
                                   for key in ("input_ids", "attention_mask", "pixel_values")},
        "decoded_actions": plain["actions"].tolist(),
        "selected_token_ids": plain["action_tokens"].tolist(),
        "seed": seed, "do_sample": False, "atol": 0, "rtol": 0,
        "ranges": "zero-based half-open intervals",
        "logits_reference": "Phase 3B language-model output hook before action mask",
        "raw_logits_without_hooks": "unavailable from native return",
    }, {"phase3b": ref_logits, "hooked": captures[0][1], "hooked_repeat": captures[1][1]}
