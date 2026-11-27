"""Validate the R0-1 identity and runtime topology for later stages."""

import math

from scripts.common.integrity import sha

from .checkpoint import read_json
from .parameter_diff import ARCHITECTURE_KEYS, runtime_parameters

ROLES = ("sft", "grpo")


def load_r0_1(root, evidence):
    status = read_json(evidence / "status.json")
    if (status.get("stage"), status.get("status"), status.get("dry_run")) != ("R0-1", "PASS", False):
        raise ValueError("R0-1 evidence must record a successful runtime smoke")

    structure = read_json(evidence / "structure.json")
    from scripts.research.cli.audit import MODEL_SHA, MODEL_SOURCE, UPSTREAM_PIN

    if (structure["upstream_commit"], structure["action_source"], structure["action_source_sha256"]) != (
        UPSTREAM_PIN, MODEL_SOURCE.as_posix(), MODEL_SHA
    ):
        raise ValueError("Unexpected R0-1 source provenance")
    inventory_path = root / "manifests/models_130.json"
    config = read_json(root / "configs/research/checkpoint_audit.json")
    for field, path in (
        ("inventory_sha256", inventory_path),
        ("phase3a_manifest_sha256", root / config["phase3a_manifest"]),
    ):
        if structure[field] != sha(path):
            raise ValueError(f"R0-1 provenance does not match {path}")
    inventory = read_json(inventory_path)

    registries, metadata = {}, {}
    for role in ROLES:
        policy = structure["policies"][role]
        metadata[role] = checkpoint = policy["checkpoint"]
        modules = read_json(evidence / f"{role}-modules.json")["modules"]
        recorded_pin = {key: checkpoint[key] for key in ("repo_id", "revision")}
        if (recorded_pin != config[role] or recorded_pin != policy["expected_identity"]
                or checkpoint["inference_adapter_applied"] is not False
                or policy["loading_config"]["is_lora"] is not False
                or policy["loading_config"]["lora_path"] is not None):
            raise ValueError(f"{role} differs from the pinned base-only inference policy")
        if policy["smoke"]["action_path"] != "shared language model vocabulary projection":
            raise ValueError(f"{role} smoke lacks the expected native action path")
        for logical in ("vision", "projector", "language_model", "transformer", "action.logits"):
            if logical not in modules:
                raise ValueError(f"{role} lacks runtime evidence for {logical}")
        if checkpoint["config"]["use_fused_vision_backbone"] and not {
            "vision.primary", "vision.fused"
        }.issubset(modules):
            raise ValueError(f"{role} lacks runtime evidence for fused vision streams")
        registries[role] = modules

    config_sft, config_grpo = (metadata[role]["config"] for role in ROLES)
    different = [key for key in ARCHITECTURE_KEYS
                 if key not in config_sft or key not in config_grpo or config_sft[key] != config_grpo[key]]
    if different:
        raise ValueError(f"UNSAFE architecture or base-model compatibility: {different}")
    topology = {
        role: {key: (module["path"], module["type"]) for key, module in registry.items()}
        for role, registry in registries.items()
    }
    if topology["sft"] != topology["grpo"]:
        raise ValueError("UNSAFE runtime module path or type mismatch")
    runtime = {
        role: runtime_parameters({"modules": registry}, config_sft["timm_model_ids"])
        for role, registry in registries.items()
    }
    details = {
        "r0_1": str(evidence),
        "r0_1_sha256": {name: sha(evidence / name) for name in (
            "status.json", "structure.json", "sft-modules.json", "grpo-modules.json",
        )},
        "runtime": {
            role: {"tensors": len(params), "elements": sum(math.prod(p[1]) for p in params.values())}
            for role, params in runtime.items()
        },
        "module_topology": topology["sft"],
        "architecture": {key: config_sft[key] for key in ARCHITECTURE_KEYS},
        "checkpoint_class_metadata": {role: metadata[role]["config"]["architectures"] for role in ROLES},
        "checkpoints": {
            role: {
                **config[role], "published_adapter": metadata[role]["published_adapter_config"],
                "published_merge_history": metadata[role]["published_merge_history"],
                "adapter_applied_in_baseline": False,
            } for role in ROLES
        },
        "scope": "Exact runtime parameter keys in published base shards. Historical training provenance unverified.",
        "change_atol": 0.0,
        "change_rtol": 0.0,
        "accumulator": "float64",
    }
    return inventory, metadata, runtime, details
