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


def load_r0_2(evidence, expected_r0_1, runtime, details):
    from collections import Counter

    status_path = evidence / "status.json"
    status = read_json(status_path)
    if (status.get("stage"), status.get("status"), status.get("dry_run")) != ("R0-2", "PASS", False):
        raise ValueError("R0-2 evidence must record a successful numerical comparison")
    files = status["evidence_sha256"]
    if set(files) != {"provenance.json", "comparison.json", "modules.csv", "tensors.csv"}:
        raise ValueError("R0-2 evidence file set differs from the comparison CLI")
    for name, digest in files.items():
        if sha(evidence / name) != digest:
            raise ValueError(f"R0-2 evidence hash mismatch: {name}")

    provenance = read_json(evidence / "provenance.json")
    for field in ("r0_1_sha256", "runtime", "module_topology", "architecture", "checkpoints"):
        # JSON uses arrays for the module path/type pairs.
        expected = expected_r0_1 if field == "r0_1_sha256" else details[field]
        if field == "module_topology":
            expected = {key: list(value) for key, value in expected.items()}
        if provenance[field] != expected:
            raise ValueError(f"R0-2 provenance mismatch: {field}")
    data = read_json(evidence / "comparison.json")
    rows = data["tensors"]
    counts = dict(Counter(row["status"] for row in rows))
    aligned = [row for row in rows if row["status"] in {"identical", "changed"}]
    if counts != status["tensor_status_counts"] or counts != data["counts"]:
        raise ValueError("R0-2 comparison status counts disagree")
    if set(counts) - {"identical", "changed", "extra_sft", "extra_grpo", "unmapped"}:
        raise ValueError("R0-2 contains incompatible runtime tensors")
    names = {row["canonical_name"] for row in aligned}
    if len(aligned) != len(names) or names != runtime["sft"].keys() or names != runtime["grpo"].keys():
        raise ValueError("R0-2 coverage differs from the verified runtime policy")
    for row in aligned:
        name = row["canonical_name"]
        group, shape, _ = runtime["sft"][name]
        if (row["checkpoint_key"] != name or row["module"] != group
                or row["sft_shape"] != list(shape) or row["grpo_shape"] != list(shape)
                or row["parameter_count"] != math.prod(shape)
                or row["sft_storage_dtype"] != row["grpo_storage_dtype"]
                or not 0 <= row["changed_count"] <= row["parameter_count"]
                or (row["status"] == "changed") != (row["changed_count"] > 0)):
            raise ValueError(f"Invalid aligned R0-2 tensor evidence: {name}")
    total = data["modules"]["policy.total"]
    for key in ("parameter_count", "changed_count"):
        if sum(row[key] for row in aligned) != total[key]:
            raise ValueError(f"R0-2 policy aggregate mismatch: {key}")
    return {"status_sha256": sha(status_path), "files": files, "tensor_status_counts": counts}
