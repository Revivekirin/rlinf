"""Authenticate local checkpoint files before the native loader reads them."""

import hashlib
import json
from pathlib import Path

from scripts.common.integrity import sha


class AuditBlocked(Exception):
    """A required local resource is unavailable."""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inspect_checkpoint(path: Path, pin: dict, inventory: dict) -> dict:
    entries = [
        model for model in inventory["models"]
        if all(model[key] == pin[key] for key in ("repo_id", "revision"))
    ]
    if len(entries) != 1:
        raise ValueError(f"No unique inventory entry for {pin}")
    if not path.is_dir():
        raise AuditBlocked(f"Checkpoint directory unavailable: {path}")

    files = entries[0]["files"]
    expected = {entry["path"] for entry in files}
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if relative.parts[0] in (".cache", ".git"):
            continue
        if candidate.is_file() and relative.as_posix() not in expected:
            raise ValueError(f"Unpinned checkpoint file: {candidate}")
    for entry in files:
        file = path / entry["path"]
        if not file.is_file():
            raise AuditBlocked(f"Checkpoint file unavailable: {file}")
        if file.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {file}")

    for entry in files:
        file = path / entry["path"]
        if entry["lfs_sha256"]:
            actual = sha(file)
            expected_digest = entry["lfs_sha256"]
        else:
            content = file.read_bytes()
            actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            expected_digest = entry["blob_id"]
        if actual != expected_digest:
            raise ValueError(f"Checkpoint digest mismatch: {file}")

    config = read_json(path / "config.json")
    index = read_json(path / "model.safetensors.index.json")["weight_map"]
    shards = {name for name in expected if "/" not in name and name.endswith(".safetensors")}
    if not index or set(index.values()) != shards or not shards:
        raise ValueError(f"Unexpected shard index: {path}")
    if config["model_type"] != "openvla" or config["text_config"]["model_type"] != "llama":
        raise ValueError(f"Unexpected model architecture: {path}")

    adapter_path = "lora_adapter/adapter_config.json"
    return {
        **pin, "path": str(path), "verified_files": len(files),
        "config": config, "indexed_tensor_count": len(index), "shards": sorted(shards),
        "published_adapter_config": read_json(path / adapter_path) if adapter_path in expected else None,
        "inference_adapter_applied": False,
        "published_merge_history": "unverified",
    }
