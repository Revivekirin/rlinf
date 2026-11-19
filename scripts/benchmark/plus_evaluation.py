"""CPU-only selection and horizon contracts for LIBERO-PLUS evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark.plus_manifest import BENCHMARK_REVISION, content_hash
from scripts.benchmark.plus_inventory import SUITE_COUNTS, original_stem


HORIZONS = {
    "rlinf-520": {suite: 520 for suite in SUITE_COUNTS},
    "lerobot-suite": {
        "libero_spatial": 280, "libero_object": 280,
        "libero_goal": 300, "libero_10": 520,
    },
}
LEGACY_PLUS_COLLECTOR_SHA256 = "5ee95909eab3b7f981c34b020b4a89cddf3e2308e0b09a49d0dc6d7f4b48e524"


def horizon_for(protocol: str, suite: str) -> int:
    return HORIZONS[protocol][suite]


def episode_id(suite: str, registry_index: int) -> str:
    return f"{suite}-t{registry_index:04d}-i00"


def prepare_run_directory(run_dir: Path, *, resume: bool) -> None:
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    elif run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=resume)


def load_selection(path: Path, classification: dict, classification_path: Path,
                   task_map_path: Path) -> tuple[dict, dict[str, list[int]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != "pilot" or manifest.get("manifest_version") != 1:
        raise ValueError("Evaluator requires a version-1 pilot manifest")
    if manifest.get("benchmark_revision") != BENCHMARK_REVISION:
        raise ValueError("Benchmark revision mismatch")
    if manifest.get("manifest_content_sha256") != content_hash(manifest):
        raise ValueError("Pilot manifest content hash mismatch")
    expected_sources = {
        "classification_sha256": hashlib.sha256(classification_path.read_bytes()).hexdigest(),
        "task_map_sha256": hashlib.sha256(task_map_path.read_bytes()).hexdigest(),
    }
    if any(manifest.get(k) != v for k, v in expected_sources.items()):
        raise ValueError("Pilot manifest source hash mismatch")
    if manifest.get("total_selected") != len(manifest.get("tasks", [])):
        raise ValueError("Pilot count mismatch")
    selection = {suite: [] for suite in SUITE_COUNTS}
    seen = set()
    for row in manifest["tasks"]:
        suite, task_id = row["suite"], row["task_id"]
        if suite not in selection or type(task_id) is not int or not 1 <= task_id <= len(classification[suite]):
            raise ValueError(f"Invalid manifest task: {suite}/{task_id}")
        task = classification[suite][task_id - 1]
        if (task["id"], task["name"], task["category"], task["difficulty_level"]) != (
            task_id, row["task_name"], row["perturbation_category"], row["difficulty_level"]
        ) or row["original_task"] != original_stem(task["name"], task["category"]):
            raise ValueError(f"Pilot metadata mismatch: {suite}/{task_id}")
        if (suite, task_id) in seen:
            raise ValueError(f"Duplicate pilot task: {suite}/{task_id}")
        seen.add((suite, task_id))
        selection[suite].append(task_id - 1)
    return manifest, {s: sorted(indices) for s, indices in selection.items()}


def resume_protocol(existing: dict, proposed: dict, *, legacy: bool) -> dict:
    """Accept only the known legacy collector-source change on old run resume."""
    if existing == proposed:
        return existing
    if legacy and existing.get("collector_sha256") == LEGACY_PLUS_COLLECTOR_SHA256 and {
        k: v for k, v in existing.items() if k != "collector_sha256"
    } == {
        k: v for k, v in proposed.items() if k != "collector_sha256"
    }:
        return existing
    raise ValueError("Resume protocol mismatch")
