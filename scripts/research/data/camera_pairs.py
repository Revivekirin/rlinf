"""Plan and verify reusable, same-simulator camera replays."""

from dataclasses import dataclass
import re

import numpy as np

from scripts.common.integrity import array_hash, sha
from scripts.research.models.checkpoint import read_json

# Canonical LIBERO suite IDs, cross-checked against the original BDDL stems.
TASKS = (
    ("libero_spatial", 1, "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"),
    ("libero_object", 0, "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_goal", 0, "open_the_middle_drawer_of_the_cabinet"),
)
SCHEMA = 1
VIEWS = 4
TRIALS = 3
OBSERVATIONS = len(TASKS) * TRIALS * VIEWS
PAIRS = len(TASKS) * TRIALS * (VIEWS - 1)


@dataclass(frozen=True)
class CameraPairSpec:
    suite: str
    task_id: int
    task: str
    variants: tuple


def plan(plus):
    classification = read_json(plus / "benchmark/task_classification.json")
    result = []
    for suite, task_id, task in TASKS:
        pattern = re.compile(re.escape(task) + r"_view_((?:-?\d+_){4}-?\d+)_initstate_0$")
        variants = []
        for row in classification[suite]:
            match = pattern.fullmatch(row["name"])
            if row["category"] == "Camera Viewpoints" and match:
                variants.append(dict(row, parameters=list(map(int, match[1].split("_")))))
        by_severity = {}
        for row in sorted(variants, key=lambda row: (row["difficulty_level"], row["id"])):
            by_severity.setdefault(row["difficulty_level"], row)
        levels = sorted(by_severity)
        if len(levels) < VIEWS - 1:
            raise ValueError(f"Insufficient distinct camera severities for {task}")
        result.append(CameraPairSpec(suite, task_id, task, tuple(
            by_severity[i] for i in (levels[0], levels[len(levels)//2], levels[-1])
        )))
    return result


def save_arrays(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "file": path.name, "sha256": sha(path),
        "arrays": {key: {"sha256": array_hash(value), "shape": list(value.shape),
                        "dtype": str(value.dtype)} for key, value in arrays.items()},
    }


def load_arrays(directory, record):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.npz", record["file"]):
        raise ValueError("Invalid array file name")
    path = directory / record["file"]
    if path.is_symlink() or sha(path) != record["sha256"]:
        raise ValueError(f"File hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    if arrays.keys() != record["arrays"].keys():
        raise ValueError(f"Unexpected arrays: {path}")
    for name, value in arrays.items():
        if record["arrays"][name] != {"sha256": array_hash(value), "shape": list(value.shape),
                                       "dtype": str(value.dtype)} or not np.isfinite(value).all():
            raise ValueError(f"Invalid array {name} in {path}")
    return arrays


def validate_pair(left, right, a, b):
    if (left["state_id"] != right["state_id"] or left["task"] != right["task"]
            or left["language"] != right["language"] or left["controller"] != right["controller"]
            or left["source"] != right["source"]
            or left["model_sha256"] != right["model_sha256"]
            or left["camera_id"] != "canonical" or right["camera_id"] == "canonical"
            or left["camera"] == right["camera"]):
        raise ValueError("Pair does not preserve scene, instruction and controller with a camera-only change")
    state_keys = {k for k in a if k.startswith("state_")}
    if not {"state_flat", "state_time", "state_qpos", "state_qvel", "state_xpos", "state_xquat"} <= state_keys:
        raise ValueError("Missing physical state evidence")
    expected = state_keys | {"main_image_policy", "proprio_policy"}
    if expected != a.keys() or expected != b.keys() or a["proprio_policy"].shape != (8,):
        raise ValueError("Pair has different state inventories")
    for key in state_keys | {"proprio_policy"}:
        if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
            raise ValueError(f"Physical state mismatch: {key}")
    for image in (a["main_image_policy"], b["main_image_policy"]):
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError("Unexpected model RGB shape or dtype")
    if np.array_equal(a["main_image_policy"], b["main_image_policy"]):
        raise ValueError("Camera perturbation produced identical RGB")
