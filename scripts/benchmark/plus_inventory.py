"""Inventory the pinned LIBERO-PLUS registry without importing its simulator.

Run from the repository root with ``python -m scripts.benchmark.plus_inventory``.
The official classification uses one-based IDs; the task map uses list indices.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.common.io import csv_save, save


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "upstream/LIBERO-plus/libero/libero/benchmark"
CLASSIFICATION = BENCHMARK / "task_classification.json"
TASK_MAP = BENCHMARK / "libero_suite_task_map.py"
OUTPUT = ROOT / "outputs/benchmark/libero_plus/inventory"
SUITE_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
CATEGORIES = (
    "Objects Layout", "Camera Viewpoints", "Robot Initial States",
    "Language Instructions", "Light Conditions", "Background Textures",
    "Sensor Noise",
)
# Registry names encode perturbation parameters, but these suffixes are not
# documented as a complete taxonomy of the claimed 21 subtypes.
SUFFIX = {
    "Background Textures": re.compile(r"_(?:table|tb)_\d+$"),
    "Robot Initial States": re.compile(r"_view_[0-9_]+_initstate_\d+$"),
    "Camera Viewpoints": re.compile(r"_view_[0-9_]+_initstate_\d+$"),
    "Language Instructions": re.compile(r"_language_\d+_view_[0-9_]+_initstate_\d+$"),
    "Sensor Noise": re.compile(r"_view_[0-9_]+_initstate_\d+_noise_\d+$"),
    "Objects Layout": re.compile(r"_(?:add_\d+|(?:moved_)?level\d+_sample\d+)$"),
    "Light Conditions": re.compile(r"_light_\d+$"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_task_map(path: Path) -> dict[str, list[str]]:
    """Read the official literal registry without executing benchmark imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "libero_task_map"
                       for t in node.targets)]
    if len(matches) != 1:
        raise ValueError("Expected one literal libero_task_map assignment")
    value = ast.literal_eval(matches[0])
    if not isinstance(value, dict):
        raise ValueError("Invalid task map")
    return value


def original_stem(name: str, category: str) -> str:
    match = SUFFIX[category].search(name)
    if not match or match.start() == 0:
        raise ValueError(f"Unrecognized official name suffix: {category}: {name}")
    return name[:match.start()]


def analyze(classification: dict, task_map: dict,
            expected_counts: dict[str, int] = SUITE_COUNTS) -> dict:
    """Validate registry alignment and return ordered, JSON-safe inventory."""
    if set(classification) != set(expected_counts):
        raise ValueError(f"Classification suites differ: {set(classification) ^ set(expected_counts)}")
    if not set(expected_counts).issubset(task_map):
        raise ValueError("Task map is missing a benchmark suite")
    rows_out = []
    by_suite_category = Counter()
    by_original_category = Counter()
    by_category_difficulty = Counter()
    original_counts = {}
    missing_difficulty = Counter()
    for suite, expected in expected_counts.items():
        rows = classification[suite]
        names = task_map[suite]
        if not isinstance(rows, list) or not isinstance(names, list):
            raise ValueError(f"Invalid suite list: {suite}")
        if len(names) != expected or len(rows) != expected:
            raise ValueError(f"Task count mismatch: {suite}: classification={len(rows)}, map={len(names)}, expected={expected}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate registry name: {suite}")
        seen_ids = set()
        seen_names = set()
        stems = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid classification row: {suite}")
            task_id = row.get("id")
            if type(task_id) is not int or task_id < 1 or task_id > expected:
                raise ValueError(f"Invalid task ID: {suite}: {task_id}")
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task ID: {suite}: {task_id}")
            seen_ids.add(task_id)
            name = row.get("name")
            if not isinstance(name, str) or not name or name != names[task_id - 1]:
                raise ValueError(f"Missing/mismatched registry task: {suite}: {task_id}")
            if name in seen_names:
                raise ValueError(f"Duplicate classification name: {suite}: {name}")
            seen_names.add(name)
            category = row.get("category")
            if category not in CATEGORIES:
                raise ValueError(f"Invalid category: {suite}: {task_id}: {category}")
            difficulty = row.get("difficulty_level")
            if difficulty is None:
                missing_difficulty[suite, category] += 1
            elif type(difficulty) is not int or not 1 <= difficulty <= 5:
                raise ValueError(f"Invalid difficulty: {suite}: {task_id}: {difficulty}")
            stem = original_stem(name, category)
            stems.add(stem)
            rows_out.append({"suite": suite, "task_id": task_id,
                             "registry_index": task_id - 1, "name": name,
                             "original_task": stem, "category": category,
                             "difficulty_level": difficulty})
            by_suite_category[suite, category] += 1
            by_original_category[suite, stem, category] += 1
            by_category_difficulty[category, difficulty] += 1
        missing = set(range(1, expected + 1)) - seen_ids
        if missing:
            raise ValueError(f"Missing task IDs: {suite}: {sorted(missing)[:10]}")
        if len(stems) != 10:
            raise ValueError(f"Expected 10 inferred original tasks: {suite}: {len(stems)}")
        original_counts[suite] = len(stems)
    if len(rows_out) != 10030 and expected_counts == SUITE_COUNTS:
        raise ValueError(f"Expected 10030 tasks, found {len(rows_out)}")
    rows_out.sort(key=lambda r: (list(expected_counts).index(r["suite"]), r["task_id"]))
    return {
        "total_tasks": len(rows_out),
        "suite_counts": {s: expected_counts[s] for s in expected_counts},
        "inferred_original_task_counts": original_counts,
        "subtype_status": "not_verifiable_as_21_official_subtypes",
        "missing_difficulty_count": sum(missing_difficulty.values()),
        "missing_difficulty": [
            {"suite": s, "category": c, "count": missing_difficulty[s, c]}
            for s in expected_counts for c in CATEGORIES if missing_difficulty[s, c]],
        "tasks": rows_out,
        "suite_category": [
            {"suite": s, "category": c, "count": by_suite_category[s, c]}
            for s in expected_counts for c in CATEGORIES],
        "suite_original_category": [
            {"suite": s, "original_task": stem, "category": c,
             "count": by_original_category[s, stem, c]}
            for s in expected_counts
            for stem in sorted({r["original_task"] for r in rows_out if r["suite"] == s})
            for c in CATEGORIES],
        "category_difficulty": [
            {"category": c, "difficulty_level": d,
             "count": by_category_difficulty[c, d]}
            for c in CATEGORIES for d in (*range(1, 6), None)],
    }


def summary_markdown(inventory: dict) -> str:
    lines = ["# LIBERO-PLUS metadata inventory", "",
             f"Total classified tasks: **{inventory['total_tasks']:,}**", "",
             "| Suite | Tasks | Inferred original tasks |", "| --- | ---: | ---: |"]
    for suite, count in inventory["suite_counts"].items():
        lines.append(f"| {suite} | {count:,} | {inventory['inferred_original_task_counts'][suite]} |")
    lines += ["", "## Suite × category", "", "| Suite | Category | Count |",
              "| --- | --- | ---: |"]
    for row in inventory["suite_category"]:
        lines.append(f"| {row['suite']} | {row['category']} | {row['count']} |")
    lines += ["", "## Metadata limits", "",
              "- `original_task` is inferred from suffixes in the official task map and cross-checked against classification names; it is not an explicit classification field.",
              "- The classification contains `id`, `name`, `category`, and `difficulty_level`. It does not contain a perturbation subtype field.",
              "- Name suffixes encode some perturbation parameters, but the checked source does not give a complete, task-level mapping to 21 officially named subtypes. No subtype labels or counts are inferred.",
              f"- Missing difficulty: {inventory['missing_difficulty_count']} tasks. These are retained with null difficulty and counted separately.",
              "", "## Provenance", ""]
    for key, value in inventory["provenance"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_inventory(output: Path, inventory: dict) -> None:
    """Create all reports exclusively; refuse to alter any existing report."""
    files = ["inventory.json", "tasks.csv", "suite_category.csv",
             "suite_original_category.csv", "category_difficulty.csv", "SUMMARY.md"]
    if any((output / name).exists() for name in files):
        raise FileExistsError(f"Inventory output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save(output / "inventory.json", inventory)
    for name, key in (("tasks.csv", "tasks"),
                      ("suite_category.csv", "suite_category"),
                      ("suite_original_category.csv", "suite_original_category"),
                      ("category_difficulty.csv", "category_difficulty")):
        csv_save(output / name, inventory[key])
    (output / "SUMMARY.md").write_text(summary_markdown(inventory), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, default=CLASSIFICATION)
    parser.add_argument("--task-map", type=Path, default=TASK_MAP)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    task_map = load_task_map(args.task_map)
    inventory = analyze(classification, task_map)
    inventory["provenance"] = {
        "classification_sha256": sha256(args.classification),
        "task_map_sha256": sha256(args.task_map),
        "analysis_code_sha256": sha256(Path(__file__)),
        "classification_source": str(args.classification),
        "task_map_source": str(args.task_map),
    }
    write_inventory(args.output, inventory)
    print(summary_markdown(inventory))
    print(f"Reports: {args.output}")


if __name__ == "__main__":
    main()
