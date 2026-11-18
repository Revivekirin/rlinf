"""Check the pinned official LIBERO-PLUS source before any GPU evaluation.

This check does not import the simulator or modify the Python environment.
Use --require-assets on the evaluation host after extracting assets.zip.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import subprocess
from pathlib import Path

REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = {
    "Background Textures", "Camera Viewpoints", "Language Instructions",
    "Light Conditions", "Objects Layout", "Robot Initial States", "Sensor Noise",
}
ASSET_DIRS = (
    "articulated_objects", "new_objects", "scenes", "stable_hope_objects",
    "stable_scanned_objects", "textures", "turbosquid_objects",
)


def inspect(root: Path, require_assets: bool) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"LIBERO-PLUS revision mismatch: {revision}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("LIBERO-PLUS checkout has local changes")

    core = root / "libero" / "libero"
    source = ast.parse((core / "benchmark" / "libero_suite_task_map.py").read_text())
    task_map = ast.literal_eval(source.body[0].value)
    classification = json.loads((core / "benchmark" / "task_classification.json").read_text())
    if set(classification) != set(SUITES):
        raise ValueError(f"Unexpected classified suites: {sorted(classification)}")

    counts = {}
    categories = collections.Counter()
    unspecified_difficulty = collections.Counter()
    missing_bddl = collections.Counter()
    missing_examples = []
    for suite in SUITES:
        names = task_map[suite]
        rows = classification[suite]
        if len(names) != len(set(names)) or len(rows) != len(names):
            raise ValueError(f"Duplicate or missing tasks in {suite}")
        if set(names) != {row["name"] for row in rows}:
            raise ValueError(f"Classification/task map mismatch in {suite}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate classification IDs in {suite}")
        for row in rows:
            if row["category"] not in CATEGORIES:
                raise ValueError(f"Unknown shift category: {row['category']}")
            if row["difficulty_level"] is None:
                unspecified_difficulty[suite] += 1
            elif not isinstance(row["difficulty_level"], int):
                raise ValueError(f"Invalid difficulty: {row['name']}")
            # Camera / robot / noise variants are encoded in a virtual BDDL
            # name; ControlEnv strips the _view_ suffix before opening it.
            source_name = row["name"].split("_view_", 1)[0]
            bddl = core / "bddl_files" / suite / (source_name + ".bddl")
            if not bddl.is_file():
                missing_bddl[suite] += 1
                if len(missing_examples) < 5:
                    missing_examples.append(str(bddl))
            categories[row["category"]] += 1
        counts[suite] = len(rows)

    if sum(counts.values()) != 10030 or set(categories) != CATEGORIES:
        raise ValueError("Official task or category coverage differs from 10,030 / seven")
    if missing_bddl:
        raise FileNotFoundError(
            f"Classification references missing BDDL files: {dict(missing_bddl)}; "
            f"unspecified difficulty: {dict(unspecified_difficulty)}; "
            f"examples: {missing_examples}"
        )
    if require_assets:
        for name in ASSET_DIRS:
            path = core / "assets" / name
            if not path.is_dir() or not any(path.iterdir()):
                raise FileNotFoundError(f"Missing or empty asset directory: {path}")
    return {
        "revision": revision, "suite_task_counts": counts,
        "shift_category_counts": dict(sorted(categories.items())),
        "unspecified_difficulty_counts": dict(unspecified_difficulty),
        "expected_episodes_per_condition": sum(counts.values()),
        "assets_checked": require_assets,
        "package_import_name": "libero",
        "rlinf_plus_import_name": "liberoplus",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--require-assets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect(args.source.resolve(), args.require_assets), indent=2))


if __name__ == "__main__":
    main()
