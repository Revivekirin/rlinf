"""Atomic episode artifacts and summaries for the nominal LIBERO collector."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ARTIFACTS = ("metadata.json", "states.npz", "actions.npz", "observations.npz")


def episode_id(suite: str, task_id: int, trial_id: int) -> str:
    if suite not in {"libero_goal", "libero_spatial", "libero_10"}:
        raise ValueError(f"Unsupported suite: {suite}")
    if task_id < 0 or trial_id < 0:
        raise ValueError("Task and trial IDs must be nonnegative")
    return f"{suite}-t{task_id:02d}-i{trial_id:02d}"


def read_complete(
    path: Path, *, protocol_hash: str | None = None, expected_id: str | None = None
) -> dict:
    if not path.is_dir() or any(not (path / name).is_file() for name in ARTIFACTS):
        raise ValueError(f"Incomplete episode artifact: {path}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("complete") is not True:
        raise ValueError(f"Episode not marked complete: {path}")
    if metadata.get("episode_id") != (expected_id or path.name):
        raise ValueError(f"Episode ID mismatch: {path}")
    if protocol_hash is not None and metadata.get("protocol_hash") != protocol_hash:
        raise ValueError(f"Protocol hash mismatch: {path}")
    for name in ARTIFACTS[1:]:
        with np.load(path / name, allow_pickle=False) as data:
            if not data.files:
                raise ValueError(f"Empty array artifact: {path / name}")
    return metadata


def write_episode(
    run_dir: Path,
    metadata: dict,
    states: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
) -> Path:
    root = run_dir / "rollouts"
    root.mkdir(parents=True, exist_ok=True)
    target = root / metadata["episode_id"]
    if target.exists():
        raise FileExistsError(target)
    temp = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    try:
        np.savez_compressed(temp / "states.npz", **states)
        np.savez_compressed(temp / "actions.npz", **actions)
        np.savez_compressed(temp / "observations.npz", **observations)
        (temp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        read_complete(
            temp,
            protocol_hash=metadata["protocol_hash"],
            expected_id=metadata["episode_id"],
        )
        if target.exists():
            raise FileExistsError(target)
        os.rename(temp, target)
    except BaseException:
        # Preserve an interrupted partial directory for diagnosis; never
        # promote it to a completed episode or silently overwrite the target.
        raise
    return target


def summarize_run(run_dir: Path, expected: set[str], protocol_hash: str) -> dict:
    root = run_dir / "rollouts"
    rows = []
    if root.exists():
        for path in sorted(root.iterdir()):
            if path.name.startswith(".partial-"):
                continue
            if not path.is_dir():
                raise ValueError(f"Unexpected rollout entry: {path}")
            metadata = read_complete(path, protocol_hash=protocol_hash)
            if path.name not in expected:
                raise ValueError(f"Unexpected episode: {path.name}")
            rows.append(metadata)
    completed = {row["episode_id"] for row in rows}
    missing = sorted(expected - completed)
    successes = sum(bool(row["success"]) for row in rows)
    by_suite = defaultdict(lambda: {"completed": 0, "successes": 0})
    for row in rows:
        item = by_suite[row["suite"]]
        item["completed"] += 1
        item["successes"] += bool(row["success"])
    summary = {
        "expected": len(expected),
        "completed": len(rows),
        "missing_episode_ids": missing,
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else None,
        "mean_episode_length": (
            sum(int(row["episode_length"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "by_suite": dict(by_suite),
        "dataset_bytes": sum(
            file.stat().st_size for path in root.glob("*") if path.is_dir()
            for file in path.iterdir() if file.is_file()
        ) if root.exists() else 0,
    }
    csv_path = run_dir / "episodes.csv"
    fields = [
        "episode_id", "suite", "task_id", "trial_id", "success",
        "episode_length", "termination_reason", "checkpoint_repo",
        "checkpoint_revision", "sampling_mode", "protocol_hash",
    ]
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    os.replace(tmp_csv, csv_path)
    tmp_json = run_dir / "summary.json.tmp"
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_json, run_dir / "summary.json")
    return summary
