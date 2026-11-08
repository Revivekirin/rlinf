"""Verify LIBERO-130 task identity against Goal, Spatial, and Long suites.

Run in the pinned RLinf virtual environment from the RLinf checkout.  The
comparison uses the installed benchmark's task metadata and the actual first
five init-state arrays; it never assumes a task-index offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SUITES = ("libero_goal", "libero_spatial", "libero_10")
EXPECTED_COUNTS = {"libero_130": 130, **dict.fromkeys(SUITES, 10)}


def state_hash(state: object) -> str:
    array = np.ascontiguousarray(np.asarray(state))
    if array.dtype.hasobject:
        raise TypeError("Object-valued init state cannot be hashed safely")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def task_identity(task: object) -> tuple[str, str, str, str]:
    return (
        str(task.name),
        str(task.problem_folder),
        str(task.bddl_file),
        str(task.init_states_file),
    )


def audit(factory, *, trials: int = 5) -> dict:
    benchmarks = {name: factory(name)() for name in EXPECTED_COUNTS}
    for name, expected in EXPECTED_COUNTS.items():
        actual = benchmarks[name].get_num_tasks()
        if actual != expected:
            raise ValueError(f"{name}: expected {expected} tasks, found {actual}")

    all_suite = benchmarks["libero_130"]
    by_identity: dict[tuple[str, str, str, str], list[int]] = {}
    for tid in range(all_suite.get_num_tasks()):
        by_identity.setdefault(task_identity(all_suite.get_task(tid)), []).append(tid)

    rows = []
    for name in SUITES:
        suite = benchmarks[name]
        for subset_tid in range(suite.get_num_tasks()):
            task = suite.get_task(subset_tid)
            identity = task_identity(task)
            candidates = by_identity.get(identity, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"{name} task {subset_tid}: expected unique 130 identity, "
                    f"found {candidates}: {identity}"
                )
            all_tid = candidates[0]
            subset_states = suite.get_task_init_states(subset_tid)
            all_states = all_suite.get_task_init_states(all_tid)
            if len(subset_states) != len(all_states) or len(subset_states) < trials:
                raise ValueError(
                    f"{name} task {subset_tid}: init-state count mismatch or "
                    f"fewer than {trials} trials"
                )
            hashes = []
            for trial in range(trials):
                left = state_hash(subset_states[trial])
                right = state_hash(all_states[trial])
                if left != right:
                    raise ValueError(
                        f"{name} task {subset_tid} -> 130 task {all_tid}, "
                        f"trial {trial}: init-state contents differ"
                    )
                hashes.append(left)
            rows.append(
                {
                    "suite": name,
                    "subset_task_id": subset_tid,
                    "libero_130_task_id": all_tid,
                    "task_name": identity[0],
                    "problem_folder": identity[1],
                    "bddl_file": identity[2],
                    "init_states_file": identity[3],
                    "language": str(task.language),
                    "num_init_states": len(subset_states),
                    "trial_sha256": hashes,
                }
            )
    return {"schema_version": 1, "trials_compared": trials, "tasks": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if os.environ.get("LIBERO_TYPE", "standard").lower() != "standard":
        parser.error("mapping audit requires LIBERO_TYPE=standard")

    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    result = audit(get_benchmark_overridden, trials=args.trials)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Verified {len(result['tasks'])} task mappings: {args.output}")


if __name__ == "__main__":
    main()
