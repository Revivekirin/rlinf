"""Nominal episode naming and canonical protocol fingerprint."""

import hashlib
import json


def protocol_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def episode_id(suite: str, task_id: int, trial_id: int) -> str:
    if suite not in {"libero_goal", "libero_spatial", "libero_10", "libero_object"}:
        raise ValueError(f"Unsupported suite: {suite}")
    if task_id < 0 or trial_id < 0:
        raise ValueError("Task and trial IDs must be nonnegative")
    return f"{suite}-t{task_id:02d}-i{trial_id:02d}"
