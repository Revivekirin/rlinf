"""Exclusive creation of research evidence."""

import json
import re
from pathlib import Path


def create_run(root: Path, run_id: str, *, stage: str = "R0") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Run ID must be a single path component")
    if stage not in {"R0", "R1", "R2A"}:
        raise ValueError("Unsupported research stage")
    parent = root / "outputs/research" / stage
    if parent.resolve() != parent:
        raise ValueError("Research output path must not traverse symlinks")
    path = parent / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
