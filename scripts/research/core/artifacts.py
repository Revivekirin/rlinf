"""Exclusive creation of research evidence."""

import json
import re
from pathlib import Path


def create_run(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Run ID must be a single path component")
    parent = root / "outputs/research/R0"
    if parent.resolve() != parent:
        raise ValueError("Research output path must not traverse symlinks")
    path = parent / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
