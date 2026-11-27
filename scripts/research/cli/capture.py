"""Construct aligned R1 camera pairs and capture the two verified policies."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

from scripts.common.integrity import sha
from scripts.research.core.artifacts import write_json
from scripts.research.core.r1 import execute
from scripts.research.data.camera_pairs import SCHEMA
from scripts.research.models.checkpoint import AuditBlocked

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-3", type=Path, required=True, help="Passing R0-3 run directory")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/nfs4/jhkim/repos/rlinf/checkpoints"))
    parser.add_argument("--assets", type=Path, help="Override installed PLUS asset directory")
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pairs-from", type=Path, help="Reuse an immutable R1-1 PASS run without rendering")
    mode.add_argument("--pairs-only", action="store_true", help="Render and validate physical pairs without CUDA inference")
    args = parser.parse_args()
    if args.pairs_from and args.assets:
        parser.error("--assets applies only when rendering pairs")
    sys.dont_write_bytecode = True
    from scripts.research.core.artifacts import create_run

    out = create_run(ROOT, args.run_id, stage="R1")
    status = {"schema": SCHEMA, "stage": "R1", "status": "PASS",
              "gates": dict.fromkeys(("R1-1", "R1-2", "R1-3", "R1-4"), "BLOCKED"),
              "valid_pairs": 0, "observations": 0, "captured_observations_per_model": 0,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        execute(ROOT, out, args, status)
    except AuditBlocked as exc:
        state = "PARTIAL" if status["gates"]["R1-1"] == "PASS" else "BLOCKED"
        status.update(status=state, reason=str(exc))
    except Exception as exc:
        pending = next(gate for gate, value in status["gates"].items() if value != "PASS")
        status["gates"][pending] = "FAIL"
        status.update(status="FAIL", reason=str(exc))
        (out / "error.log").write_text(traceback.format_exc())
    status["evidence_sha256"] = {p.relative_to(out).as_posix(): sha(p)
                                 for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "status.json", status)
    print(f"{status['status']} {out}")
    if "reason" in status:
        print(status["reason"], file=sys.stderr)
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2, "PARTIAL": 2}[status["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
