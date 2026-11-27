"""Run or resume audited R2-A group/alpha interventions."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

from scripts.research.core.artifacts import create_run, write_json
from scripts.research.experiments.module_rollback import execute
from scripts.research.models.checkpoint import AuditBlocked

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/nfs4/jhkim/repos/rlinf/checkpoints"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "coarse", "full"), default="full")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--resume", type=Path, help="Existing run directory with immutable completed units")
    args = parser.parse_args()
    if args.max_units is not None and args.max_units <= 0:
        parser.error("--max-units must be positive")
    sys.dont_write_bytecode = True
    if args.resume:
        out = args.resume.resolve()
        if out.parent != ROOT / "outputs/research/R2A" or out.name != args.run_id or not (out / "manifest.json").is_file():
            parser.error("--resume must point to the named existing R2A run")
    else:
        out = create_run(ROOT, args.run_id, stage="R2A")
    attempts = out / "attempts"
    attempts.mkdir(exist_ok=True)
    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    try:
        status = execute(ROOT, out, args.r1.resolve(), args.mode, args.checkpoint_root.resolve(),
                         args.preflight, args.max_units)
    except (AuditBlocked, FileNotFoundError) as exc:
        status = {"status": "BLOCKED", "reason": str(exc)}
    except Exception as exc:
        status = {"status": "FAIL", "reason": str(exc), "traceback": traceback.format_exc()}
    status["started_utc"] = run_time
    write_json(attempts / (run_time + ".json"), status)
    print(status["status"], out, status.get("reason", ""))
    return {"BLOCKED": 2, "FAIL": 1, "PARTIAL": 2}.get(status["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
