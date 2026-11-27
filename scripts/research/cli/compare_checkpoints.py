"""R0-2 CPU comparison of the pinned SFT and GRPO checkpoint tensors."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

from scripts.common.integrity import sha
from scripts.research.core.artifacts import create_run, write_json
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint
from scripts.research.models.diff_figure import plot_modules
from scripts.research.models.parameter_diff import compare
from scripts.research.models.runtime_evidence import load_r0_1
from scripts.research.models.tensor_shards import indexed_tensors

ROOT = Path(__file__).resolve().parents[3]


def write_csv(path, rows):
    columns = dict.fromkeys(key for row in rows for key in row)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args, out, status):
    inventory, metadata, runtime, details = load_r0_1(ROOT, args.r0_1.resolve())
    write_json(out / "provenance.json", details)
    checkpoints = {}
    for role, recorded in metadata.items():
        path = Path(recorded["path"])
        if args.checkpoint_root:
            path = args.checkpoint_root.resolve() / recorded["repo_id"].rsplit("/", 1)[-1]
        pin = {key: recorded[key] for key in ("repo_id", "revision")}
        verified = inspect_checkpoint(path, pin, inventory)
        if verified != dict(recorded, path=str(path)):
            raise ValueError(f"{role} checkpoint evidence disagrees with the authenticated files")
        tensors = indexed_tensors(path, verified["shards"])
        if len(tensors) != verified["indexed_tensor_count"]:
            raise ValueError(f"{role} indexed tensor count disagrees with R0-1")
        checkpoints[role] = tensors
    rows, modules, complete = compare(
        runtime["sft"], runtime["grpo"], checkpoints["sft"], checkpoints["grpo"], dry_run=args.dry_run,
    )
    counts = dict(Counter(row["status"] for row in rows))
    module_counts = {}
    for row in rows:
        module_counts.setdefault(row["module"], Counter())[row["status"]] += 1
    write_json(out / "comparison.json", {
        "counts": counts, "module_tensor_status_counts": module_counts,
        "modules": modules, "tensors": rows,
    })
    write_csv(out / "tensors.csv", rows)
    if modules:
        write_csv(out / "modules.csv", [dict(module=key, **value) for key, value in modules.items()])
    status["tensor_status_counts"] = counts
    if not complete:
        raise ValueError("UNSAFE or missing runtime parameters. See comparison.json.")
    if args.dry_run:
        raise AuditBlocked("Tensor correspondence checked. Full numerical comparison pending.")
    if modules["policy.total"]["parameter_count"] != details["runtime"]["sft"]["elements"]:
        raise ValueError("Measured parameter count differs from verified runtime count")
    figures = ROOT / "reports/research/R0/figures"
    if figures.resolve() != figures:
        raise ValueError("Figure output path must not traverse symlinks")
    figures.mkdir(parents=True, exist_ok=True)
    figure = figures / f"{out.name}-modules.png"
    try:
        plot_modules(figure, modules, details["checkpoints"])
    except ImportError as exc:
        raise AuditBlocked(f"Module figure dependency unavailable: {exc}") from exc
    status["figure"] = str(figure.relative_to(ROOT))
    status["figure_sha256"] = sha(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-1", type=Path, required=True, help="Passing R0-1 evidence directory")
    parser.add_argument("--checkpoint-root", type=Path, help="Override the checkpoint parent in R0-1")
    parser.add_argument("--run-id", required=True, help="New immutable output directory name")
    parser.add_argument("--dry-run", action="store_true", help="Verify files and tensor alignment without computing deltas")
    args = parser.parse_args()
    out = create_run(ROOT, args.run_id)
    status = {"stage": "R0-2", "status": "PASS", "dry_run": args.dry_run,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(args, out, status)
    except (AuditBlocked, FileNotFoundError) as exc:
        status.update(status="BLOCKED", reason=str(exc))
    except Exception as exc:
        status.update(status="FAIL", reason=str(exc))
        (out / "error.log").write_text(traceback.format_exc())
    status["evidence_sha256"] = {path.name: sha(path) for path in sorted(out.iterdir())}
    write_json(out / "status.json", status)
    print(f"{status['status']} {out}")
    if "reason" in status:
        print(status["reason"], file=sys.stderr)
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[status["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
