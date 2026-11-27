"""R0-1 checkpoint preflight and bounded native structure smoke."""

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from scripts.common.integrity import sha
from scripts.research.core.artifacts import create_run, write_json
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint, read_json
from scripts.research.models.rlinf_oft import MODEL_SHA, MODEL_SOURCE, RLinfOFTAdapter, baseline_config

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_PIN = "e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d"


def run(args, out):
    config = read_json(args.config)
    manifest_path = ROOT / config["phase3a_manifest"]
    protocols = read_json(manifest_path)["protocols"]
    inventory_path = ROOT / "manifests/models_130.json"
    inventory = read_json(inventory_path)
    upstream = ROOT / "upstream/RLinf"
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    action_source_sha = sha(upstream / MODEL_SOURCE)
    if revision != UPSTREAM_PIN or action_source_sha != MODEL_SHA:
        raise ValueError("Upstream revision or action source differs from Phase 3B")
    upstream_status = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(upstream), "status", "--porcelain",
         "--untracked-files=no"],
        text=True,
    )
    if upstream_status.strip():
        raise ValueError("Tracked files in upstream RLinf differ from the pinned checkout")

    evidence = {
        "upstream_commit": revision, "action_source": MODEL_SOURCE.as_posix(),
        "action_source_sha256": action_source_sha,
        "phase3a_manifest_sha256": sha(manifest_path), "inventory_sha256": sha(inventory_path),
        "host": platform.platform(), "policies": {},
    }
    blocked = []
    try:
        for role in ("sft", "grpo"):
            pin, protocol = config[role], protocols[role]
            path = Path(protocol["checkpoint_path"])
            if args.checkpoint_root:
                path = args.checkpoint_root.resolve() / pin["repo_id"].rsplit("/", 1)[-1]
            cfg = baseline_config(protocol, pin, path)
            record = evidence["policies"][role] = {
                "expected_identity": pin, "loading_config": cfg,
            }
            try:
                record["checkpoint"] = inspect_checkpoint(path, pin, inventory)
            except AuditBlocked as exc:
                record["blocker"] = str(exc)
                blocked.append(str(exc))
        if blocked:
            raise AuditBlocked("; ".join(blocked))
        if args.dry_run:
            raise AuditBlocked("CPU preflight finished. Runtime smoke required for R0-1 PASS.")

        sys.dont_write_bytecode = True
        for role, record in evidence["policies"].items():
            adapter = RLinfOFTAdapter.load(upstream, record["loading_config"])
            try:
                write_json(out / f"{role}-modules.json", adapter.registry.summarize())
                record["smoke"] = adapter.smoke(record["loading_config"])
            finally:
                del adapter
                gc.collect()
                import torch

                torch.cuda.empty_cache()
    finally:
        write_json(out / "structure.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/checkpoint_audit.json")
    parser.add_argument("--checkpoint-root", type=Path, help="Override the Phase 3A checkpoint parent")
    parser.add_argument("--run-id", required=True, help="New immutable output directory name")
    parser.add_argument("--dry-run", action="store_true", help="Verify source and checkpoint files without loading a model")
    args = parser.parse_args()
    out = create_run(ROOT, args.run_id)
    result = {"stage": "R0-1", "status": "PASS", "dry_run": args.dry_run,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(args, out)
    except AuditBlocked as exc:
        result.update(status="BLOCKED", reason=str(exc))
    except Exception as exc:
        result.update(status="FAIL", reason=str(exc))
        (out / "error.log").write_text(traceback.format_exc())
    write_json(out / "status.json", result)
    print(f"{result['status']} {out}")
    if "reason" in result:
        print(result["reason"], file=sys.stderr)
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
