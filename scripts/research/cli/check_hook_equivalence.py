"""R0-3 replay of one authenticated Phase 3A observation per policy."""

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import traceback

import numpy as np

from scripts.common.integrity import array_hash, sha
from scripts.research.core.artifacts import create_run, write_json
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint, read_json
from scripts.research.models.rlinf_oft import RLinfOFTAdapter, baseline_config
from scripts.research.models.runtime_evidence import load_r0_1, load_r0_2
from scripts.research.models.parameter_diff import runtime_parameters

ROOT = Path(__file__).resolve().parents[3]


def load_observation(phase3a, observation_id):
    index = phase3a / "replay_bank/index.json"
    entries = read_json(index)["observations"]
    selected = [row for row in entries if row["observation_id"] == observation_id] if observation_id else entries[:1]
    if len(selected) != 1:
        raise ValueError("Expected one matching saved observation")
    entry, = selected
    path = (phase3a / entry["file"]).resolve()
    if not path.is_relative_to(phase3a.resolve()) or sha(path) != entry["file_sha256"]:
        raise ValueError("Phase 3A observation path or hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        bank = {key: arrays[key] for key in arrays.files}
    if set(bank) != entry["arrays"].keys() or any(
        array_hash(value) != entry["arrays"][key]["sha256"] for key, value in bank.items()
    ):
        raise ValueError("Phase 3A observation array inventory or hash mismatch")
    return entry, bank


def run(args, out):
    inventory, metadata, runtime, details = load_r0_1(ROOT, args.r0_1.resolve())
    gate = load_r0_2(args.r0_2.resolve(), details["r0_1_sha256"], runtime, details)
    config = read_json(ROOT / "configs/research/checkpoint_audit.json")
    phase3a = (ROOT / config["phase3a_manifest"]).parent
    entry, bank = load_observation(phase3a, args.observation_id)
    reference_path = ROOT / "scripts/analysis/phase3b_distribution.py"
    write_json(out / "provenance.json", {
        "r0_2": str(args.r0_2.resolve()), "r0_2_gate": gate, "r0_1": details["r0_1"],
        "reference_source_sha256": sha(reference_path),
        "observation_id": entry["observation_id"], "observation_sha256": entry["file_sha256"],
    })
    if args.dry_run:
        raise AuditBlocked("Evidence and observation verified. Real CUDA inference pending.")

    import torch
    from scripts.analysis.phase3b import UPSTREAM, command
    from scripts.analysis.phase3b_distribution import capture_native_logits
    from scripts.research.capture.equivalence import evaluate

    if not torch.cuda.is_available():
        raise AuditBlocked("CUDA unavailable for the native policy")
    upstream = ROOT / "upstream/RLinf"
    revision = command(["git", "rev-parse", "HEAD"], cwd=upstream)
    tracked = command(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                      cwd=upstream)
    if revision["returncode"] or revision["stdout"] != UPSTREAM or tracked["returncode"] or tracked["stdout"]:
        raise ValueError("Upstream RLinf differs from the pinned Phase 3B checkout")
    protocols = read_json(ROOT / config["phase3a_manifest"])["protocols"]
    for role, recorded in metadata.items():
        path = Path(recorded["path"])
        if args.checkpoint_root:
            path = args.checkpoint_root.resolve() / recorded["repo_id"].rsplit("/", 1)[-1]
        if inspect_checkpoint(path, config[role], inventory) != dict(recorded, path=str(path)):
            raise ValueError(f"{role} checkpoint differs from R0-1 evidence")
        adapter = RLinfOFTAdapter.load(upstream, baseline_config(protocols[role], config[role], path))
        try:
            actual = adapter.registry.summarize()
            params = runtime_parameters(actual, details["architecture"]["timm_model_ids"])
            topology = {key: (value["path"], value["type"]) for key, value in actual["modules"].items()}
            if params != runtime[role] or topology != details["module_topology"]:
                raise ValueError(f"{role} loaded runtime differs from R0-1")
            report, logits = evaluate(adapter, bank, entry["source_metadata"]["language"],
                                      entry["observation_id"], capture_native_logits)
            write_json(out / f"{role}-equivalence.json", report)
            with (out / f"{role}-action-logits.npz").open("xb") as stream:
                np.savez_compressed(stream, **logits)
            if not report["passed"]:
                raise ValueError(f"{role} inference divergence; see {role}-equivalence.json")
        finally:
            del adapter
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-1", required=True, type=Path, help="Passing R0-1 evidence directory")
    parser.add_argument("--r0-2", required=True, type=Path, help="Passing R0-2 evidence directory")
    parser.add_argument("--checkpoint-root", type=Path, help="Override the recorded checkpoint parent")
    parser.add_argument("--observation-id", help="Phase 3A observation ID; default is the first indexed observation")
    parser.add_argument("--run-id", required=True, help="New immutable output directory name")
    parser.add_argument("--dry-run", action="store_true", help="Validate evidence and observation on CPU")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    out = create_run(ROOT, args.run_id)
    status = {"stage": "R0-3", "status": "PASS", "dry_run": args.dry_run,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(args, out)
    except (AuditBlocked, FileNotFoundError) as exc:
        status.update(status="BLOCKED", reason=str(exc))
    except Exception as exc:
        status.update(status="FAIL", reason=str(exc))
        with (out / "error.log").open("x") as stream:
            stream.write(traceback.format_exc())
    status["evidence_sha256"] = {path.name: sha(path) for path in sorted(out.iterdir())}
    write_json(out / "status.json", status)
    print(f"{status['status']} {out}")
    if "reason" in status:
        print(status["reason"], file=sys.stderr)
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[status["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
