"""R1 stage orchestration and measured R0 gate."""

import gc
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

import numpy as np

from scripts.common.integrity import sha
from scripts.research.cli.check_hook_equivalence import load_observation
from scripts.research.core.artifacts import write_json
from scripts.research.data.camera_pairs import OBSERVATIONS, PAIRS, SCHEMA, load_arrays, save_arrays
from scripts.research.data.simulator_state import collect
from scripts.research.data.validation import comparisons, verify_capture, verify_dataset
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint, read_json
from scripts.research.models.rlinf_oft import RLinfOFTAdapter, baseline_config
from scripts.research.models.runtime_evidence import load_r0_1, load_r0_2


def prior_evidence(root, r0):
    status = read_json(r0 / "status.json")
    if (status["stage"], status["status"], status["dry_run"]) != ("R0-3", "PASS", False):
        raise ValueError("R1 requires a real R0-3 PASS")
    expected = {"provenance.json", "sft-equivalence.json", "grpo-equivalence.json",
                "sft-action-logits.npz", "grpo-action-logits.npz"}
    if set(status["evidence_sha256"]) != expected:
        raise ValueError("Incomplete R0-3 evidence")
    for key, digest in status["evidence_sha256"].items():
        if sha(r0 / key) != digest:
            raise ValueError(f"R0-3 evidence hash mismatch: {key}")
    provenance = read_json(r0 / "provenance.json")
    prior = root / "outputs/research/R0"
    inventory, checkpoints, runtime, details = load_r0_1(root, prior / Path(provenance["r0_1"]).name)
    result = load_r0_2(prior / Path(provenance["r0_2"]).name, details["r0_1_sha256"], runtime, details)
    if result != provenance["r0_2_gate"] or sha(root / "scripts/analysis/phase3b_distribution.py") != provenance["reference_source_sha256"]:
        raise ValueError("R0 evidence provenance mismatch")
    references = {role: read_json(r0 / f"{role}-equivalence.json") for role in checkpoints}
    for role, ref in references.items():
        if not ref["passed"] or not ref["repeated_hook_metadata_equal"] or ref["do_sample"] or ref["atol"] or ref["rtol"]:
            raise ValueError(f"Missing measured inference equivalence for {role}")
        with np.load(r0 / f"{role}-action-logits.npz", allow_pickle=False) as archive:
            baseline = archive["phase3b"]
            if (baseline.shape != (56, 256) or not np.isfinite(baseline).all()
                    or not all(np.array_equal(baseline, archive[key]) for key in ("hooked", "hooked_repeat"))
                    or not np.array_equal((baseline.argmax(-1) + 31744).reshape(8, 7), ref["selected_token_ids"])):
                raise ValueError(f"R0 saved logit comparison mismatch for {role}")
    return inventory, checkpoints, details, provenance, references


def capture(root, out, r0, prior, dataset, checkpoint_root):
    try:
        import torch
    except ImportError as exc:
        raise AuditBlocked(f"Native runtime unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise AuditBlocked("CUDA unavailable for R1 capture")
    from scripts.analysis import phase3b
    from scripts.analysis.phase3b_distribution import capture_native_logits
    from scripts.research.capture.equivalence import evaluate
    from scripts.research.capture.activations import collect_observation

    inventory, checkpoints, details, provenance, references = prior
    config = read_json(root / "configs/research/checkpoint_audit.json")
    manifest = root / config["phase3a_manifest"]
    protocols = read_json(manifest)["protocols"]
    entry, bank = load_observation(manifest.parent, provenance["observation_id"])
    phase3b.require(entry["file_sha256"] == provenance["observation_sha256"],
                    "Saved R0 regression observation changed")
    identity = {role: config[role] for role in checkpoints}
    manifest = {"schema": SCHEMA, "checkpoints": identity, "policies": {}, "r0_regression": {}}
    captured = {}
    for role, recorded in checkpoints.items():
        checkpoint = checkpoint_root / recorded["repo_id"].split("/")[-1]
        if inspect_checkpoint(checkpoint, identity[role], inventory) != dict(recorded, path=str(checkpoint)):
            raise ValueError(f"Checkpoint provenance mismatch: {role}")
        adapter = RLinfOFTAdapter.load(root / "upstream/RLinf", baseline_config(protocols[role], identity[role], checkpoint))
        try:
            report, logits = evaluate(adapter, bank, entry["source_metadata"]["language"],
                                      entry["observation_id"], capture_native_logits)
            ref = references[role]
            with np.load(r0 / f"{role}-action-logits.npz", allow_pickle=False) as archived:
                logits_equal = np.array_equal(archived["phase3b"], logits["phase3b"])
            if (not report["passed"] or not logits_equal or any(report[key] != ref[key] for key in (
                "processed_input_sha256", "decoded_actions", "selected_token_ids", "hook_calls"
            ))):
                raise ValueError(f"Native R0 regression mismatch: {role}")
            write_json(out / f"{role}-r0-regression.json", report)
            manifest["r0_regression"][role] = "PASS"
            records = manifest["policies"][role] = {}
            captured[role] = {}
            for obs_id, obs in dataset["observations"].items():
                values = load_arrays(out, obs["data"])
                arrays, record = collect_observation(adapter, values["main_image_policy"], values["proprio_policy"],
                                                     obs["language"], obs_id, ref)
                file = save_arrays(out / f"{role}-{obs_id}.npz", arrays)
                records[obs_id] = dict(record, data=file, observation_id=obs_id,
                                       input_sha256=obs["data"]["sha256"],
                                       pair_ids=sorted(p["pair_id"] for p in dataset["pairs"]
                                                       if obs_id in (p["id"], p["ood"])))
                captured[role][obs_id] = {"actions": arrays["actions"], "action_tokens": arrays["action_tokens"],
                                          "hash": file["sha256"]}
        finally:
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    write_json(out / "captures.json", manifest)
    write_json(out / "comparisons.json", comparisons(dataset, captured, identity))


def sealed_files(run, status):
    files = {p.relative_to(run).as_posix(): p for p in run.rglob("*")
             if p.is_file() and p != run / "status.json"}
    if files.keys() != status["evidence_sha256"].keys():
        raise ValueError("Run evidence inventory mismatch")
    for key, path in files.items():
        if path.is_symlink() or sha(path) != status["evidence_sha256"][key]:
            raise ValueError(f"Run evidence hash mismatch: {key}")
    return files


def import_pairs(source, out, r0_status_sha):
    status_path = source / "status.json"
    status = read_json(status_path)
    if (status["stage"] != "R1" or status["schema"] != SCHEMA or status["status"] != "PARTIAL"
            or status["gates"]["R1-1"] != "PASS"
            or (status["valid_pairs"], status["observations"]) != (PAIRS, OBSERVATIONS)):
        raise ValueError("Pair source must be a sealed PARTIAL run with R1-1 PASS")
    files = sealed_files(source, status)
    if read_json(files["provenance.json"])["r0_3_status_sha256"] != r0_status_sha:
        raise ValueError("Pair source refers to a different R0-3 baseline")
    dataset = verify_dataset(source)
    names = {"plan.json", "dataset.json"} | {row["data"]["file"] for row in dataset["observations"].values()}
    for name in sorted(names):
        with files[name].open("rb") as stream, (out / name).open("xb") as target:
            copyfileobj(stream, target)
        if sha(out / name) != status["evidence_sha256"][name]:
            raise ValueError(f"Copied pair evidence hash mismatch: {name}")
    return {"run_id": source.name, "status_sha256": sha(status_path),
            "dataset_sha256": status["evidence_sha256"]["dataset.json"]}


def execute(root, out, args, status):
    r0 = args.r0_3.resolve()
    prior = prior_evidence(root, r0)
    provenance = {
        "schema": SCHEMA, "r0_3": r0.name, "r0_3_status_sha256": sha(r0 / "status.json"),
        "expected_pairs": PAIRS, "expected_observations": OBSERVATIONS,
    }
    with TemporaryDirectory(prefix="rlinf-r1-plus-") as staging:
        if args.pairs_from:
            provenance["pair_source"] = import_pairs(args.pairs_from.resolve(), out, provenance["r0_3_status_sha256"])
        write_json(out / "provenance.json", provenance)
        if not args.pairs_from:
            collect(root, out, args.assets, Path(staging))
        dataset = verify_dataset(out)
        status["gates"]["R1-1"] = "PASS"
        status.update(valid_pairs=PAIRS, observations=OBSERVATIONS)
        if args.pairs_only:
            raise AuditBlocked("Pairs-only run finished. Model capture pending.")
        capture(root, out, r0, prior, dataset, args.checkpoint_root.resolve())
    status["gates"]["R1-2"] = "PASS"
    verify_capture(out, dataset, {role: {k: c[k] for k in ("repo_id", "revision")}
                                  for role, c in prior[1].items()}, prior[4])
    status["gates"].update({"R1-3": "PASS", "R1-4": "PASS"})
    status["captured_observations_per_model"] = OBSERVATIONS


def validate_run(root, run):
    status = read_json(run / "status.json")
    if (status["schema"] != SCHEMA or status["stage"] != "R1" or status["status"] != "PASS"
            or status["gates"] != dict.fromkeys(("R1-1", "R1-2", "R1-3", "R1-4"), "PASS")
            or (status["valid_pairs"], status["observations"], status["captured_observations_per_model"])
            != (PAIRS, OBSERVATIONS, OBSERVATIONS)):
        raise ValueError("Incomplete or invalid R1 run")
    sealed_files(run, status)
    evidence = read_json(run / "provenance.json")
    if "pair_source" in evidence and sha(run / "dataset.json") != evidence["pair_source"]["dataset_sha256"]:
        raise ValueError("Copied dataset differs from the parent run")
    path = Path(evidence["r0_3"])
    if (path.name != evidence["r0_3"] or not path.name.startswith("r0-3-")
            or evidence["schema"] != SCHEMA or evidence["expected_pairs"] != PAIRS
            or evidence["expected_observations"] != OBSERVATIONS):
        raise ValueError("Invalid R1 provenance")
    r0 = root / "outputs/research/R0" / path
    if sha(r0 / "status.json") != evidence["r0_3_status_sha256"]:
        raise ValueError("R1 R0 gate identity mismatch")
    _, checkpoints, _, _, references = prior_evidence(root, r0)
    dataset = verify_dataset(run)
    verify_capture(run, dataset, {role: {k: c[k] for k in ("repo_id", "revision")}
                                  for role, c in checkpoints.items()}, references)
    return status
