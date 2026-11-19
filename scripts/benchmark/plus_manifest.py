"""Create and validate immutable LIBERO-PLUS pilot/held-out manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scripts.benchmark.plus_inventory import (
    CLASSIFICATION, TASK_MAP, OUTPUT as INVENTORY_OUTPUT,
    ROOT, SUITE_COUNTS, CATEGORIES, analyze, load_task_map, sha256,
)
from scripts.benchmark.plus_sampling import canonical_hash, distribution, key, select
from scripts.common.io import csv_save, save


BENCHMARK_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
MANIFEST_DIR = ROOT / "manifests/libero_plus"
REPORT_DIR = ROOT / "outputs/benchmark/libero_plus/sampling"
CONDITIONS = ("sft_deterministic", "sft_stochastic",
              "grpo_deterministic", "grpo_stochastic")


def content_hash(manifest: dict) -> str:
    return canonical_hash({k: v for k, v in manifest.items()
                           if k not in ("created_at", "manifest_content_sha256")})


def make_manifest(split: str, selected: dict, *, classification_hash: str,
                  task_map_hash: str, created_at: str) -> dict:
    tasks = selected[split]
    manifest = {
        "manifest_version": 1,
        "benchmark_revision": BENCHMARK_REVISION,
        "classification_sha256": classification_hash,
        "task_map_sha256": task_map_hash,
        "sampling_algorithm": selected["sampling_algorithm"],
        "sampling_seed": selected["sampling_seed"],
        "created_at": created_at,
        "split": split,
        "total_selected": len(tasks),
        "stratum_allocation": [
            {"suite": a["suite"], "category": a["category"],
             "population": a["population"], "selected": a[split],
             "sampling_fraction": (a[split] / a["population"])}
            for a in selected["allocation"]],
        "policy_conditions": list(CONDITIONS),
        "tasks": tasks,
    }
    manifest["manifest_content_sha256"] = content_hash(manifest)
    return manifest


def validate_manifests(pilot: dict, heldout: dict, inventory: dict,
                       expected_pilot: int, expected_heldout: int) -> dict:
    reference = {key(r): r for r in inventory["tasks"]}
    issues = []
    for manifest, split, expected in ((pilot, "pilot", expected_pilot),
                                      (heldout, "heldout", expected_heldout)):
        rows = manifest["tasks"]
        if manifest["total_selected"] != expected or len(rows) != expected:
            issues.append(f"{split} count mismatch")
        if content_hash(manifest) != manifest["manifest_content_sha256"]:
            issues.append(f"{split} content hash mismatch")
        if len({key(r) for r in rows}) != len(rows):
            issues.append(f"{split} duplicate variant IDs")
        if len(manifest["stratum_allocation"]) != 28:
            issues.append(f"{split} missing strata")
        observed = Counter((r["suite"], r["perturbation_category"]) for r in rows)
        for a in manifest["stratum_allocation"]:
            if observed[a["suite"], a["category"]] != a["selected"]:
                issues.append(f"{split} allocation mismatch: {a['suite']}/{a['category']}")
        for r in rows:
            source = reference.get(key(r))
            if source is None or (
                r["task_name"], r["original_task"],
                r["perturbation_category"], r["difficulty_level"]
            ) != (
                source["name"], source["original_task"],
                source["category"], source["difficulty_level"]
            ):
                issues.append(f"{split} metadata mismatch: {key(r)}")
            if r["split"] != split or r["sampling_seed"] != manifest["sampling_seed"]:
                issues.append(f"{split} provenance mismatch: {key(r)}")
    overlap = {key(r) for r in pilot["tasks"]} & {key(r) for r in heldout["tasks"]}
    if overlap:
        issues.append(f"pilot/heldout overlap: {len(overlap)}")
    if pilot["classification_sha256"] != heldout["classification_sha256"] or \
       pilot["task_map_sha256"] != heldout["task_map_sha256"]:
        issues.append("source hashes differ")
    coverage = []
    for suite in SUITE_COUNTS:
        for category in CATEGORIES:
            originals = {r["original_task"] for r in inventory["tasks"]
                         if r["suite"] == suite}
            selected_originals = {r["original_task"] for r in pilot["tasks"]
                                  if r["suite"] == suite and r["perturbation_category"] == category}
            coverage.append({"suite": suite, "category": category,
                             "population_original_tasks": len(originals),
                             "pilot_original_tasks": len(selected_originals),
                             "complete": originals == selected_originals})
    result = {"valid": not issues, "issues": issues,
              "pilot_count": len(pilot["tasks"]), "heldout_count": len(heldout["tasks"]),
              "overlap_count": len(overlap), "strata_count": len(coverage),
              "pilot_original_task_coverage_complete": all(r["complete"] for r in coverage),
              "pilot_unknown_difficulty": sum(r["difficulty_level"] is None for r in pilot["tasks"]),
              "heldout_unknown_difficulty": sum(r["difficulty_level"] is None for r in heldout["tasks"])}
    if issues:
        raise ValueError("; ".join(issues[:10]))
    return result


def reports(inventory: dict, selected: dict, pilot: dict, heldout: dict,
            validation: dict) -> tuple[list[dict], list[dict], str]:
    allocation = selected["allocation"]
    coverage = []
    for suite in SUITE_COUNTS:
        for category in CATEGORIES:
            population = [r for r in inventory["tasks"] if r["suite"] == suite and r["category"] == category]
            p = [r for r in pilot["tasks"] if r["suite"] == suite and r["perturbation_category"] == category]
            h = [r for r in heldout["tasks"] if r["suite"] == suite and r["perturbation_category"] == category]
            for original in sorted({r["original_task"] for r in population}):
                coverage.append({"suite": suite, "category": category, "original_task": original,
                                 "population": sum(r["original_task"] == original for r in population),
                                 "pilot": sum(r["original_task"] == original for r in p),
                                 "heldout": sum(r["original_task"] == original for r in h)})
    dists = {"benchmark": distribution(inventory["tasks"]),
             "pilot": distribution(pilot["tasks"], manifest=True),
             "heldout": distribution(heldout["tasks"], manifest=True)}
    lines = ["# LIBERO-PLUS stratified sampling", "",
             f"Pilot: **{len(pilot['tasks'])}** variants; held-out: **{len(heldout['tasks'])}** variants; overlap: **{validation['overlap_count']}**.",
             "", "The split is OOD-variant-disjoint, not original-task-disjoint.",
             "The 560/1,000 defaults are pilot design choices, not statistically optimized sample sizes.",
             "", "## Validation", "",
             f"- All 28 strata covered: {validation['strata_count'] == 28}",
             f"- Original-task coverage in pilot: {validation['pilot_original_task_coverage_complete']}",
             f"- Unknown difficulty: pilot {validation['pilot_unknown_difficulty']}, held-out {validation['heldout_unknown_difficulty']}",
             "- Pilot is sampled without replacement within each original-task cell. Its exact inclusion probability is the cell quota divided by cell population.",
             "- Held-out is sampled without replacement from remaining variants. Its conditional and unconditional inclusion probabilities are recorded per task. If fallback allocation is used, exact probabilities are null rather than estimated.",
             "", "## Marginal distributions", ""]
    lines += ["| Dimension | Value | Benchmark | Pilot | Pilot fraction | Held-out | Held-out fraction |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dimension, counts in dists["benchmark"].items():
        for value, population in counts.items():
            p = dists["pilot"][dimension].get(value, 0)
            h = dists["heldout"][dimension].get(value, 0)
            lines.append(f"| {dimension} | {value} | {population} | {p} | {p / population:.6f} | {h} | {h / population:.6f} |")
    lines.append("")
    lines += ["## Files", "",
              "- `allocation_report.csv`: population, selected counts, fractions, and any quota deficits per stratum.",
              "- `coverage_report.csv`: original-task coverage within every suite × category.",
              "- `validation.json`: checked manifest counts, hashes, identities, and split overlap.", ""]
    return allocation, coverage, "\n".join(lines)


def write_outputs(manifest_dir: Path, report_dir: Path, pilot: dict,
                  heldout: dict, allocation: list[dict], coverage: list[dict],
                  summary: str, validation: dict) -> None:
    paths = [manifest_dir / "pilot_v1.json", manifest_dir / "heldout_v1.json",
             report_dir / "allocation_report.csv", report_dir / "coverage_report.csv",
             report_dir / "sampling_summary.md", report_dir / "validation.json"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Sampling output exists; refusing to overwrite")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    save(paths[0], pilot)
    save(paths[1], heldout)
    csv_save(paths[2], allocation)
    csv_save(paths[3], coverage)
    paths[4].write_text(summary, encoding="utf-8")
    save(paths[5], validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-stratum", type=int, default=20)
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory", type=Path,
                        default=INVENTORY_OUTPUT / "inventory.json")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    official_root = CLASSIFICATION.parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != BENCHMARK_REVISION:
        raise ValueError(f"Official benchmark revision mismatch: {revision}")
    classification = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    source_hashes = {"classification_hash": sha256(CLASSIFICATION),
                     "task_map_hash": sha256(TASK_MAP)}
    if (inventory.get("provenance", {}).get("classification_sha256") != source_hashes["classification_hash"] or
        inventory.get("provenance", {}).get("task_map_sha256") != source_hashes["task_map_hash"] or
        inventory.get("tasks") != analyze(classification, load_task_map(TASK_MAP))["tasks"]):
        raise ValueError("Inventory differs from pinned official metadata")
    selected = select(inventory, per_stratum=args.samples_per_stratum,
                      heldout_size=args.heldout_size, seed=args.seed)
    now = datetime.now(timezone.utc).isoformat()
    pilot = make_manifest("pilot", selected, created_at=now, **source_hashes)
    heldout = make_manifest("heldout", selected, created_at=now, **source_hashes)
    validation = validate_manifests(
        pilot, heldout, inventory, args.samples_per_stratum * 28, args.heldout_size)
    regenerated = select(inventory, per_stratum=args.samples_per_stratum,
                         heldout_size=args.heldout_size, seed=args.seed)
    validation["deterministic_regeneration"] = (
        pilot["manifest_content_sha256"] == make_manifest(
            "pilot", regenerated, created_at="excluded-from-hash", **source_hashes
        )["manifest_content_sha256"] and
        heldout["manifest_content_sha256"] == make_manifest(
            "heldout", regenerated, created_at="excluded-from-hash", **source_hashes
        )["manifest_content_sha256"])
    if not validation["deterministic_regeneration"]:
        raise ValueError("Sampling regeneration changed manifest content")
    allocation, coverage, summary = reports(inventory, selected, pilot, heldout, validation)
    write_outputs(args.manifest_dir, args.report_dir, pilot, heldout,
                  allocation, coverage, summary, validation)
    print(summary)
    print(f"Manifests: {args.manifest_dir}\nReports: {args.report_dir}")


if __name__ == "__main__":
    main()
