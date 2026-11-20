"""CPU-only, non-destructive analysis of the completed LIBERO-PLUS pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS, analyze, load_task_map, original_stem
from scripts.benchmark.plus_manifest import CONDITIONS, content_hash
from scripts.benchmark.plus_results import percentile, validated_records
from scripts.common.io import csv_save, save

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = frozenset(("libero_spatial-t2396-i00", "libero_goal-t2410-i00"))
MODES = ("deterministic", "stochastic")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_delta(rows: list[dict], mode: str, *, repetitions: int = 500,
                    seed: int = 42) -> tuple[float, float]:
    """Resample whole original-task clusters, preserving each outcome pair."""
    return bootstrap_rates(rows, mode, repetitions=repetitions, seed=seed)["delta"]


def bootstrap_rates(rows: list[dict], mode: str, *, repetitions: int = 500,
                    seed: int = 42) -> dict[str, tuple[float, float]]:
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[row["suite"]][row["original_task"]].append(row)
    rng = random.Random(seed)
    draws = {"sft": [], "grpo": [], "delta": []}
    for _ in range(repetitions):
        sample = []
        for suite in sorted(cells):
            names = sorted(cells[suite])
            for _ in names:
                sample.extend(cells[suite][rng.choice(names)])
        s = sum(r[f"sft_{mode}"] for r in sample) / len(sample)
        g = sum(r[f"grpo_{mode}"] for r in sample) / len(sample)
        for key, value in (("sft", s), ("grpo", g), ("delta", g - s)):
            draws[key].append(value)
    return {key: (percentile(values, .025), percentile(values, .975))
            for key, values in draws.items()}


def outcomes(rows: list[dict], mode: str, *, interval: bool = False) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("Empty analysis cell")
    s = sum(r[f"sft_{mode}"] for r in rows)
    g = sum(r[f"grpo_{mode}"] for r in rows)
    both = sum(r[f"sft_{mode}"] and r[f"grpo_{mode}"] for r in rows)
    gained = sum(not r[f"sft_{mode}"] and r[f"grpo_{mode}"] for r in rows)
    lost = sum(r[f"sft_{mode}"] and not r[f"grpo_{mode}"] for r in rows)
    fail = n - both - gained - lost
    assert both + gained + lost + fail == n and g - s == gained - lost
    result = {"N": n, "sft_successes": s, "sft_sr": s / n,
              "grpo_successes": g, "grpo_sr": g / n,
              "delta_sr_pp": 100 * (g - s) / n, "both_success": both,
              "gained": gained, "lost": lost, "both_fail": fail}
    if interval:
        bounds = bootstrap_rates(rows, mode)
        for key in ("sft", "grpo"):
            result[f"{key}_ci_lower"] = bounds[key][0]
            result[f"{key}_ci_upper"] = bounds[key][1]
        lo, hi = bounds["delta"]
        result.update(ci_lower_pp=100 * lo, ci_upper_pp=100 * hi)
    return result


def summarize(paired: list[dict], inventory: dict) -> dict:
    originals = {s: sorted({r["original_task"] for r in inventory["tasks"]
                            if r["suite"] == s}) for s in SUITE_COUNTS}
    original_ids = {(s, name): f"{s}:{i + 1:02d}"
                    for s, names in originals.items() for i, name in enumerate(names)}
    if any(len(names) != 10 for names in originals.values()):
        raise ValueError("Expected ten original tasks per suite")
    strata = []
    original = []
    difficulty = []
    for mode in MODES:
        for suite in SUITE_COUNTS:
            for category in CATEGORIES:
                cell = [r for r in paired if r["suite"] == suite and
                        r["perturbation_category"] == category]
                strata.append({"suite": suite, "shift_category": category,
                               "sampling_mode": mode, **outcomes(cell, mode, interval=True)})
            for name in originals[suite]:
                subset = [r for r in paired if r["suite"] == suite and r["original_task"] == name]
                for category in ("ALL", *CATEGORIES):
                    cell = subset if category == "ALL" else [r for r in subset if r["perturbation_category"] == category]
                    original.append({"original_task_id": original_ids[suite, name],
                                     "original_task_name": name, "suite": suite,
                                     "shift_category": category, "sampling_mode": mode,
                                     **outcomes(cell, mode)})
        for category in (*range(1, 6), None):
            for shift in ("ALL", *CATEGORIES):
                cell = [r for r in paired if r["difficulty_level"] == category and
                        (shift == "ALL" or r["perturbation_category"] == shift)]
                if cell:
                    difficulty.append({"difficulty_level": category if category is not None else "unknown",
                                       "shift_category": shift, "sampling_mode": mode,
                                       **outcomes(cell, mode)})
    concentration = {}
    for mode in MODES:
        task_rows = [r for r in original if r["sampling_mode"] == mode and r["shift_category"] == "ALL"]
        net = sum(r["grpo_successes"] - r["sft_successes"] for r in task_rows)
        total_gained = sum(r["gained"] for r in task_rows)
        ranked = sorted(task_rows, key=lambda r: (-(r["grpo_successes"] - r["sft_successes"]), r["original_task_id"]))
        gains_ranked = sorted(task_rows, key=lambda r: (-r["gained"], r["original_task_id"]))
        top5_net = sum(r["grpo_successes"] - r["sft_successes"] for r in ranked[:5])
        top5_gained = sum(r["gained"] for r in gains_ranked[:5])
        level = [r for r in strata if r["sampling_mode"] == mode]
        concentration[mode] = {
            "positive_strata": sum(r["delta_sr_pp"] > 0 for r in level),
            "negative_strata": sum(r["delta_sr_pp"] < 0 for r in level),
            "unchanged_strata": sum(r["delta_sr_pp"] == 0 for r in level),
            "positive_original_tasks": sum(r["delta_sr_pp"] > 0 for r in task_rows),
            "negative_original_tasks": sum(r["delta_sr_pp"] < 0 for r in task_rows),
            "unchanged_original_tasks": sum(r["delta_sr_pp"] == 0 for r in task_rows),
            "net_success_gain": net, "total_gained": total_gained,
            "top5_net_task_ids": [r["original_task_id"] for r in ranked[:5]],
            "top5_net_contribution": top5_net,
            "top5_gained_task_ids": [r["original_task_id"] for r in gains_ranked[:5]],
            "top5_positive_gains": top5_gained,
            "top5_share_of_positive_gains": top5_gained / total_gained if total_gained else None,
            "top5_share_of_net_gain": top5_net / net if net >= 10 else None,
            "net_share_minimum_gain": 10,
            "suite_contribution": {s: sum(r[f"grpo_{mode}"] - r[f"sft_{mode}"]
                                           for r in paired if r["suite"] == s) for s in SUITE_COUNTS},
            "category_contribution": {c: sum(r[f"grpo_{mode}"] - r[f"sft_{mode}"]
                                              for r in paired if r["perturbation_category"] == c)
                                      for c in CATEGORIES},
            "original_task_macro_delta_pp": sum(r["delta_sr_pp"] for r in task_rows) / len(task_rows),
            "episode_weighted_delta_pp": outcomes(paired, mode)["delta_sr_pp"],
            "overall": outcomes(paired, mode, interval=True),
        }
    return {"strata": strata, "original": original, "difficulty": difficulty,
            "concentration": concentration, "original_ids": original_ids}


def population_audit(pilot: dict, paired: list[dict], inventory: dict) -> dict:
    population = Counter((r["suite"], r["category"], r["original_task"])
                         for r in inventory["tasks"])
    valid_probabilities = all(r["pilot_inclusion_probability"] ==
                              2 / population[r["suite"], r["perturbation_category"], r["original_task"]]
                              for r in pilot["tasks"])
    counts = Counter((r["suite"], r["perturbation_category"], r["original_task"])
                     for r in paired)
    return {"population": inventory["total_tasks"],
            "suite_category_population": inventory["suite_category"],
            "original_task_cells": len(population),
            "pilot_inclusion_probabilities_valid": valid_probabilities,
            "complete_case_cell_counts": dict(sorted(("|".join(k), v) for k, v in counts.items())),
            "unbalanced_cells": sorted("|".join(k) for k, v in counts.items() if v != 2),
            "weighted_estimate": "NOT IDENTIFIABLE",
            "reason": "Two exclusions were selected by observed reset mismatch. Their complete-case inclusion probabilities are unknown; original pilot probabilities cannot be reused as if missing at random."}


def nominal_audit(root: Path, protocols: dict) -> dict:
    nominal = root / "outputs/nominal"
    paths = sorted(nominal.glob("*/manifest.json")) if nominal.exists() else []
    entries = []
    for path in paths:
        data = json.loads(path.read_text())
        p = data.get("protocol", {})
        condition = p.get("condition")
        plus = protocols.get(condition, {})
        rollouts = path.parent / "rollouts"
        covered = sorted({item.name.split("-t", 1)[0] for item in rollouts.iterdir()
                          if "-t" in item.name}) if rollouts.is_dir() else []
        entries.append({"path": str(path), "sha256": digest(path),
                        "suite_coverage": covered, "condition": condition,
                        "checkpoint_revision": p.get("checkpoint_revision"),
                        "checkpoint_matches_plus": p.get("checkpoint_revision") == plus.get("checkpoint_revision"),
                        "unnorm_key": p.get("unnorm_key"),
                        "normalization_matches_plus": p.get("unnorm_key") == plus.get("unnorm_key"),
                        "sampling_mode": p.get("sampling_mode"),
                        "sampling_matches_plus": (p.get("do_sample") == plus.get("do_sample") and
                                                  p.get("temperature") == plus.get("temperature")),
                        "horizon": p.get("horizon"), "wait_steps": p.get("num_steps_wait"),
                        "trials_per_task": p.get("trials_per_task"),
                        "mapping_sha256": p.get("mapping_sha256"),
                        "initial_state_protocol": "nominal saved state applied by RLinf reset",
                        "plus_initial_state_protocol": plus.get("initial_state_protocol"),
                        "directly_compatible": False})
    return {"status": "AUDIT ONLY", "nominal_manifests_found": len(entries),
            "entries": entries, "plus_protocols": protocols,
            "task_level_comparison": "NOT AVAILABLE: nominal and PLUS evaluation distributions, horizons, trials, and reset protocols differ",
            "robustness_gap": "NOT CALCULATED"}


def load_inputs(root: Path, pilot_path: Path, run_dir: Path) -> tuple[dict, dict, list[dict], list[dict], dict]:
    pilot = json.loads(pilot_path.read_text())
    inventory_path = root / "outputs/benchmark/libero_plus/inventory/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    if content_hash(pilot) != pilot["manifest_content_sha256"] or inventory["total_tasks"] != 10030:
        raise ValueError("Pilot or inventory provenance mismatch")
    if (pilot["classification_sha256"] != inventory["provenance"]["classification_sha256"] or
        pilot["task_map_sha256"] != inventory["provenance"]["task_map_sha256"]):
        raise ValueError("Official inventory source mismatch")
    benchmark = root / "upstream/LIBERO-plus/libero/libero/benchmark"
    classification_path = benchmark / "task_classification.json"
    task_map_path = benchmark / "libero_suite_task_map.py"
    if classification_path.exists() and task_map_path.exists():
        if (digest(classification_path) != pilot["classification_sha256"] or
            digest(task_map_path) != pilot["task_map_sha256"] or
            analyze(json.loads(classification_path.read_text()), load_task_map(task_map_path))["tasks"] != inventory["tasks"]):
            raise ValueError("Official task mapping differs from inventory")
    indexed = {(r["suite"], r["task_id"]): r for r in inventory["tasks"]}
    for row in pilot["tasks"]:
        source = indexed[row["suite"], row["task_id"]]
        if (source["original_task"] != row["original_task"] or
            original_stem(source["name"], source["category"]) != row["original_task"]):
            raise ValueError("Original task mapping mismatch")
    episodes, paired = validated_records(run_dir, pilot, exclude_episode_ids=EXCLUDED)
    if len(episodes) != 2232 or len(paired) != 558:
        raise ValueError("Expected 2,232 records and 558 paired episodes")
    expected_success = {"sft_deterministic": 70, "sft_stochastic": 123,
                        "grpo_deterministic": 167, "grpo_stochastic": 202}
    if {c: sum(r[c] for r in paired) for c in CONDITIONS} != expected_success:
        raise ValueError("Pilot success-count regression mismatch")
    protocols = {c: json.loads((run_dir / c / "manifest.json").read_text())["protocol"] for c in CONDITIONS}
    return pilot, inventory, episodes, paired, protocols


def write_analysis(root: Path, pilot_path: Path, run_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    pilot, inventory, episodes, paired, protocols = load_inputs(root, pilot_path, run_dir)
    data = summarize(paired, inventory)
    source_paths = [pilot_path, root / "outputs/benchmark/libero_plus/inventory/inventory.json"]
    benchmark = root / "upstream/LIBERO-plus/libero/libero/benchmark"
    source_paths.extend(p for p in (benchmark / "task_classification.json",
                                    benchmark / "libero_suite_task_map.py") if p.exists())
    for condition in CONDITIONS:
        source_paths.append(run_dir / condition / "manifest.json")
        source_paths.extend(sorted((run_dir / condition / "rollouts").glob("*/metadata.json")))
    sources = {str(p): digest(p) for p in source_paths}
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    names = data["original_ids"]
    table = []
    for row in episodes:
        table.append({"episode_id": row["episode_id"], "suite": row["suite"],
                      "original_task_id": names[row["suite"], row["original_task"]],
                      "original_task_name": row["original_task"],
                      "shift_category": row["perturbation_category"],
                      "difficulty_level": row["difficulty_level"], "condition": row["condition"],
                      "model": row["condition"].split("_")[0].upper(),
                      "sampling_mode": row["sampling_mode"], "temperature": row["temperature"],
                      "success": row["success"], "episode_length": row["episode_length"],
                      "start_observation_sha256": row["initial_state_sha256"],
                      "protocol_hash": json.loads((run_dir / row["condition"] / "manifest.json").read_text())["protocol_hash"],
                      "source_result_path": row["metadata_path"],
                      "input_set_sha256": source_digest})
    paired_table = [{**r, "input_set_sha256": source_digest} for r in paired]
    output.mkdir(parents=True)
    save(output / "validation.json", {"status": "PASS", "conditions": list(CONDITIONS),
                                      "excluded_episode_ids": sorted(EXCLUDED),
                                      "exclusion_reason": "Observed-start hash differs across four conditions",
                                      "source_records_validated": 2240, "included_records": 2232,
                                      "paired_episodes": 558,
                                      "physical_state_equivalence": "UNVERIFIED",
                                      "official_stored_init_state_applied": False,
                                      "input_set_sha256": source_digest})
    for filename, rows in (("episode_table.csv", table), ("strata_summary.csv", data["strata"]),
                           ("original_task_summary.csv", data["original"]),
                           ("paired_outcomes.csv", paired_table),
                           ("difficulty_summary.csv", data["difficulty"])):
        csv_save(output / filename, [{**r, "input_set_sha256": source_digest} for r in rows])
    save(output / "concentration_summary.json", {"input_set_sha256": source_digest,
                                                  "sampling_seed": 42, **data["concentration"]})
    save(output / "population_weighting_audit.json", {"input_set_sha256": source_digest,
                                                       **population_audit(pilot, paired, inventory)})
    save(output / "nominal_compatibility.json", {"input_set_sha256": source_digest,
                                                  **nominal_audit(root, protocols)})
    mapping = [{"original_task_id": short, "suite": s, "original_task_name": name}
               for (s, name), short in sorted(names.items())]
    csv_save(output / "original_task_mapping.csv", [{**r, "input_set_sha256": source_digest} for r in mapping])
    lines = ["# LIBERO-PLUS pilot generalization", "",
             f"Input set SHA256: `{source_digest}`.", "",
             "558 matched task / verified initial observation episodes; two documented exclusions.",
             "Physical simulator-state equivalence is unverified. Stored official init states were not applied.",
             "Population-weighted estimate: NOT IDENTIFIABLE after outcome-dependent exclusions.",
             "Severity analysis: NOT AVAILABLE (no explicit severity metadata).", ""]
    for mode in MODES:
        item = data["concentration"][mode]
        x = item["overall"]
        lines += [f"## {mode}", "",
                  f"SFT {x['sft_successes']}/{x['N']} ({100*x['sft_sr']:.2f}%); GRPO {x['grpo_successes']}/{x['N']} ({100*x['grpo_sr']:.2f}%).",
                  f"Paired Δ {x['delta_sr_pp']:+.2f} pp; original-task clustered 95% interval [{x['ci_lower_pp']:+.2f}, {x['ci_upper_pp']:+.2f}] pp.",
                  f"Strata: {item['positive_strata']} positive, {item['negative_strata']} negative, {item['unchanged_strata']} unchanged.",
                  f"Original tasks: {item['positive_original_tasks']} positive, {item['negative_original_tasks']} negative, {item['unchanged_original_tasks']} unchanged.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    output_files = sorted(p for p in output.rglob("*") if p.is_file())
    save(output / "manifest.json", {"source_files_sha256": sources,
                                    "input_set_sha256": source_digest,
                                    "output_files_sha256": {str(p.relative_to(output)): digest(p) for p in output_files},
                                    "analysis_code_sha256": digest(Path(__file__)),
                                    "plus_results_code_sha256": digest(Path(__file__).with_name("plus_results.py")),
                                    "run_id": run_dir.name, "excluded_episode_ids": sorted(EXCLUDED)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--pilot", type=Path, default=ROOT / "manifests/libero_plus/pilot_v1.json")
        p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/libero_plus/plus-pilot-suite-nolora-v2")
        p.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/plus_generalization/pilot_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "validate":
        _, _, episodes, paired, _ = load_inputs(root, args.pilot.resolve(), args.run_dir.resolve())
        print(json.dumps({"status": "PASS", "records": len(episodes), "paired": len(paired),
                          "excluded": sorted(EXCLUDED)}, indent=2))
    else:
        write_analysis(root, args.pilot.resolve(), args.run_dir.resolve(), args.output.resolve())
        print(f"Analysis: {args.output.resolve()}")


if __name__ == "__main__":
    main()
