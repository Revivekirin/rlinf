"""CPU-only paired and design-weighted analysis of completed pilot rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from scripts.benchmark.plus_evaluation import HORIZONS, episode_id
from scripts.benchmark.plus_manifest import CONDITIONS, content_hash
from scripts.common.io import save


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def cluster_intervals(paired: list[dict], *, repetitions: int = 500,
                      seed: int = 42) -> dict:
    """Exploratory original-task cluster bootstrap within each stratum."""
    cells = defaultdict(lambda: defaultdict(list))
    for row in paired:
        cells[row["suite"], row["perturbation_category"]][row["original_task"]].append(row)
    rng = random.Random(seed)
    keys = [*CONDITIONS, "grpo_minus_sft_deterministic", "grpo_minus_sft_stochastic"]
    draws = {k: [] for k in keys}
    for _ in range(repetitions):
        totals = dict.fromkeys(keys, 0.0)
        population = 0.0
        for originals in cells.values():
            names = sorted(originals)
            for _ in names:
                group = originals[rng.choice(names)]
                size = group[0]["original_population"]
                population += size
                for condition in CONDITIONS:
                    totals[condition] += size * sum(r[condition] for r in group) / len(group)
                for mode in ("deterministic", "stochastic"):
                    label = f"grpo_minus_sft_{mode}"
                    totals[label] += size * sum(r[f"grpo_{mode}"] - r[f"sft_{mode}"]
                                                for r in group) / len(group)
        for k in keys:
            draws[k].append(totals[k] / population)
    return {k: {"lower_95": percentile(v, .025), "upper_95": percentile(v, .975)}
            for k, v in draws.items()}


def analyze_run(run_dir: Path, pilot: dict, *, repetitions: int = 500) -> tuple[dict, list[dict], list[dict]]:
    if content_hash(pilot) != pilot.get("manifest_content_sha256"):
        raise ValueError("Pilot manifest hash mismatch")
    expected = {episode_id(r["suite"], r["task_id"] - 1): r for r in pilot["tasks"]}
    if len(expected) != len(pilot["tasks"]):
        raise ValueError("Duplicate pilot variant")
    episodes = []
    paired = []
    by_condition = {}
    for condition in CONDITIONS:
        condition_dir = run_dir / condition
        run_manifest = json.loads((condition_dir / "manifest.json").read_text())
        protocol = run_manifest["protocol"]
        actual_protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if (protocol.get("condition") != condition or
            run_manifest.get("protocol_hash") != actual_protocol_hash or
            protocol.get("task_manifest_content_sha256") != pilot["manifest_content_sha256"] or
            protocol.get("horizon_protocol") != "lerobot-suite" or
            protocol.get("selected_episode_ids") != sorted(expected) or
            protocol.get("protocol_version") != "plus_pilot_suite_v1" or
            protocol.get("suite_horizons") != HORIZONS["lerobot-suite"] or
            protocol.get("wait_steps") != 15 or
            protocol.get("action_chunks") != 8 or
            protocol.get("do_sample") != condition.endswith("stochastic") or
            protocol.get("temperature") != (1.6 if condition.endswith("stochastic") else 1.0) or
            protocol.get("lora_applied") is not False):
            raise ValueError(f"Pilot protocol mismatch: {condition}")
        found = set()
        for path in sorted((condition_dir / "rollouts").glob("*")):
            if path.name.startswith(".partial-"):
                continue
            if path.name not in expected:
                raise ValueError(f"Unexpected pilot episode: {path}")
            metadata_path = path / "metadata.json"
            row = json.loads(metadata_path.read_text())
            task = expected[path.name]
            if (row.get("episode_id") != path.name or
                row.get("complete") is not True or
                row.get("protocol_hash") != run_manifest["protocol_hash"] or
                row.get("suite") != task["suite"] or
                row.get("task_id") != task["task_id"] - 1 or
                row.get("plus_classification_id") != task["task_id"] or
                row.get("task_name") != task["task_name"] or
                row.get("shift_category") != task["perturbation_category"] or
                row.get("difficulty") != task["difficulty_level"] or
                row.get("horizon") != HORIZONS["lerobot-suite"][task["suite"]] or
                row.get("protocol_version") != protocol["protocol_version"] or
                row.get("task_manifest_content_sha256") != pilot["manifest_content_sha256"] or
                row.get("checkpoint_revision") != protocol["checkpoint_revision"] or
                row.get("sampling_mode") != protocol["sampling_mode"] or
                row.get("lora_applied") != protocol["lora_applied"] or
                row.get("policy_seed") != 100000 * list(HORIZONS["lerobot-suite"]).index(task["suite"]) + task["task_id"] - 1 or
                row.get("temperature") != protocol["temperature"] or
                row.get("wait_steps") != 15 or
                row.get("action_chunks") != 8 or
                not isinstance(row.get("initial_state_sha256"), str) or
                len(row["initial_state_sha256"]) != 64 or
                not isinstance(row.get("success"), bool) or
                not 1 <= row.get("episode_length", 0) <= row["horizon"] or
                row.get("termination_reason") != ("success" if row["success"] else "horizon")):
                raise ValueError(f"Episode metadata mismatch: {metadata_path}")
            found.add(path.name)
            episodes.append({"condition": condition, "episode_id": path.name,
                             "suite": task["suite"], "task_id": task["task_id"],
                             "task_name": task["task_name"],
                             "original_task": task["original_task"],
                             "perturbation_category": task["perturbation_category"],
                             "difficulty_level": task["difficulty_level"],
                             "success": row["success"],
                             "episode_length": row["episode_length"],
                             "sampling_mode": row["sampling_mode"],
                             "temperature": row["temperature"],
                             "policy_seed": row["policy_seed"],
                             "initial_state_sha256": row["initial_state_sha256"],
                             "metadata_path": str(metadata_path),
                             "action_trajectory_path": str(path) if row.get("storage_mode") == "raw" else None,
                             "pilot_inclusion_probability": task["pilot_inclusion_probability"]})
        if found != set(expected):
            raise ValueError(f"Incomplete pilot condition: {condition}: {len(found)}/{len(expected)}")
        by_condition[condition] = {r["episode_id"]: r for r in episodes if r["condition"] == condition}
    original_population = Counter((r["suite"], r["perturbation_category"], r["original_task"])
                                  for r in pilot["tasks"])
    for eid, task in sorted(expected.items()):
        rows = [by_condition[c][eid] for c in CONDITIONS]
        identity = {(r["suite"], r["task_id"], r["task_name"],
                     r["policy_seed"], r["initial_state_sha256"]) for r in rows}
        if len(identity) != 1:
            raise ValueError(f"Paired episode identity/observed reset differs: {eid}")
        p = task["pilot_inclusion_probability"]
        if p is None or p <= 0:
            raise ValueError("Exact pilot inclusion probability required for weighting")
        paired.append({"episode_id": eid, "suite": task["suite"],
                       "task_id": task["task_id"],
                       "original_task": task["original_task"],
                       "perturbation_category": task["perturbation_category"],
                       "difficulty_level": task["difficulty_level"],
                       "pilot_inclusion_probability": p,
                       "original_population": round(2 / p),
                       **{c: int(by_condition[c][eid]["success"]) for c in CONDITIONS}})
    if any(original_population[k] != 2 for k in original_population):
        raise ValueError("Bootstrap requires two pilot variants per original-task cell")

    def estimate(rows: list[dict], condition: str) -> dict:
        n = sum(1 / r["pilot_inclusion_probability"] for r in rows)
        weighted = sum(r[condition] / r["pilot_inclusion_probability"] for r in rows) / n
        return {"completed": len(rows), "successes": sum(r[condition] for r in rows),
                "sample_success_rate": sum(r[condition] for r in rows) / len(rows),
                "benchmark_weighted_success_rate": weighted,
                "represented_population": round(n)}

    estimates = {}
    for condition in CONDITIONS:
        estimates[condition] = {
            "overall": estimate(paired, condition),
            "suite": {s: estimate([r for r in paired if r["suite"] == s], condition)
                      for s in HORIZONS["lerobot-suite"]},
            "category": {c: estimate([r for r in paired if r["perturbation_category"] == c], condition)
                         for c in sorted({r["perturbation_category"] for r in paired})},
            "suite_category": [
                {"suite": s, "category": c,
                 **estimate([r for r in paired if r["suite"] == s and r["perturbation_category"] == c], condition)}
                for s in HORIZONS["lerobot-suite"]
                for c in sorted({r["perturbation_category"] for r in paired})],
        }
    pair_outcomes = {}
    differences = {}
    for mode in ("deterministic", "stochastic"):
        sft, grpo = f"sft_{mode}", f"grpo_{mode}"
        pair_outcomes[mode] = {
            "sft_success_grpo_failure": sum(r[sft] == 1 and r[grpo] == 0 for r in paired),
            "sft_failure_grpo_success": sum(r[sft] == 0 and r[grpo] == 1 for r in paired),
            "both_success": sum(r[sft] == 1 and r[grpo] == 1 for r in paired),
            "both_failure": sum(r[sft] == 0 and r[grpo] == 0 for r in paired),
        }
        differences[mode] = (sum((r[grpo] - r[sft]) / r["pilot_inclusion_probability"]
                                 for r in paired) / 10030)
    result = {
        "pilot_manifest_content_sha256": pilot["manifest_content_sha256"],
        "paired_episodes": len(paired), "estimates": estimates,
        "pair_outcomes": pair_outcomes,
        "benchmark_weighted_grpo_minus_sft": differences,
        "exploratory_cluster_bootstrap_95_ci": cluster_intervals(paired, repetitions=repetitions),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": 42,
        "stochastic_repeats_per_variant": 1,
        "uncertainty_note": "Original-task cluster bootstrap within suite-category strata; exploratory with 10 original tasks per suite. Single stochastic policy seed per variant.",
        "state_note": "Initial-state hash covers selected robot observations and camera pixels, not full simulator state.",
        "trajectory_note": "Metadata-only rollouts do not contain an action trajectory; path is null for those episodes.",
    }
    return result, episodes, paired


def write_results(run_dir: Path, result: dict, episodes: list[dict], paired: list[dict]) -> None:
    paths = [run_dir / "pilot_validation.json", run_dir / "pilot_episodes.csv", run_dir / "pilot_paired.csv"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Pilot analysis output already exists")
    save(paths[0], result)
    for path, rows in ((paths[1], episodes), (paths[2], paired)):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
