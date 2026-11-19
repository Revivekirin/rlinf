"""Validate LIBERO-PLUS runs and write paired/category success summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plus_collect import CONDITIONS, SUITES, atomic_json, eid, read_episode, summarize


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--task-manifest", type=Path,
                        help="Analyze a completed manifest-based pilot without changing legacy output")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.task_manifest:
        if args.allow_incomplete:
            parser.error("Pilot statistics require all four conditions to be complete")
        from scripts.benchmark.plus_results import analyze_run, write_results
        pilot = json.loads(args.task_manifest.read_text(encoding="utf-8"))
        result, episodes, paired = analyze_run(run_dir, pilot)
        write_results(run_dir, result, episodes, paired)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    all_rows = []
    per_condition = {}
    for condition in CONDITIONS:
        condition_dir = run_dir / condition
        manifest = json.loads((condition_dir / "manifest.json").read_text())
        protocol = manifest["protocol"]
        if protocol["condition"] != condition:
            raise ValueError(f"Condition mismatch: {condition}")
        counts = protocol["task_counts"]
        if set(counts) - set(SUITES):
            raise ValueError(f"Unexpected suite: {set(counts) - set(SUITES)}")
        # Smoke manifests retain official suite sizes in task_counts, while
        # selecting only the first spatial task. Keep their protocol hashes
        # unchanged so already collected episodes remain resumable.
        if protocol["storage_mode"] == "raw" and protocol["selected_task_count"] == 1:
            if set(counts) != {"libero_spatial"}:
                raise ValueError(f"Unexpected smoke suites: {sorted(counts)}")
            expected = {eid("libero_spatial", 0)}
        else:
            expected = {eid(suite, task_id) for suite, count in counts.items()
                        for task_id in range(count)}
        if len(expected) != protocol["selected_task_count"]:
            raise ValueError(f"Task count mismatch: {condition}")
        summary = summarize(condition_dir, expected, manifest["protocol_hash"])
        per_condition[condition] = summary
        if summary["missing"] and not args.allow_incomplete:
            raise ValueError(f"Incomplete {condition}: {summary['missing']} missing")
        for path in sorted((condition_dir / "rollouts").glob("*")):
            if path.name.startswith(".partial-"):
                continue
            row = read_episode(path, manifest["protocol_hash"])
            if row["sampling_mode"] != protocol["sampling_mode"]:
                raise ValueError(f"Sampling mode mismatch: {path}")
            if row["checkpoint_revision"] != protocol["checkpoint_revision"]:
                raise ValueError(f"Checkpoint mismatch: {path}")
            all_rows.append({"condition": condition, **row})

    by_episode = defaultdict(dict)
    categories = defaultdict(lambda: {"completed": 0, "successes": 0})
    for row in all_rows:
        episode = row["episode_id"]
        by_episode[episode][row["condition"]] = row
        item = categories[(row["condition"], row["shift_category"])]
        item["completed"] += 1
        item["successes"] += int(row["success"])

    paired = []
    for episode, condition_rows in sorted(by_episode.items()):
        if len(condition_rows) != len(CONDITIONS):
            continue
        identity = {(row["suite"], row["task_id"], row["task_name"],
                     row["shift_category"], row["difficulty"], row["policy_seed"],
                     row["initial_state_sha256"])
                    for row in condition_rows.values()}
        if len(identity) != 1:
            raise ValueError(f"Task or initial observation differs across conditions: {episode}")
        paired.append({
            "episode_id": episode,
            "sft_stochastic": int(condition_rows["sft_stochastic"]["success"]),
            "grpo_stochastic": int(condition_rows["grpo_stochastic"]["success"]),
            "sft_deterministic": int(condition_rows["sft_deterministic"]["success"]),
            "grpo_deterministic": int(condition_rows["grpo_deterministic"]["success"]),
        })

    category_rows = [
        {"condition": condition, "shift_category": category,
         "completed": value["completed"], "successes": value["successes"],
         "success_rate": value["successes"] / value["completed"]}
        for (condition, category), value in sorted(categories.items())
    ]
    pair_rates = {
        "sft_vs_grpo_stochastic": sum(r["grpo_stochastic"] - r["sft_stochastic"] for r in paired) / len(paired) if paired else None,
        "sft_vs_grpo_deterministic": sum(r["grpo_deterministic"] - r["sft_deterministic"] for r in paired) / len(paired) if paired else None,
        "sft_deterministic_vs_stochastic": sum(r["sft_deterministic"] - r["sft_stochastic"] for r in paired) / len(paired) if paired else None,
        "grpo_deterministic_vs_stochastic": sum(r["grpo_deterministic"] - r["grpo_stochastic"] for r in paired) / len(paired) if paired else None,
    }
    result = {
        "conditions": per_condition,
        "paired_episodes": len(paired),
        "paired_success_rate_differences": pair_rates,
        "by_shift_category": category_rows,
    }
    atomic_json(run_dir / "validation.json", result)
    atomic_csv(run_dir / "episodes.csv", all_rows,
               ["condition", "episode_id", "suite", "task_id", "task_name",
                "shift_category", "difficulty", "initial_state_sha256",
                "policy_seed", "success", "episode_length", "sampling_mode",
                "temperature", "checkpoint_revision", "protocol_hash"])
    atomic_csv(run_dir / "paired.csv", paired,
               ["episode_id", *CONDITIONS])
    atomic_csv(run_dir / "shift_category.csv", category_rows,
               ["condition", "shift_category", "completed", "successes", "success_rate"])
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.allow_incomplete and len(paired) != per_condition["sft_stochastic"]["expected"]:
        raise ValueError("Four-way paired task coverage incomplete")


if __name__ == "__main__":
    main()
