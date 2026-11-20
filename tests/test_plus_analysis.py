"""CPU-only contracts for the 558-episode pilot analysis."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark.plus_analysis import (
    EXCLUDED, bootstrap_delta, load_inputs, outcomes, population_audit,
    summarize, write_analysis,
)
from scripts.benchmark.plus_evaluation import HORIZONS, episode_id
from scripts.benchmark.plus_figures import verify_sources
from scripts.benchmark.plus_inventory import SUITE_COUNTS
from scripts.benchmark.plus_manifest import CONDITIONS
from scripts.benchmark.plus_results import validated_records

ROOT = Path(__file__).resolve().parents[1]


class PlusAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_dir = cls.root / "run"
        cls.pilot_path = ROOT / "manifests/libero_plus/pilot_v1.json"
        cls.pilot = json.loads(cls.pilot_path.read_text())
        cls.inventory = json.loads((ROOT / "outputs/benchmark/libero_plus/inventory/inventory.json").read_text())
        (cls.root / "outputs/benchmark/libero_plus/inventory").mkdir(parents=True)
        (cls.root / "outputs/benchmark/libero_plus/inventory/inventory.json").write_text(json.dumps(cls.inventory))
        included = [r for r in cls.pilot["tasks"] if episode_id(r["suite"], r["task_id"] - 1) not in EXCLUDED]
        success_counts = {"sft_deterministic": 70, "sft_stochastic": 123,
                          "grpo_deterministic": 167, "grpo_stochastic": 202}
        expected = sorted(episode_id(r["suite"], r["task_id"] - 1) for r in cls.pilot["tasks"])
        for condition in CONDITIONS:
            protocol = {
                "condition": condition, "task_manifest_content_sha256": cls.pilot["manifest_content_sha256"],
                "horizon_protocol": "lerobot-suite", "selected_episode_ids": expected,
                "protocol_version": "plus_pilot_suite_v1", "suite_horizons": HORIZONS["lerobot-suite"],
                "wait_steps": 15, "action_chunks": 8, "do_sample": condition.endswith("stochastic"),
                "temperature": 1.6 if condition.endswith("stochastic") else 1.0,
                "lora_applied": False, "checkpoint_revision": condition.split("_")[0] + "-revision",
                "sampling_mode": condition.split("_")[1],
            }
            phash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            directory = cls.run_dir / condition
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_hash": phash}))
            successes = {episode_id(r["suite"], r["task_id"] - 1)
                         for r in included[:success_counts[condition]]}
            if condition == "grpo_stochastic":
                successes.add("libero_spatial-t2396-i00")
            for task in cls.pilot["tasks"]:
                eid = episode_id(task["suite"], task["task_id"] - 1)
                path = directory / "rollouts" / eid
                path.mkdir(parents=True)
                success = eid in successes
                start = "a" * 64
                if eid in EXCLUDED:
                    start = ("b" if condition == "sft_deterministic" else "c") * 64
                metadata = {
                    "complete": True, "episode_id": eid, "protocol_hash": phash,
                    "suite": task["suite"], "task_id": task["task_id"] - 1,
                    "plus_classification_id": task["task_id"], "task_name": task["task_name"],
                    "shift_category": task["perturbation_category"], "difficulty": task["difficulty_level"],
                    "horizon": HORIZONS["lerobot-suite"][task["suite"]],
                    "protocol_version": "plus_pilot_suite_v1",
                    "task_manifest_content_sha256": cls.pilot["manifest_content_sha256"],
                    "checkpoint_revision": protocol["checkpoint_revision"],
                    "sampling_mode": protocol["sampling_mode"], "lora_applied": False,
                    "policy_seed": 100000 * list(SUITE_COUNTS).index(task["suite"]) + task["task_id"] - 1,
                    "temperature": protocol["temperature"], "wait_steps": 15, "action_chunks": 8,
                    "success": success, "episode_length": 10 if success else protocol["suite_horizons"][task["suite"]],
                    "termination_reason": "success" if success else "horizon",
                    "initial_state_sha256": start, "storage_mode": "metadata",
                }
                (path / "metadata.json").write_text(json.dumps(metadata))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_alignment_exclusion_and_success_counts(self):
        episodes, paired = validated_records(self.run_dir, self.pilot, exclude_episode_ids=EXCLUDED)
        self.assertEqual((len(episodes), len(paired)), (2232, 558))
        self.assertEqual({c: sum(r[c] for r in paired) for c in CONDITIONS},
                         {"sft_deterministic": 70, "sft_stochastic": 123,
                          "grpo_deterministic": 167, "grpo_stochastic": 202})
        self.assertFalse({r["episode_id"] for r in paired} & EXCLUDED)
        with self.assertRaisesRegex(ValueError, "observed reset differs"):
            validated_records(self.run_dir, self.pilot)

    def test_strata_mapping_difficulty_and_weighting(self):
        _, paired = validated_records(self.run_dir, self.pilot, exclude_episode_ids=EXCLUDED)
        data = summarize(paired, self.inventory)
        self.assertEqual(len(data["strata"]), 56)
        self.assertEqual(len({(r["suite"], r["original_task_name"])
                              for r in data["original"]}), 40)
        self.assertEqual(sum(r["difficulty_level"] is None for r in paired), 6)
        self.assertEqual(sorted(r["N"] for r in data["strata"] if r["sampling_mode"] == "deterministic"),
                         [19, 19] + [20] * 26)
        audit = population_audit(self.pilot, paired, self.inventory)
        self.assertTrue(audit["pilot_inclusion_probabilities_valid"])
        self.assertEqual(audit["weighted_estimate"], "NOT IDENTIFIABLE")
        self.assertEqual(len(audit["unbalanced_cells"]), 2)
        for mode in ("deterministic", "stochastic"):
            concentration = data["concentration"][mode]
            self.assertEqual(sum(concentration[k] for k in
                             ("positive_strata", "negative_strata", "unchanged_strata")), 28)
            self.assertEqual(sum(concentration[k] for k in
                             ("positive_original_tasks", "negative_original_tasks", "unchanged_original_tasks")), 40)
            self.assertEqual(sum(concentration["suite_contribution"].values()),
                             concentration["net_success_gain"])
            self.assertLessEqual(concentration["top5_positive_gains"], concentration["total_gained"])

    def test_paired_identities_and_cluster_bootstrap(self):
        _, paired = validated_records(self.run_dir, self.pilot, exclude_episode_ids=EXCLUDED)
        for mode in ("deterministic", "stochastic"):
            result = outcomes(paired, mode)
            self.assertEqual(result["N"], sum(result[k] for k in
                             ("both_success", "gained", "lost", "both_fail")))
            self.assertEqual(result["grpo_successes"] - result["sft_successes"],
                             result["gained"] - result["lost"])
            self.assertEqual(bootstrap_delta(paired, mode, repetitions=20),
                             bootstrap_delta(paired, mode, repetitions=20))

    def test_missing_episode_and_protocol_tampering_fail(self):
        task = self.pilot["tasks"][0]
        eid = episode_id(task["suite"], task["task_id"] - 1)
        path = self.run_dir / "sft_deterministic" / "rollouts" / eid / "metadata.json"
        original = path.read_text()
        try:
            path.unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                validated_records(self.run_dir, self.pilot, exclude_episode_ids=EXCLUDED)
            path.write_text(original)
            row = json.loads(original)
            row["checkpoint_revision"] = "wrong"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Episode metadata mismatch"):
                validated_records(self.run_dir, self.pilot, exclude_episode_ids=EXCLUDED)
        finally:
            path.write_text(original)

    def test_write_is_exclusive_and_figure_sources_match(self):
        out = self.root / "analysis"
        first = self.pilot["tasks"][0]
        source = (self.run_dir / "sft_deterministic" / "rollouts" /
                  episode_id(first["suite"], first["task_id"] - 1) / "metadata.json")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        write_analysis(self.root, self.pilot_path, self.run_dir, out)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual(verify_sources(out)[1], json.loads((out / "validation.json").read_text())["input_set_sha256"])
        with (out / "episode_table.csv").open(newline="") as f:
            self.assertEqual(sum(1 for _ in csv.DictReader(f)), 2232)
        with self.assertRaises(FileExistsError):
            write_analysis(self.root, self.pilot_path, self.run_dir, out)


if __name__ == "__main__":
    unittest.main()
