"""CPU-only pilot sampling, manifest, horizon, and resume contracts."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

try:
    import numpy as np
except ImportError:  # The metadata-only test suite runs on bare Python too.
    np = None

from scripts.benchmark.plus_inventory import (
    CATEGORIES, CLASSIFICATION, SUITE_COUNTS, TASK_MAP,
    analyze, load_task_map, sha256,
)
from scripts.benchmark.plus_sampling import canonical_hash, key, select
from scripts.benchmark.plus_manifest import (
    CONDITIONS, content_hash, make_manifest, validate_manifests, write_outputs,
)
from scripts.benchmark.plus_evaluation import (
    HORIZONS, LEGACY_PLUS_COLLECTOR_SHA256, episode_id, horizon_for,
    load_selection, prepare_run_directory, resume_protocol,
)
from scripts.benchmark.plus_results import analyze_run, write_results


class PlusSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classification = json.loads(CLASSIFICATION.read_text())
        cls.inventory = analyze(cls.classification, load_task_map(TASK_MAP))
        cls.selected = select(cls.inventory)
        source = {"classification_hash": sha256(CLASSIFICATION),
                  "task_map_hash": sha256(TASK_MAP),
                  "created_at": "2026-11-19T00:00:00+00:00"}
        cls.pilot = make_manifest("pilot", cls.selected, **source)
        cls.heldout = make_manifest("heldout", cls.selected, **source)

    def test_exact_allocation_coverage_and_nonoverlap(self):
        self.assertEqual(len(self.selected["pilot"]), 560)
        self.assertEqual(len(self.selected["heldout"]), 1000)
        self.assertEqual(len(self.selected["allocation"]), 28)
        self.assertTrue(all(a["pilot"] == 20 and not a["fallback_used"]
                            for a in self.selected["allocation"]))
        self.assertFalse({key(r) for r in self.selected["pilot"]} &
                         {key(r) for r in self.selected["heldout"]})
        for suite in SUITE_COUNTS:
            for category in CATEGORIES:
                rows = [r for r in self.selected["pilot"]
                        if r["suite"] == suite and r["perturbation_category"] == category]
                self.assertEqual(len(rows), 20)
                originals = {r["original_task"] for r in rows}
                self.assertEqual(len(originals), 10)
                self.assertTrue(all(sum(r["original_task"] == name for r in rows) == 2
                                    for name in originals))

    def test_sampling_reproducibility_and_probabilities(self):
        self.assertEqual(canonical_hash(self.selected),
                         canonical_hash(select(self.inventory)))
        self.assertNotEqual(canonical_hash(self.selected),
                            canonical_hash(select(self.inventory, seed=43)))
        self.assertTrue(all(r["pilot_inclusion_probability"] is not None
                            and 0 < r["pilot_inclusion_probability"] <= 1
                            for r in self.selected["pilot"] + self.selected["heldout"]))
        self.assertAlmostEqual(sum(r["pilot_inclusion_probability"]
                                   for r in self.selected["pilot"] + self.selected["heldout"]
                                   if r["split"] == "pilot"),
                               sum(r["pilot_inclusion_probability"] for r in self.selected["pilot"]))

    def test_metadata_unknown_and_manifest_hash(self):
        validation = validate_manifests(self.pilot, self.heldout, self.inventory, 560, 1000)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["pilot_unknown_difficulty"], 7)
        self.assertEqual(validation["heldout_unknown_difficulty"], 12)
        later = copy.deepcopy(self.pilot)
        later["created_at"] = "later"
        self.assertEqual(content_hash(later), self.pilot["manifest_content_sha256"])
        later["tasks"][0]["task_id"] = 999
        self.assertNotEqual(content_hash(later), self.pilot["manifest_content_sha256"])

    def test_selection_identity_and_four_condition_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.json"
            path.write_text(json.dumps(self.pilot))
            loaded, indices = load_selection(path, self.classification, CLASSIFICATION, TASK_MAP)
            self.assertEqual(loaded["manifest_content_sha256"], self.pilot["manifest_content_sha256"])
            ids = {episode_id(suite, i) for suite, values in indices.items() for i in values}
            self.assertEqual(len(ids), 560)
            self.assertEqual({condition: ids for condition in CONDITIONS},
                             {condition: ids for condition in CONDITIONS})
            tampered = copy.deepcopy(self.pilot)
            tampered["tasks"][0]["task_name"] = "wrong"
            tampered["manifest_content_sha256"] = content_hash(tampered)
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_selection(path, self.classification, CLASSIFICATION, TASK_MAP)

    def test_horizons_and_resume(self):
        self.assertEqual(HORIZONS["rlinf-520"], {s: 520 for s in SUITE_COUNTS})
        self.assertEqual(HORIZONS["lerobot-suite"],
                         {"libero_spatial": 280, "libero_object": 280,
                          "libero_goal": 300, "libero_10": 520})
        for suite in SUITE_COUNTS:
            self.assertEqual(horizon_for("rlinf-520", suite), 520)
        old = {"horizon": 520, "collector_sha256": LEGACY_PLUS_COLLECTOR_SHA256,
               "condition": "sft_stochastic"}
        new = {**old, "collector_sha256": "new"}
        self.assertEqual(resume_protocol(old, new, legacy=True), old)
        with self.assertRaisesRegex(ValueError, "Resume protocol mismatch"):
            resume_protocol(old, new, legacy=False)
        with self.assertRaisesRegex(ValueError, "Resume protocol mismatch"):
            resume_protocol(old, {**new, "horizon": 280}, legacy=True)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "new-run" / "sft_stochastic"
            with self.assertRaises(FileNotFoundError):
                prepare_run_directory(run_dir, resume=True)
            prepare_run_directory(run_dir, resume=False)
            with self.assertRaises(FileExistsError):
                prepare_run_directory(run_dir, resume=False)
            prepare_run_directory(run_dir, resume=True)

    @unittest.skipIf(np is None, "NumPy unavailable; action-loop mock needs NumPy")
    def test_action_step_horizon_and_immediate_success(self):
        from scripts.evaluate import nominal_collect

        class FakeEnv:
            def __init__(self, horizon, success_at=None):
                self.cfg = types.SimpleNamespace(max_episode_steps=horizon)
                self.cumsum_trial_id_bins = np.array([1])
                self.task_ids = np.array([0])
                self.trial_ids = np.array([0])
                self.task_descriptions = ["example"]
                self.success_once = [False]
                self.current_raw_obs = [self.raw()]
                self.steps = 0
                self.success_at = success_at

            @staticmethod
            def raw():
                return {"robot0_eef_pos": np.zeros(3),
                        "robot0_eef_quat": np.array([1, 0, 0, 0]),
                        "robot0_gripper_qpos": np.zeros(2),
                        "agentview_image": np.zeros((1, 1, 3), dtype=np.uint8)}

            def reset(self, **_kwargs):
                self.steps = 0
                self.current_raw_obs = [self.raw()]
                return {"main_images": np.zeros((1, 1)),
                        "states": np.zeros((1, 8))}, {}

            def step(self, _action, auto_reset=False):
                self.steps += 1
                success = self.steps == self.success_at
                self.success_once[0] = success
                return ({"main_images": np.zeros((1, 1)), "states": np.zeros((1, 8))},
                        None, np.array([success]),
                        np.array([self.steps >= self.cfg.max_episode_steps]), {})

        class FakeModel:
            vision_backbone = types.SimpleNamespace(get_num_images_in_input=lambda: 1)

            def predict_action_batch(self, **_kwargs):
                return np.zeros((1, 8, 7)), {"forward_inputs": {
                    "action_tokens": np.zeros((1, 8, 7)),
                    "pixel_values": np.zeros((1, 1)),
                    "input_ids": np.zeros((1, 1)),
                    "attention_mask": np.zeros((1, 1))}}

        fake_torch = types.SimpleNamespace(
            manual_seed=lambda _seed: None,
            cuda=types.SimpleNamespace(manual_seed_all=lambda _seed: None),
            inference_mode=nullcontext,
        )
        fake_action_utils = types.SimpleNamespace(
            prepare_actions_for_libero=lambda x, _model: x)
        fake_envs = types.ModuleType("rlinf.envs")
        fake_sim = types.ModuleType("rlinf.envs.sim")
        fake_rlinf = types.ModuleType("rlinf")
        fake_libero = types.ModuleType("rlinf.envs.sim.libero")
        with patch.dict(sys.modules, {
            "torch": fake_torch, "rlinf": fake_rlinf,
            "rlinf.envs": fake_envs, "rlinf.envs.sim": fake_sim,
            "rlinf.envs.sim.libero": fake_libero,
            "rlinf.envs.action_utils": fake_action_utils,
        }), patch.object(nominal_collect, "write_episode", side_effect=lambda _run, metadata, *_: metadata):
            for horizon in (280, 300, 520):
                env = FakeEnv(horizon)
                metadata = {}
                nominal_collect.collect_episode(
                    env, FakeModel(), suite="libero_spatial", task_id=0,
                    trial_id=0, sampling=False, temperature=1.0,
                    policy_seed=0, metadata=metadata, run_dir=Path("unused"))
                self.assertEqual(env.steps, horizon)
                self.assertEqual(metadata["episode_length"], horizon)
                self.assertFalse(metadata["success"])
                self.assertEqual(metadata["termination_reason"], "horizon")
            env = FakeEnv(280, success_at=17)
            metadata = {}
            nominal_collect.collect_episode(
                env, FakeModel(), suite="libero_spatial", task_id=0,
                trial_id=0, sampling=False, temperature=1.0,
                policy_seed=0, metadata=metadata, run_dir=Path("unused"))
            self.assertEqual(env.steps, 17)
            self.assertTrue(metadata["success"])
            self.assertEqual(metadata["termination_reason"], "success")

    def test_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir, report_dir = Path(directory) / "manifests", Path(directory) / "reports"
            from scripts.benchmark.plus_manifest import reports
            validation = validate_manifests(self.pilot, self.heldout, self.inventory, 560, 1000)
            allocation, coverage, summary = reports(
                self.inventory, self.selected, self.pilot, self.heldout, validation)
            write_outputs(manifest_dir, report_dir, self.pilot, self.heldout,
                          allocation, coverage, summary, validation)
            self.assertEqual(json.loads((manifest_dir / "pilot_v1.json").read_text()), self.pilot)
            with self.assertRaises(FileExistsError):
                write_outputs(manifest_dir, report_dir, self.pilot, self.heldout,
                              allocation, coverage, summary, validation)

    def test_pilot_result_identity_weighting_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = sorted(episode_id(r["suite"], r["task_id"] - 1)
                              for r in self.pilot["tasks"])
            for condition in CONDITIONS:
                protocol = {
                    "condition": condition,
                    "task_manifest_content_sha256": self.pilot["manifest_content_sha256"],
                    "horizon_protocol": "lerobot-suite",
                    "selected_episode_ids": expected,
                    "protocol_version": "plus_pilot_suite_v1",
                    "suite_horizons": HORIZONS["lerobot-suite"],
                    "wait_steps": 15, "action_chunks": 8,
                    "do_sample": condition.endswith("stochastic"),
                    "temperature": 1.6 if condition.endswith("stochastic") else 1.0,
                    "lora_applied": False,
                    "checkpoint_revision": condition.split("_")[0] + "-revision",
                    "sampling_mode": condition.split("_")[1],
                }
                protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
                condition_dir = root / condition
                condition_dir.mkdir()
                (condition_dir / "manifest.json").write_text(json.dumps({
                    "protocol": protocol, "protocol_hash": protocol_hash}))
                for task in self.pilot["tasks"]:
                    eid = episode_id(task["suite"], task["task_id"] - 1)
                    path = condition_dir / "rollouts" / eid
                    path.mkdir(parents=True)
                    metadata = {
                        "complete": True, "episode_id": eid,
                        "protocol_hash": protocol_hash,
                        "suite": task["suite"], "task_id": task["task_id"] - 1,
                        "plus_classification_id": task["task_id"],
                        "task_name": task["task_name"],
                        "shift_category": task["perturbation_category"],
                        "difficulty": task["difficulty_level"],
                        "horizon": HORIZONS["lerobot-suite"][task["suite"]],
                        "protocol_version": "plus_pilot_suite_v1",
                        "task_manifest_content_sha256": self.pilot["manifest_content_sha256"],
                        "checkpoint_revision": protocol["checkpoint_revision"],
                        "sampling_mode": protocol["sampling_mode"],
                        "lora_applied": protocol["lora_applied"],
                        "policy_seed": 100000 * list(SUITE_COUNTS).index(task["suite"]) + task["task_id"] - 1,
                        "temperature": protocol["temperature"],
                        "wait_steps": 15, "action_chunks": 8,
                        "success": condition.startswith("grpo"),
                        "episode_length": 10,
                        "termination_reason": "success" if condition.startswith("grpo") else "horizon",
                        "initial_state_sha256": "a" * 64,
                        "storage_mode": "metadata",
                    }
                    (path / "metadata.json").write_text(json.dumps(metadata))
            result, episodes, paired = analyze_run(root, self.pilot, repetitions=10)
            self.assertEqual(result["paired_episodes"], 560)
            self.assertEqual(result["benchmark_weighted_grpo_minus_sft"],
                             {"deterministic": 1.0, "stochastic": 1.0})
            self.assertEqual(result["estimates"]["sft_deterministic"]["overall"]["represented_population"], 10030)
            self.assertEqual(result["pair_outcomes"]["deterministic"]["sft_failure_grpo_success"], 560)
            self.assertTrue(all(r["action_trajectory_path"] is None for r in episodes))
            write_results(root, result, episodes, paired)
            with self.assertRaises(FileExistsError):
                write_results(root, result, episodes, paired)


if __name__ == "__main__":
    unittest.main()
