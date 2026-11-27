from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.common.integrity import array_hash, sha
from scripts.research.capture_contract import SELECTORS
from scripts.research.core.artifacts import write_json
from scripts.research.data.camera_pairs import save_arrays
from scripts.research.experiments.module_rollback import unit_result, units
from scripts.research.metrics.action import differences
from scripts.research.metrics.intervention import activation_difference


class UnitEvidenceTests(unittest.TestCase):
    def test_plan_has_three_alphas_for_every_safe_leaf(self):
        plan = {"inventory": {"leaves": {
            "a": {"region": "transformer.0", "status": "NO_UPDATE", "parameters": ["a.w"], "elements": 1},
            "b": {"region": "transformer.1", "status": "READY", "parameters": ["b.w"], "elements": 2},
            "c": {"region": "transformer.1", "status": "UNSAFE", "parameters": ["c.w"], "elements": 4},
        }}}
        result = list(units(plan, {"observations": {"obs": {}}}, "full"))
        self.assertEqual(len(result), 6)
        self.assertEqual(len({key for key, _ in result}), 6)
        self.assertEqual({unit["alpha"] for _, unit in result if unit["group"] == "a"}, {0, 0.5, 1})
        self.assertEqual({unit["group"] for _, unit in result}, {"a", "b"})

    def test_validates_control_artifact_and_detects_tampering(self):
        outputs = {"actions": np.zeros((8, 7)), "action_tokens": np.full((8, 7), 31744),
                   "action.logits": np.zeros((56, 256))}
        values = {**dict.fromkeys(SELECTORS, np.zeros((2, 3))), **outputs}
        unit = {"alpha": 1.0, "no_update": False, "elements": 2, "observations": ["obs"]}
        data = {"observations": {"obs": {"data": {"sha256": "rgb"}, "camera_id": "canonical"}},
                "pairs": []}
        records = {"obs": {"pair_ids": [], "hooks": []}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            saved = save_arrays(path / "obs.npz", outputs)
            row = {
                "input_sha256": "rgb", "camera_id": "canonical", "pair_ids": [], "hooks": [],
                "hook_registrations_restored": True, "data": saved, "vocab_start": 31744,
                "tensor_sha256": {key: array_hash(value) for key, value in values.items()},
                "functional": {key: activation_difference(values[key], values[key]) for key in SELECTORS},
                "direct": differences(outputs["actions"], outputs["actions"],
                                      outputs["action_tokens"], outputs["action_tokens"]),
            }
            weights = {"restored": True, "non_selected_unchanged": True, "elements": 2,
                       "effective_changed": 0, "effective_l2": 0.0, "effective_relative_l2": 0.0}
            write_json(path / "measurements.json", {
                "weights": weights, "observations": {"obs": row}, "pairs": [],
                "r0_grpo_regression": "PASS", "r1_grpo_regression": "PASS",
                "sft_reference": "sealed R1 deterministic captures", "elapsed_seconds": 0, "cuda_peak_bytes": 0,
            })
            write_json(path / "status.json", {"unit": unit, "status": "PASS",
                       "evidence_sha256": {p.name: sha(p) for p in path.iterdir()}})
            baseline = lambda obs: (outputs, {key: {"sha256": array_hash(value), "shape": value.shape}
                                              for key, value in values.items()})
            unit_result(path, unit, data, baseline, records)
            with (path / "obs.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                unit_result(path, unit, data, baseline, records)
