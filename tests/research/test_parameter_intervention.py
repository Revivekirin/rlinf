from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.common.integrity import sha
from scripts.research.interventions.parameter import cast_interpolation, edited, groups_from_audit, live_parameters
from scripts.research.metrics.intervention import pair_change
from scripts.research.models.tensor_shards import Tensor


class ParameterTests(unittest.TestCase):
    def test_exact_partition_and_unsafe_shape(self):
        runtime = {f"layer.{name}": ("transformer.0", (2,), "torch.float32")
                   for name in ("weight", "bias")}
        rows = [{
            "canonical_name": name, "checkpoint_key": name, "status": "identical",
            "sft_shape": [2], "grpo_shape": [2], "sft_storage_dtype": "F32",
            "grpo_storage_dtype": "F32", "parameter_count": 2, "changed_count": 0,
        } for name in runtime]
        groups = groups_from_audit({"tensors": rows}, runtime)
        self.assertEqual(groups["parents"], {"transformer": ["layer"]})
        self.assertEqual(groups["leaves"]["layer"]["status"], "NO_UPDATE")
        self.assertEqual(groups["runtime_elements"], 4)
        rows[1]["grpo_shape"] = [3]
        self.assertEqual(groups_from_audit({"tensors": rows}, runtime)["leaves"]["layer"]["status"], "UNSAFE")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            groups_from_audit({"tensors": [*rows, rows[0]]}, runtime)

    def test_tied_parameters_require_audited_alias_correspondence(self):
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.ones(2))
        model.b = model.a
        with self.assertRaisesRegex(ValueError, "shared storage"):
            live_parameters(model, {})

    def test_bf16_midpoint_can_round_to_endpoint(self):
        sft, grpo = torch.tensor([1.0078125]), torch.tensor([1.0])
        self.assertEqual(cast_interpolation(sft, grpo, 0.0, torch.bfloat16).item(), 1.0078125)
        self.assertEqual(cast_interpolation(sft, grpo, 0.5, torch.bfloat16).item(), 1.0)
        self.assertEqual(cast_interpolation(sft, grpo, 1.0, torch.bfloat16).item(), 1.0)

    def test_streamed_apply_restore_and_shards_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            left, right = p / "sft", p / "grpo"
            left.write_bytes(np.array([1, 2], dtype="<f4").tobytes())
            right.write_bytes(np.array([3, 4], dtype="<f4").tobytes())
            digests = [sha(left), sha(right)]
            a, b = ({"weight": Tensor(file, 0, (2,), "F32")} for file in (left, right))
            live = {key: torch.nn.Parameter(torch.tensor(value)) for key, value in
                    {"weight": [3., 4.], "other": [5.]}.items()}
            for alpha, expected in ((0, [1., 2.]), (0.5, [2., 3.]), (1, [3., 4.])):
                with edited(live, ["weight"], a, b, alpha, chunk=1) as stats:
                    self.assertEqual(live["weight"].tolist(), expected)
                    self.assertEqual(live["other"].item(), 5)
                self.assertTrue(stats["restored"] and stats["non_selected_unchanged"])
                self.assertEqual(live["weight"].tolist(), [3, 4])
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with edited(live, ["weight"], a, b, 0, chunk=1):
                    raise RuntimeError("inference failed")
            self.assertEqual(live["weight"].tolist(), [3, 4])
            self.assertEqual([sha(left), sha(right)], digests)

    def test_paired_difference_is_difference_in_differences(self):
        a = np.zeros((8, 7))
        b = np.ones((8, 7))
        result = pair_change(a, b, a + 2, b + 3)
        self.assertEqual(result["camera_response_delta"], np.ones((8, 7)).tolist())
        self.assertEqual(result["gripper_abs"], [1] * 8)
        self.assertEqual(pair_change(a, b, a + 2, b + 2)["gripper_abs"], [0] * 8)
