import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.research.models.parameter_diff import compare, reduce_pair, runtime_parameters
from scripts.research.models.tensor_shards import indexed_tensors


def write_shard(path, entries):
    path.mkdir()
    data = bytearray()
    header = {}
    for name, (dtype, values) in entries.items():
        if dtype == "BF16":
            payload = (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        else:
            payload = values.astype("<f4" if dtype == "F32" else "<f2")
        start = len(data)
        data.extend(payload.tobytes())
        header[name] = {"dtype": dtype, "shape": list(values.shape),
                        "data_offsets": [start, len(data)]}
    shard = "model-00001-of-00001.safetensors"
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for name in entries},
    }))
    return shard


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def tensor_index(self, side, entries):
        path = self.root / side
        shard = write_shard(path, entries)
        return indexed_tensors(path, [shard])

    def test_bf16_cast_and_chunk_reductions(self):
        left = self.tensor_index("left", {"p": ("BF16", np.array([-2, 0, 1, 2]))})
        right = self.tensor_index("right", {"p": ("BF16", np.array([-2, 0, 1.0078125, 3]))})
        row = reduce_pair(left["p"], right["p"], chunk_elements=3).metrics()
        a = np.array([-2, 0, 1, 2])
        b = np.array([-2, 0, 1.0078125, 3])
        self.assertEqual(row["parameter_count"], 4)
        self.assertEqual(row["changed_count"], 2)
        self.assertEqual(row["changed_fraction"], 0.5)
        self.assertEqual(row["max_absolute_delta"], 1)
        self.assertAlmostEqual(row["absolute_l2"], np.linalg.norm(b - a))
        self.assertAlmostEqual(row["relative_l2"], np.linalg.norm(b - a) / np.linalg.norm(a))
        self.assertAlmostEqual(row["cosine_similarity"], (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_reports_safe_aggregates_missing_and_incompatible(self):
        left = self.tensor_index("left", {
            "same": ("F32", np.zeros(3)), "changed": ("F32", np.ones(1)),
            "missing": ("F32", np.zeros(1)), "bad": ("F32", np.zeros(1)),
            "shape": ("F32", np.zeros(1)), "alias.p": ("F32", np.zeros(1)),
        })
        right = self.tensor_index("right", {
            "same": ("F32", np.zeros(3)), "changed": ("F32", np.ones(1) * 2),
            "bad": ("F16", np.zeros(1)), "shape": ("F32", np.zeros(2)),
            "critic": ("F32", np.zeros(1)),
        })
        runtime = {key: ("vision.siglip", shape, "torch.bfloat16")
                   for key, shape in (("same", (3,)), ("changed", (1,)),
                                      ("missing", (1,)), ("bad", (1,)), ("shape", (1,)))}
        rows, modules, complete = compare(runtime, runtime, left, right)
        statuses = {row["checkpoint_key"]: row["status"] for row in rows}
        self.assertEqual(statuses, {
            "same": "identical", "changed": "changed", "missing": "missing_grpo",
            "bad": "unsafe", "shape": "unsafe", "critic": "extra_grpo", "alias.p": "extra_sft",
        })
        self.assertFalse(complete)
        self.assertEqual(modules["vision.siglip"], modules["vision.total"])
        self.assertEqual(modules["policy.total"]["changed_fraction"], 0.25)
        self.assertIsNone(next(r for r in rows if r["checkpoint_key"] == "same")["cosine_similarity"])

    def test_dry_run_only_aligns_without_reductions(self):
        values = {"p": ("F32", np.array(np.nan))}
        left = self.tensor_index("left", values)
        right = self.tensor_index("right", values)
        runtime = {"p": ("action.logits", (), "torch.bfloat16")}
        rows, groups, complete = compare(runtime, runtime, left, right, dry_run=True)
        self.assertTrue(complete)
        self.assertEqual(rows[0]["status"], "aligned")
        self.assertEqual(groups, {})
        rows, _, complete = compare(runtime, runtime, left, right)
        self.assertFalse(complete)
        self.assertEqual(rows[0]["status"], "unsafe")

    def test_rejects_index_and_header_disagreement(self):
        path = self.root / "model"
        shard = write_shard(path, {"actual": ("F32", np.ones(1))})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"other": shard}}))
        with self.assertRaisesRegex(ValueError, "Shard/index disagreement"):
            indexed_tensors(path, [shard])

    def test_registry_precise_group_wins_without_double_counting(self):
        name = "vision_backbone.featurizer.weight"
        parameter = {"path": name, "shape": [2], "numel": 2, "dtype": "torch.bfloat16"}
        groups = {"modules": {
            "vision": {"path": "vision_backbone", "parameters": [parameter], "parameter_count": 2},
            "vision.primary": {
                "path": "vision_backbone.featurizer", "parameters": [parameter], "parameter_count": 2,
            },
        }}
        params = runtime_parameters(groups, ["vit_large_patch14_reg4_dinov2.lvd142m"])
        self.assertEqual(params, {name: ("vision.dinov2", (2,), "torch.bfloat16")})
