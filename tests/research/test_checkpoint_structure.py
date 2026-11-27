import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.core.artifacts import create_run, write_json
from scripts.research.models.checkpoint import AuditBlocked, inspect_checkpoint


class CheckpointStructureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.pin = {"repo_id": "org/model", "revision": "pin"}
        payloads = {
            "config.json": json.dumps({
                "model_type": "openvla", "text_config": {"model_type": "llama"},
            }).encode(),
            "model.safetensors.index.json": b'{"weight_map":{"layer.weight":"weights.safetensors"}}',
            "weights.safetensors": b"mock shard",
        }
        self.files = []
        for name, data in payloads.items():
            (self.root / name).write_bytes(data)
            self.files.append({
                "path": name, "size_bytes": len(data),
                "lfs_sha256": hashlib.sha256(data).hexdigest() if name.endswith(".safetensors") else None,
                "blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            })
        self.inventory = {"models": [dict(self.pin, files=self.files)]}

    def test_verified_checkpoint_structure(self):
        result = inspect_checkpoint(self.root, self.pin, self.inventory)
        self.assertEqual(result["verified_files"], 3)
        self.assertEqual(result["indexed_tensor_count"], 1)
        self.assertEqual(result["published_merge_history"], "unverified")

    def test_rejects_same_size_digest_mismatch(self):
        (self.root / "weights.safetensors").write_bytes(b"fake shard")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            inspect_checkpoint(self.root, self.pin, self.inventory)

    def test_reports_missing_checkpoint(self):
        with self.assertRaisesRegex(AuditBlocked, "directory unavailable"):
            inspect_checkpoint(self.root / "missing", self.pin, self.inventory)

    def test_rejects_unpinned_checkpoint_file(self):
        (self.root / "pytorch_model.bin").touch()
        with self.assertRaisesRegex(ValueError, "Unpinned checkpoint file"):
            inspect_checkpoint(self.root, self.pin, self.inventory)

    def test_preserves_run_evidence(self):
        out = create_run(self.root, "test-run")
        write_json(out / "status.json", {"status": "BLOCKED"})
        with self.assertRaises(FileExistsError):
            create_run(self.root, "test-run")
        with self.assertRaises(FileExistsError):
            write_json(out / "status.json", {"status": "PASS"})
        with self.assertRaises(ValueError):
            create_run(self.root, "../elsewhere")
