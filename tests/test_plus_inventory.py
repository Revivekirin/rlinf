"""Pinned, CPU-only tests for the LIBERO-PLUS metadata inventory."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark.plus_inventory import (
    CATEGORIES, CLASSIFICATION, SUITE_COUNTS, TASK_MAP,
    analyze, load_task_map, sha256, write_inventory,
)


class PlusInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classification = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
        cls.task_map = load_task_map(TASK_MAP)
        cls.inventory = analyze(cls.classification, cls.task_map)

    def test_official_counts_and_mapping(self):
        self.assertEqual(self.inventory["total_tasks"], 10030)
        self.assertEqual(self.inventory["suite_counts"], SUITE_COUNTS)
        self.assertEqual(self.inventory["inferred_original_task_counts"],
                         {suite: 10 for suite in SUITE_COUNTS})
        for suite in SUITE_COUNTS:
            tasks = [r for r in self.inventory["tasks"] if r["suite"] == suite]
            self.assertEqual(len(tasks), SUITE_COUNTS[suite])
            self.assertEqual([r["task_id"] for r in tasks],
                             list(range(1, SUITE_COUNTS[suite] + 1)))
            self.assertEqual([r["name"] for r in tasks], self.task_map[suite])

    def test_missing_duplicate_and_misaligned_ids(self):
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"].pop()
        with self.assertRaisesRegex(ValueError, "Task count mismatch"):
            analyze(altered, self.task_map)
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"][1]["id"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate task ID"):
            analyze(altered, self.task_map)
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"][0]["id"] = 2403
        with self.assertRaisesRegex(ValueError, "Invalid task ID"):
            analyze(altered, self.task_map)
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"][0]["name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "registry task"):
            analyze(altered, self.task_map)

    def test_category_and_difficulty_consistency(self):
        tasks = self.inventory["tasks"]
        self.assertEqual({r["category"] for r in tasks}, set(CATEGORIES))
        self.assertEqual(sum(r["count"] for r in self.inventory["suite_category"]), 10030)
        self.assertEqual(sum(r["count"] for r in self.inventory["suite_original_category"]), 10030)
        self.assertEqual(sum(r["count"] for r in self.inventory["category_difficulty"]), 10030)
        self.assertEqual(self.inventory["missing_difficulty_count"], 121)
        self.assertEqual(self.inventory["missing_difficulty"],
                         [{"suite": "libero_goal", "category": "Light Conditions", "count": 121}])
        self.assertEqual(sum(r["difficulty_level"] is None for r in tasks), 121)
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"][0]["category"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Invalid category"):
            analyze(altered, self.task_map)
        altered = copy.deepcopy(self.classification)
        altered["libero_spatial"][0]["difficulty_level"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid difficulty"):
            analyze(altered, self.task_map)

    def test_json_csv_are_deterministic_and_exclusive(self):
        data = copy.deepcopy(self.inventory)
        data["provenance"] = {
            "classification_sha256": sha256(CLASSIFICATION),
            "task_map_sha256": sha256(TASK_MAP),
            "analysis_code_sha256": sha256(Path(__import__("scripts.benchmark.plus_inventory", fromlist=["x"]).__file__)),
        }
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            write_inventory(first, data)
            write_inventory(second, data)
            for name in ("inventory.json", "tasks.csv", "suite_category.csv",
                         "suite_original_category.csv", "category_difficulty.csv", "SUMMARY.md"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with (first / "tasks.csv").open(newline="") as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 10030)
            self.assertEqual(json.loads((first / "inventory.json").read_text()), data)
            with self.assertRaises(FileExistsError):
                write_inventory(first, data)


if __name__ == "__main__":
    unittest.main()
