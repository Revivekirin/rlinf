# LIBERO-PLUS metadata inventory protocol

This module inventories the pinned LIBERO-PLUS classification and task registry. It does not select an evaluation subset, load a policy, start a simulator, or read rollout results.

## Inputs and identity

- Classification: `upstream/LIBERO-plus/libero/libero/benchmark/task_classification.json`.
- Ordered registry: `upstream/LIBERO-plus/libero/libero/benchmark/libero_suite_task_map.py`.
- Four perturbed suites: `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
- The classification `id` is **one-based** and must match the name at registry index `id - 1`. The inventory records both identifiers. IDs are unique within a suite; `(suite, task_id)` is the global key.

The analyzer parses the registry's literal `libero_task_map` assignment with `ast.literal_eval`; it does not import LIBERO, MuJoCo, Torch, or RLinf environment code. It checks exact suite counts (2,402 / 2,518 / 2,591 / 2,519), all 10,030 IDs, registry alignment, duplicate IDs/names, known categories, and difficulty values.

## Original-task mapping

The classification has only `id`, `name`, `category`, and `difficulty_level`; it does not provide an `original_task` field. The official registry likewise lists variant names, without explicit parent IDs. The inventory therefore strips the perturbation suffix from each official name using category-specific patterns (`_table_`, `_tb_`, `_view_..._initstate_`, `_language_`, `_noise_`, `_add_`, `_moved_level..._sample...`, `_level..._sample...`, `_light_`). Every classification name is checked against the same position in the official registry, and the resulting stems yield exactly 10 original tasks per suite. This is a **name-derived grouping**, not a supplied parent-ID annotation.

## Metadata limits found in the pinned source

- The seven categories are explicit. The claimed 21 perturbation subtypes have **no explicit per-task field or complete subtype mapping** in the checked classification, registry, README, or available randomizer code. Some filename suffixes expose parameters or families, but these are not enough to assign all 21 official subtype labels without interpretation. The inventory leaves subtype unassigned.
- `difficulty_level` is `null` for **121** tasks, all `libero_goal` / `Light Conditions`. They remain in all task and category totals and appear under `null` in the category × difficulty table. The analyzer never fills these values.
- `difficulty_level` values outside integers 1–5, unknown categories, absent IDs, and duplicate IDs cause validation errors.

## Run and outputs

From the repository root:

```bash
python3 -m unittest tests.test_plus_inventory
python3 -m scripts.benchmark.plus_inventory
```

The second command creates `outputs/benchmark/libero_plus/inventory/` with `inventory.json`, `tasks.csv`, `suite_category.csv`, `suite_original_category.csv`, `category_difficulty.csv`, and `SUMMARY.md`. Existing report files are never overwritten. For another pinned source or output location, use `--classification`, `--task-map`, and `--output`.

The JSON and Markdown record SHA256 hashes of the classification, official task map, and analysis code. CSV and JSON rows have fixed order and contain no timestamps. An absolute input path is recorded for provenance, so byte-for-byte comparison across machines requires the same paths or comparison of the report data after excluding path fields.
