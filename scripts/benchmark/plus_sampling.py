"""Deterministic, metadata-only LIBERO-PLUS pilot and held-out sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS


ALGORITHM = "original_balanced_v1+proportional_heldout_v1"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def key(row: dict) -> tuple[str, int]:
    return row["suite"], row["task_id"]


def strata(inventory: dict) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    for row in inventory["tasks"]:
        groups[row["suite"], row["category"]].append(row)
    return {(suite, category): sorted(groups[suite, category], key=lambda r: r["task_id"])
            for suite in SUITE_COUNTS for category in CATEGORIES}


def largest_remainder(sizes: dict, total: int) -> dict:
    if total < 0 or total > sum(sizes.values()):
        raise ValueError("Requested held-out size exceeds available variants")
    population = sum(sizes.values())
    raw = {k: total * n / population for k, n in sizes.items()}
    allocation = {k: int(raw[k]) for k in sizes}
    remaining = total - sum(allocation.values())
    for k in sorted(sizes, key=lambda x: (-(raw[x] - allocation[x]), x))[:remaining]:
        allocation[k] += 1
    return allocation


def select(inventory: dict, *, per_stratum: int = 20,
           heldout_size: int = 1000, seed: int = 42) -> dict:
    if per_stratum < 1 or heldout_size < 0:
        raise ValueError("Sampling sizes must be nonnegative and pilot nonempty")
    groups = strata(inventory)
    rng = random.Random(seed)
    pilot = []
    pilot_prob = {}
    allocation = []
    suite_originals = {suite: {r["original_task"] for r in inventory["tasks"]
                               if r["suite"] == suite} for suite in SUITE_COUNTS}
    for (suite, category), candidates in groups.items():
        original_groups = defaultdict(list)
        for row in candidates:
            original_groups[row["original_task"]].append(row)
        original_names = sorted(original_groups)
        missing_originals = sorted(suite_originals[suite] - set(original_names))
        if not candidates or per_stratum > len(candidates):
            raise ValueError(f"Insufficient stratum: {suite}/{category}")
        target_each = per_stratum // len(original_names)
        extras = per_stratum % len(original_names)
        selected = []
        deficits = []
        # Reserve an equal quota per original task. Sorted names and IDs keep
        # all RNG draws reproducible across Python processes.
        for i, original in enumerate(original_names):
            pool = original_groups[original]
            quota = target_each + int(i < extras)
            chosen = rng.sample(pool, min(quota, len(pool)))
            selected.extend(chosen)
            if len(pool) < quota:
                deficits.append({"original_task": original, "requested": quota,
                                 "available": len(pool), "deficit": quota - len(pool)})
            for row in pool:
                pilot_prob[key(row)] = min(quota, len(pool)) / len(pool)
        if len(selected) < per_stratum:
            chosen_ids = {key(r) for r in selected}
            replacement = [r for r in candidates if key(r) not in chosen_ids]
            fill = rng.sample(replacement, per_stratum - len(selected))
            selected.extend(fill)
            # The fallback is a two-stage design with overlapping selection
            # paths. Its exact marginal probabilities are not claimed here.
            for row in candidates:
                pilot_prob[key(row)] = None
        pilot.extend(selected)
        allocation.append({"suite": suite, "category": category,
                           "population": len(candidates), "pilot": len(selected),
                           "sampling_fraction": len(selected) / len(candidates),
                           "original_tasks_present": len(original_names),
                           "missing_original_tasks": missing_originals,
                           "original_task_deficits": deficits,
                           "fallback_used": bool(deficits)})
    pilot_ids = {key(row) for row in pilot}
    if len(pilot_ids) != len(pilot):
        raise ValueError("Pilot contains duplicate variant IDs")
    remaining_groups = {k: [r for r in rows if key(r) not in pilot_ids]
                        for k, rows in groups.items()}
    heldout_allocation = largest_remainder(
        {k: len(rows) for k, rows in remaining_groups.items()}, heldout_size)
    heldout = []
    for group_key, rows in remaining_groups.items():
        count = heldout_allocation[group_key]
        heldout.extend(rng.sample(rows, count))
        report = next(r for r in allocation
                      if (r["suite"], r["category"]) == group_key)
        report["heldout"] = count
        report["heldout_sampling_fraction_remaining"] = count / len(rows)
        report["heldout_sampling_fraction_population"] = count / report["population"]
    if {key(r) for r in heldout} & pilot_ids:
        raise ValueError("Pilot and held-out overlap")

    def item(row: dict, split: str) -> dict:
        group = (row["suite"], row["category"])
        group_report = next(r for r in allocation
                            if (r["suite"], r["category"]) == group)
        p_pilot = pilot_prob[key(row)]
        p_heldout_conditional = (group_report["heldout"] /
                                 (group_report["population"] - group_report["pilot"]))
        p_heldout = ((1 - p_pilot) * p_heldout_conditional
                     if p_pilot is not None else None)
        return {"suite": row["suite"], "task_id": row["task_id"],
                "task_name": row["name"], "original_task": row["original_task"],
                "perturbation_category": row["category"],
                "difficulty_level": row["difficulty_level"],
                "sampling_seed": seed, "split": split,
                "pilot_inclusion_probability": p_pilot,
                "heldout_inclusion_probability": p_heldout,
                "heldout_conditional_probability_given_not_pilot": p_heldout_conditional}

    order = lambda r: (list(SUITE_COUNTS).index(r["suite"]), r["task_id"])
    return {"pilot": [item(r, "pilot") for r in sorted(pilot, key=order)],
            "heldout": [item(r, "heldout") for r in sorted(heldout, key=order)],
            "allocation": allocation,
            "sampling_algorithm": ALGORITHM,
            "sampling_seed": seed,
            "requested_per_stratum": per_stratum,
            "requested_heldout_size": heldout_size}


def distribution(rows: list[dict], *, manifest: bool = False) -> dict:
    cat_field = "perturbation_category" if manifest else "category"
    return {
        "suite": dict(sorted(Counter(r["suite"] for r in rows).items())),
        "category": dict(sorted(Counter(r[cat_field] for r in rows).items())),
        "difficulty": {str(k) if k is not None else "unknown": v
                       for k, v in sorted(Counter(r["difficulty_level"] for r in rows).items(),
                                          key=lambda x: (x[0] is None, x[0] or 0))},
    }
