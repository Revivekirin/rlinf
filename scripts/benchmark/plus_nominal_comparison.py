"""Compare nominal LIBERO with seven PLUS shifts for matching original tasks.

This is a descriptive comparison: nominal and PLUS use different trials,
reset protocols, and action-step limits. Missing nominal suites remain missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS


MODES = ("deterministic", "stochastic")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def success(value: str) -> int:
    if value not in ("True", "False"):
        raise ValueError(f"Invalid success value: {value}")
    return int(value == "True")


def build_rows(root: Path) -> tuple[list[dict], dict[str, str]]:
    plus_dir = root / "outputs/analysis/plus_generalization/pilot_v1"
    phase3a = root / "outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246/paired/paired_episodes.csv"
    sft_sampling = root / "outputs/analysis/sft_sampling/phase3b-20261117T111837Z-2822692/paired_episodes.csv"
    strata_file = plus_dir / "strata_summary.csv"
    mapping_file = plus_dir / "original_task_mapping.csv"
    sources = (phase3a, sft_sampling, strata_file, mapping_file)
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}

    nominal = read_csv(phase3a)
    sft = {row["episode_id"]: row for row in read_csv(sft_sampling)}
    primary = {row["episode_id"]: row for row in nominal if row["comparison"] == "primary"}
    secondary = {row["episode_id"]: row for row in nominal if row["comparison"] == "secondary"}
    if (len(primary) != 150 or len(secondary) != 150 or len(sft) != 150 or
            set(primary) != set(secondary) or set(primary) != set(sft)):
        raise ValueError("Nominal episode IDs or counts differ")
    names = {suite: set() for suite in SUITE_COUNTS}
    for row in read_csv(mapping_file):
        names[row["suite"]].add(row["original_task_name"])
    if any(len(names[suite]) != 10 for suite in SUITE_COUNTS):
        raise ValueError("Expected ten original tasks per PLUS suite")
    nominal_names = {suite: set() for suite in SUITE_COUNTS}
    counts = {(suite, mode, model): 0 for suite in SUITE_COUNTS
              for mode in MODES for model in ("sft", "grpo")}
    suite_n = {suite: 0 for suite in SUITE_COUNTS}
    for eid, row in primary.items():
        suite = row["suite"]
        other, sft_row = secondary[eid], sft[eid]
        if (other["suite"] != suite or sft_row["suite"] != suite or
                success(row["left_success"]) != success(sft_row["deterministic_success"]) or
                success(row["right_success"]) != success(other["left_success"])):
            raise ValueError(f"Nominal condition mismatch: {eid}")
        nominal_names[suite].add(row["task_name"])
        suite_n[suite] += 1
        counts[suite, "deterministic", "sft"] += success(row["left_success"])
        counts[suite, "deterministic", "grpo"] += success(row["right_success"])
        counts[suite, "stochastic", "sft"] += success(sft_row["stochastic_success"])
        counts[suite, "stochastic", "grpo"] += success(other["right_success"])
    for suite in SUITE_COUNTS:
        expected_n = 0 if suite == "libero_object" else 50
        if suite_n[suite] != expected_n or (expected_n and nominal_names[suite] != names[suite]):
            raise ValueError(f"Nominal original-task coverage differs: {suite}")

    rows = []
    strata = read_csv(strata_file)
    if len(strata) != 56:
        raise ValueError("Expected 56 PLUS suite × category × mode rows")
    plus = {(row["suite"], row["shift_category"], row["sampling_mode"]): row
            for row in strata}
    if len(plus) != 56 or len({row["input_set_sha256"] for row in strata}) != 1:
        raise ValueError("PLUS strata identity or provenance mismatch")
    for suite in SUITE_COUNTS:
        for mode in MODES:
            for model in ("sft", "grpo"):
                if suite_n[suite]:
                    successes = counts[suite, mode, model]
                    rows.append({"suite": suite, "sampling_mode": mode, "setting": "Original LIBERO",
                                 "model": model.upper(), "n": suite_n[suite], "successes": successes,
                                 "success_rate": successes / suite_n[suite]})
                for category in CATEGORIES:
                    row = plus[suite, category, mode]
                    n, successes = int(row["N"]), int(row[f"{model}_successes"])
                    rows.append({"suite": suite, "sampling_mode": mode, "setting": category,
                                 "model": model.upper(), "n": n, "successes": successes,
                                 "success_rate": successes / n})
    return rows, hashes


def render(root: Path, output: Path) -> tuple[Path, Path]:
    rows, hashes = build_rows(root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    png = output / "nominal_vs_plus_success_rates.png"
    csv_file = output / "comparison.csv"
    manifest = output / "manifest.json"
    if any(path.exists() for path in (png, csv_file, manifest)):
        raise FileExistsError("Comparison output already exists")

    lookup = {(r["suite"], r["sampling_mode"], r["setting"], r["model"]): r for r in rows}
    labels = ["Original\nLIBERO", "Objects\nLayout", "Camera\nViewpoints",
              "Robot Initial\nStates", "Language\nInstructions", "Light\nConditions",
              "Background\nTextures", "Sensor\nNoise"]
    settings = ["Original LIBERO", *CATEGORIES]
    colors = {"SFT": "#4878A8", "GRPO": "#D3843B"}
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 4, figsize=(23, 10), sharey=True)
    for i, mode in enumerate(MODES):
        for j, suite in enumerate(SUITE_COUNTS):
            ax = axes[i, j]
            for model, offset in (("SFT", -0.18), ("GRPO", 0.18)):
                for k, setting in enumerate(settings):
                    row = lookup.get((suite, mode, setting, model))
                    if row is None:
                        continue
                    rate = row["success_rate"] * 100
                    ax.bar(k + offset, rate, width=0.35, color=colors[model])
                    ax.text(k + offset, rate + 1.5, str(row["successes"]),
                            ha="center", va="bottom", fontsize=7, color="#202020")
            ax.set_xticks(range(8), labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 110)
            ax.set_yticks((0, 25, 50, 75, 100))
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
            ax.set_title(suite.replace("libero_", "").replace("10", "Long").title())
            if j == 0:
                ax.set_ylabel(f"{mode.capitalize()}\nSuccess rate (%)", fontweight="bold")
            if suite == "libero_object":
                ax.text(0, 50, "N/A", ha="center", va="center", fontsize=9,
                        color="#666666", rotation=90)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[model]) for model in ("SFT", "GRPO")]
    fig.legend(handles, ("SFT", "GRPO"), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.96), frameon=False)
    fig.suptitle("Same original tasks: standard LIBERO versus LIBERO-PLUS shifts", fontsize=15)
    fig.text(0.5, 0.015,
             "Original LIBERO: 10 tasks × 5 trials/suite, 520 steps; Object unavailable. "
             "PLUS: same 10 tasks, ~20 variants/category/suite, suite-specific horizons. "
             "Numbers above bars are successes. Different trials and resets; descriptive rates only.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.89, bottom=0.16,
                        wspace=0.18, hspace=0.60)
    fig.savefig(png, dpi=160)
    plt.close(fig)

    with csv_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("suite", "sampling_mode", "setting", "model",
                                                    "n", "successes", "success_rate"))
        writer.writeheader()
        writer.writerows(rows)
    manifest.write_text(json.dumps({"source_sha256": hashes,
                                    "figure_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                                    "comparison_csv_sha256": hashlib.sha256(csv_file.read_bytes()).hexdigest(),
                                    "nominal_object_status": "unavailable",
                                    "interpretation": "descriptive; different trials, resets, horizons"},
                                   indent=2, sort_keys=True) + "\n")
    return png, csv_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.root.resolve(), args.output.resolve()):
        print(path)


if __name__ == "__main__":
    main()
