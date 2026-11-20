"""Render LIBERO-PLUS pilot figures from analyzed CSV files only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS

FIGURES = (
    "A1_strata_delta_deterministic.png", "A2_strata_delta_stochastic.png",
    "A3_gained_lost.png", "A4_suite_success.png",
    "B1_task_shift_deterministic.png", "B1_task_shift_stochastic.png",
    "B2_task_ranked.png", "B3_task_heterogeneity.png",
    "C_improvement_concentration.png",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_sources(input_dir: Path) -> tuple[dict[str, list[dict]], str]:
    names = ("strata_summary", "original_task_summary", "paired_outcomes", "difficulty_summary")
    rows = {name: read_rows(input_dir / f"{name}.csv") for name in names}
    hashes = {r["input_set_sha256"] for group in rows.values() for r in group}
    if len(hashes) != 1 or not all(rows.values()):
        raise ValueError("Figure input CSVs have inconsistent or missing source hashes")
    if len(rows["strata_summary"]) != 56 or len(rows["paired_outcomes"]) != 558:
        raise ValueError("Figure input dimensions differ from validated pilot")
    return rows, hashes.pop()


def plot(input_dir: Path) -> list[Path]:
    rows, source_hash = verify_sources(input_dir)
    figures = input_dir / "figures"
    if figures.exists() and any(figures.iterdir()):
        raise FileExistsError("Figure output already exists")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figures.mkdir(exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})

    def save(fig, name: str) -> None:
        fig.text(.01, .005,
                 "Matched task / observed start; N=558. Deterministic: do_sample=False, T=1.0; stochastic: do_sample=True, T=1.6. Cell N shown in tables.",
                 fontsize=6, color="#555555")
        fig.tight_layout()
        target = figures / name
        fig.savefig(target, bbox_inches="tight", metadata={"Description": f"input_set_sha256={source_hash}"})
        plt.close(fig)
        paths.append(target)

    strata = rows["strata_summary"]
    tasks = rows["original_task_summary"]
    paired = rows["paired_outcomes"]
    max_strata = max(abs(float(r["delta_sr_pp"])) for r in strata) or 1
    task_cells = [r for r in tasks if r["shift_category"] != "ALL"]
    max_task = max(abs(float(r["delta_sr_pp"])) for r in task_cells) or 1
    for mode, name in (("deterministic", FIGURES[0]), ("stochastic", FIGURES[1])):
        lookup = {(r["suite"], r["shift_category"]): r for r in strata if r["sampling_mode"] == mode}
        matrix = [[float(lookup[s, c]["delta_sr_pp"]) for c in CATEGORIES] for s in SUITE_COUNTS]
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-max_strata, vmax=max_strata))
        ax.set_xticks(range(7), CATEGORIES, rotation=35, ha="right")
        ax.set_yticks(range(4), SUITE_COUNTS)
        for i, s in enumerate(SUITE_COUNTS):
            for j, c in enumerate(CATEGORIES):
                r = lookup[s, c]
                ax.text(j, i, f"{float(r['delta_sr_pp']):+.0f}\nN={r['N']}", ha="center", va="center", fontsize=7)
        ax.set_title(f"GRPO − SFT ΔSR (pp), {mode}; do_sample={mode == 'stochastic'}, T={1.6 if mode == 'stochastic' else 1.0}")
        fig.colorbar(im, ax=ax, label="percentage points")
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode in zip(axes, ("deterministic", "stochastic")):
        selected = [r for r in strata if r["sampling_mode"] == mode]
        left = [0] * 7
        for field, color, label in (("both_success", "#4d79a8", "both success"),
                                    ("gained", "#247a53", "gained"),
                                    ("lost", "#b3443b", "lost"),
                                    ("both_fail", "#aaaaaa", "both fail")):
            vals = [sum(int(r[field]) for r in selected if r["shift_category"] == c) for c in CATEGORIES]
            ax.barh(range(7), vals, left=left, color=color, label=label)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(range(7), [f"{c} (N={n})" for c, n in zip(CATEGORIES, left)])
        ax.set_title(f"{mode}: gained/lost; N=80 per category except Light N=78")
        ax.set_xlabel("paired episode count")
        ax.legend()
    save(fig, FIGURES[2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ("deterministic", "stochastic")):
        for j, model in enumerate(("sft", "grpo")):
            values = []
            for suite in SUITE_COUNTS:
                cell = [r for r in paired if r["suite"] == suite]
                values.append(100 * sum(int(r[f"{model}_{mode}"]) for r in cell) / len(cell))
            ax.bar([i + (j - .5) * .35 for i in range(4)], values, width=.35,
                   label=model.upper(), color=("#6e91bb" if j == 0 else "#da8b42"))
        ax.set_xticks(range(4), [s.replace("libero_", "") for s in SUITE_COUNTS])
        ax.set_title(f"{mode}; N=139/140/139/140")
        ax.set_ylabel("success rate (%)")
        ax.legend()
    save(fig, FIGURES[3])

    task_names = sorted({r["original_task_id"] for r in task_cells})
    for mode, name in (("deterministic", FIGURES[4]), ("stochastic", FIGURES[5])):
        lookup = {(r["original_task_id"], r["shift_category"]): r for r in task_cells if r["sampling_mode"] == mode}
        matrix = [[float(lookup[t, c]["delta_sr_pp"]) for c in CATEGORIES] for t in task_names]
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(matrix, cmap="RdBu", aspect="auto",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-max_task, vmax=max_task))
        ax.set_xticks(range(7), CATEGORIES, rotation=35, ha="right")
        ax.set_yticks(range(40), task_names, fontsize=6)
        ax.set_title(f"Original task × shift ΔSR (pp), {mode}; ~2 episodes/cell; see original_task_mapping.csv")
        fig.colorbar(im, ax=ax, label="percentage points")
        save(fig, name)

    all_tasks = [r for r in tasks if r["shift_category"] == "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 11), sharex=True)
    for ax, mode in zip(axes, ("deterministic", "stochastic")):
        selected = sorted((r for r in all_tasks if r["sampling_mode"] == mode),
                          key=lambda r: float(r["delta_sr_pp"]))
        ax.scatter([float(r["delta_sr_pp"]) for r in selected], range(40), s=14)
        ax.set_yticks(range(40), [r["original_task_id"] for r in selected], fontsize=6)
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_xlabel("GRPO − SFT ΔSR (pp)")
        ax.set_title(f"{mode}; N≈14 per task")
    save(fig, FIGURES[6])

    fig, ax = plt.subplots(figsize=(10, 4))
    labels, positions, values = [], [], []
    for i, suite in enumerate(SUITE_COUNTS):
        for j, mode in enumerate(("deterministic", "stochastic")):
            vals = [float(r["delta_sr_pp"]) for r in all_tasks if r["suite"] == suite and r["sampling_mode"] == mode]
            labels.append(f"{suite.replace('libero_', '')}\n{mode[:3]}")
            positions.append(2 * i + j)
            values.append(vals)
    ax.boxplot(values, positions=positions, widths=.6)
    ax.set_xticks(positions, labels)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("Original task ΔSR (pp); 10 tasks/suite")
    save(fig, FIGURES[7])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, dimension, labels in ((axes[0], "suite", list(SUITE_COUNTS)),
                                  (axes[1], "perturbation_category", list(CATEGORIES))):
        for j, mode in enumerate(("deterministic", "stochastic")):
            vals = [sum(int(r[f"grpo_{mode}"]) - int(r[f"sft_{mode}"])
                        for r in paired if r[dimension] == label) for label in labels]
            ax.bar([i + (j - .5) * .35 for i in range(len(labels))], vals, width=.35,
                   label=mode, color=("#4d79a8" if j == 0 else "#c47750"))
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_ylabel("Net gained successes (GRPO − SFT)")
        ax.legend()
    top = []
    for mode in ("deterministic", "stochastic"):
        ranked = sorted((r for r in all_tasks if r["sampling_mode"] == mode),
                        key=lambda r: -(int(r["grpo_successes"]) - int(r["sft_successes"])))[:5]
        top.extend((f"{r['original_task_id']}\n{mode[:3]}",
                    int(r["grpo_successes"]) - int(r["sft_successes"])) for r in ranked)
    axes[2].barh(range(len(top)), [item[1] for item in top], color="#638d75")
    axes[2].set_yticks(range(len(top)), [item[0] for item in top], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Task net gained successes")
    axes[2].set_title("Top 5 original tasks per sampling mode")
    fig.suptitle("Improvement concentration; paired N=558; descriptive counts")
    save(fig, FIGURES[8])

    if rows["difficulty_summary"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        data = [r for r in rows["difficulty_summary"] if r["shift_category"] == "ALL"]
        for mode, marker in (("deterministic", "o"), ("stochastic", "s")):
            selected = [r for r in data if r["sampling_mode"] == mode]
            ax.plot(range(len(selected)), [float(r["delta_sr_pp"]) for r in selected], marker=marker, label=mode)
        ax.set_xticks(range(len(selected)), [f"{r['difficulty_level']}\nN={r['N']}" for r in selected])
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xlabel("Official difficulty level (unknown kept separate)")
        ax.set_ylabel("GRPO − SFT ΔSR (pp)")
        ax.legend()
        save(fig, "D_difficulty.png")
    source_csv = {name: hashlib.sha256((input_dir / f"{name}.csv").read_bytes()).hexdigest()
                  for name in rows}
    figure_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = figures / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps({"input_set_sha256": source_hash,
                                    "source_csv_sha256": source_csv,
                                    "figure_sha256": figure_hashes}, indent=2, sort_keys=True) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.input.resolve()):
        print(path)


if __name__ == "__main__":
    main()
