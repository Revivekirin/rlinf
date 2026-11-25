"""Plot SFT and GRPO success counts by LIBERO-PLUS suite and shift category."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS


MODES = ("deterministic", "stochastic")


def plot_counts(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    source = input_dir / "strata_summary.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(suite, category, mode)
                for suite in SUITE_COUNTS for category in CATEGORIES for mode in MODES}
    lookup = {(row["suite"], row["shift_category"], row["sampling_mode"]): row
              for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Expected exactly one row for each suite × category × mode")
    hashes = {row["input_set_sha256"] for row in rows}
    if len(hashes) != 1:
        raise ValueError("Input rows have inconsistent source hashes")
    for row in rows:
        n = int(row["N"])
        if n not in (19, 20) or any(not 0 <= int(row[key]) <= n
                                     for key in ("sft_successes", "grpo_successes")):
            raise ValueError("Invalid stratum denominator or success count")
    for mode in MODES:
        if sum(int(row["N"]) for row in rows if row["sampling_mode"] == mode) != 558:
            raise ValueError(f"Expected 558 paired episodes for {mode}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    colors = {"sft": "#4878A8", "grpo": "#D3843B"}
    for i, mode in enumerate(MODES):
        for j, suite in enumerate(SUITE_COUNTS):
            ax = axes[i, j]
            for k, category in enumerate(CATEGORIES):
                row = lookup[suite, category, mode]
                for model, offset in (("sft", -0.19), ("grpo", 0.19)):
                    value = int(row[f"{model}_successes"])
                    ax.barh(k + offset, value, height=0.34, color=colors[model],
                            label=model.upper() if (i, j, k) == (0, 0, 0) else None)
                    ax.text(value + 0.25, k + offset, str(value), va="center", fontsize=8)
            ax.set_yticks(range(7), CATEGORIES)
            ax.invert_yaxis()
            ax.set_xlim(0, 22)
            ax.set_xticks((0, 5, 10, 15, 20))
            ax.grid(axis="x", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(suite.replace("libero_", "").replace("10", "long").title())
            if j == 0:
                ax.set_ylabel(mode.capitalize(), fontweight="bold")
            if i == 1:
                ax.set_xlabel("Successful episodes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[model]) for model in ("sft", "grpo")]
    fig.legend(handles, ("SFT", "GRPO"), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle("LIBERO-PLUS pilot: success counts by suite and shift category", fontsize=15)
    fig.text(0.5, 0.012,
             "Paired episodes: 558 per mode; N=20 per cell except Spatial/Goal × Light Conditions (N=19). "
             "Two episodes with mismatched observed starts excluded.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.09,
                        wspace=0.42, hspace=0.27)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "suite_category_success_counts.png"
    svg = output_dir / "suite_category_success_counts.svg"
    for path in (png, svg):
        if path.exists():
            raise FileExistsError(path)
    fig.savefig(png, dpi=160)
    fig.savefig(svg)
    plt.close(fig)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps({
        "input_set_sha256": hashes.pop(),
        "strata_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_counts(args.input.resolve(), args.output.resolve()):
        print(path)


if __name__ == "__main__":
    main()
