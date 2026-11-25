"""Render the pinned LIBERO / LIBERO-PLUS inventory as a table figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.benchmark.plus_inventory import CATEGORIES, SUITE_COUNTS


def render(inventory: Path, output: Path) -> tuple[Path, Path]:
    category_file = inventory / "suite_category.csv"
    original_file = inventory / "suite_original_category.csv"
    with category_file.open(newline="") as stream:
        category_rows = list(csv.DictReader(stream))
    with original_file.open(newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    counts = {(row["suite"], row["category"]): int(row["count"])
              for row in category_rows}
    expected = {(suite, category) for suite in SUITE_COUNTS for category in CATEGORIES}
    if len(category_rows) != 28 or set(counts) != expected:
        raise ValueError("Expected one count for each of the 28 suite × category cells")
    originals = {suite: {row["original_task"] for row in original_rows if row["suite"] == suite}
                 for suite in SUITE_COUNTS}
    if any(len(originals[suite]) != 10 for suite in SUITE_COUNTS):
        raise ValueError("Expected ten original tasks in each PLUS suite")
    if sum(counts.values()) != 10030:
        raise ValueError("Expected 10,030 classified PLUS variants")
    for suite in SUITE_COUNTS:
        for category in CATEGORIES:
            grouped = sum(int(row["count"]) for row in original_rows
                          if row["suite"] == suite and row["category"] == category)
            if grouped != counts[suite, category]:
                raise ValueError(f"Original-task counts do not sum to {suite} × {category}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["Suite", "Original\ntasks", "Objects\nlayout", "Camera\nviewpoints",
              "Robot initial\nstates", "Language\ninstructions", "Light\nconditions",
              "Background\ntextures", "Sensor\nnoise", "PLUS\nvariants"]
    suite_labels = {"libero_spatial": "Spatial", "libero_object": "Object",
                    "libero_goal": "Goal", "libero_10": "Long (LIBERO-10)"}
    data = []
    for suite in SUITE_COUNTS:
        values = [counts[suite, category] for category in CATEGORIES]
        data.append([suite_labels[suite], "10", *(f"{value:,}" for value in values),
                     f"{sum(values):,}"])
    category_totals = [sum(counts[suite, category] for suite in SUITE_COUNTS)
                       for category in CATEGORIES]
    data.append(["PLUS total", "40", *(f"{value:,}" for value in category_totals),
                 "10,030"])

    output.mkdir(parents=True, exist_ok=True)
    png = output / "libero_plus_inventory_table.png"
    svg = output / "libero_plus_inventory_table.svg"
    manifest = output / "manifest.json"
    if any(path.exists() for path in (png, svg, manifest)):
        raise FileExistsError("Inventory table output already exists")

    fig, ax = plt.subplots(figsize=(18, 6.1), dpi=160)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    fig.text(0.035, 0.945, "LIBERO tasks and LIBERO-PLUS perturbations",
             fontsize=19, weight="bold", color="#17324d", va="top")
    fig.text(0.035, 0.885, "Official metadata counts  ·  4 evaluated suites  ·  7 perturbation categories",
             fontsize=10, color="#526778", va="top")
    widths = [0.145, 0.09, 0.096, 0.105, 0.108, 0.112, 0.095, 0.105, 0.09, 0.084]
    table = ax.table(cellText=data, colLabels=titles, colWidths=widths,
                     cellLoc="center", colLoc="center", bbox=[0.025, 0.26, 0.95, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8e2e9")
        cell.set_linewidth(0.8)
        cell.PAD = 0.045
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9)
        elif row == 5:
            cell.set_facecolor("#dceaf4")
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#17324d")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f4f8fb")
            cell.get_text().set_color("#243746")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")
    fig.text(0.035, 0.175,
             "Original LIBERO: 130 tasks = 10 Spatial + 10 Object + 10 Goal + 10 Long + 90 LIBERO-90.",
             fontsize=10, weight="bold", color="#17324d")
    fig.text(0.035, 0.125,
             "LIBERO-PLUS variant inventory covers the first four suites (40 original tasks). "
             "LIBERO-90 has no PLUS category counts in this inventory.",
             fontsize=9.5, color="#526778")
    fig.text(0.035, 0.055,
             "Source: pinned LIBERO-PLUS task_classification.json and benchmark task map; "
             "counts are distinct OOD task IDs, not rollout trials.",
             fontsize=8.5, color="#667987")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.write_text(json.dumps({
        "suite_category_sha256": hashlib.sha256(category_file.read_bytes()).hexdigest(),
        "suite_original_category_sha256": hashlib.sha256(original_file.read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.inventory.resolve(), args.output.resolve()):
        print(path)


if __name__ == "__main__":
    main()
