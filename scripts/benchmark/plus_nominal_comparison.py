"""Compare nominal LIBERO with seven PLUS shifts for matching original tasks.

This is a descriptive comparison: nominal and PLUS use different trials,
reset protocols, and action-step limits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

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
    strata_file = plus_dir / "strata_summary.csv"
    mapping_file = plus_dir / "original_task_mapping.csv"
    nominal_base = root / "outputs/nominal"
    run_dirs = {
        (model, mode): nominal_base / f"nominal-4suite-{condition}-v1"
        for model, prefix in (("sft", "sft_base_only"), ("grpo", "grpo"))
        for mode in MODES
        for condition in (f"{prefix}_{mode}",)
    }
    sources = (strata_file, mapping_file, *(
        run_dir / name for run_dir in run_dirs.values()
        for name in ("episodes.csv", "summary.json", "sources.json")
    ))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    names = {suite: set() for suite in SUITE_COUNTS}
    for row in read_csv(mapping_file):
        names[row["suite"]].add(row["original_task_name"])
    if any(len(names[suite]) != 10 for suite in SUITE_COUNTS):
        raise ValueError("Expected ten original tasks per PLUS suite")
    counts = {}
    episode_sets = []
    reset_hashes = {}
    for (model, mode), run_dir in run_dirs.items():
        summary = json.loads((run_dir / "summary.json").read_text())
        if (summary.get("aggregate_view") is not True or
                summary.get("expected") != 200 or summary.get("completed") != 200 or
                set(summary["by_suite"]) != set(SUITE_COUNTS) or
                any(item["completed"] != 50 for item in summary["by_suite"].values())):
            raise ValueError(f"Incomplete four-suite nominal view: {run_dir}")
        by_id = {row["episode_id"]: row for row in read_csv(run_dir / "episodes.csv")}
        if len(by_id) != 200:
            raise ValueError(f"Duplicate or missing nominal episodes: {run_dir}")
        expected_ids = {f"{suite}-t{task:02d}-i{trial:02d}"
                        for suite in SUITE_COUNTS for task in range(10) for trial in range(5)}
        if set(by_id) != expected_ids:
            raise ValueError(f"Nominal episode coverage differs: {run_dir}")
        episode_sets.append(set(by_id))
        seen_names = {suite: set() for suite in SUITE_COUNTS}
        for eid, row in by_id.items():
            metadata_path = run_dir / "rollouts" / eid / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            suite = row["suite"]
            if (suite not in SUITE_COUNTS or metadata["episode_id"] != eid or
                    metadata["suite"] != suite or
                    metadata["success"] != bool(success(row["success"])) or
                    metadata["sampling_mode"] != mode or
                    metadata["lora_applied"] is not False or
                    int(row["task_id"]) != metadata["task_id"] or
                    int(row["trial_id"]) != metadata["trial_id"]):
                raise ValueError(f"Nominal metadata differs: {metadata_path}")
            if eid in reset_hashes and reset_hashes[eid] != metadata["initial_state_sha256"]:
                raise ValueError(f"Nominal reset differs across conditions: {eid}")
            reset_hashes[eid] = metadata["initial_state_sha256"]
            seen_names[suite].add(metadata["task_name"])
            counts[suite, mode, model] = counts.get((suite, mode, model), 0) + success(row["success"])
        if any(seen_names[suite] != names[suite] for suite in SUITE_COUNTS):
            raise ValueError(f"Nominal original-task names differ: {run_dir}")
        if summary["successes"] != sum(success(row["success"]) for row in by_id.values()):
            raise ValueError(f"Nominal summary differs: {run_dir}")
    if any(episode_set != episode_sets[0] for episode_set in episode_sets[1:]):
        raise ValueError("Nominal conditions contain different episode IDs")

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
                successes = counts[suite, mode, model]
                rows.append({"suite": suite, "sampling_mode": mode, "setting": "Original LIBERO",
                             "model": model.upper(), "n": 50, "successes": successes,
                             "success_rate": successes / 50})
                for category in CATEGORIES:
                    row = plus[suite, category, mode]
                    n, successes = int(row["N"]), int(row[f"{model}_successes"])
                    rows.append({"suite": suite, "sampling_mode": mode, "setting": category,
                                 "model": model.upper(), "n": n, "successes": successes,
                                 "success_rate": successes / n})
    return rows, hashes


def svg_chart(rows: list[dict]) -> str:
    lookup = {(r["suite"], r["sampling_mode"], r["setting"], r["model"]): r for r in rows}
    settings = ["Original LIBERO", *CATEGORIES]
    labels = (("Original", "LIBERO"), ("Objects", "layout"), ("Camera", "views"),
              ("Robot", "initial"), ("Language", "instructions"),
              ("Lighting", ""), ("Background", "textures"), ("Sensor", "noise"))
    colors = {"SFT": "#4878A8", "GRPO": "#D3843B"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1230" viewBox="0 0 2400 1230">',
             '<rect width="2400" height="1230" fill="#f7f9fc"/>',
             '<text x="1200" y="35" text-anchor="middle" font-family="Arial,sans-serif" font-size="24" font-weight="bold" fill="#17243a">Original LIBERO versus LIBERO-PLUS: matched task families</text>']
    for model, x in (("SFT", 1100), ("GRPO", 1210)):
        parts.append(f'<rect x="{x}" y="52" width="18" height="14" fill="{colors[model]}"/>')
        parts.append(f'<text x="{x+25}" y="64" font-family="Arial,sans-serif" font-size="14" fill="#17243a">{model}</text>')
    for i, mode in enumerate(MODES):
        for j, suite in enumerate(SUITE_COUNTS):
            px, py = 48 + j * 583, 82 + i * 565
            gx, gy, gh = px + 48, py + 104, 315
            title = "Long" if suite == "libero_10" else suite.removeprefix("libero_").title()
            parts.append(f'<rect x="{px}" y="{py}" width="555" height="530" rx="10" fill="white" stroke="#d6dfe9"/>')
            parts.append(f'<text x="{px+277}" y="{py+34}" text-anchor="middle" font-family="Arial,sans-serif" font-size="19" font-weight="bold" fill="#17243a">{title} · {mode.title()}</text>')
            for tick in (0, 25, 50, 75, 100):
                y = gy + gh - gh * tick / 100
                parts.append(f'<line x1="{gx}" y1="{y}" x2="{gx+464}" y2="{y}" stroke="#e2e8ef"/>')
                parts.append(f'<text x="{gx-9}" y="{y+4}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#52647a">{tick}</text>')
            parts.append(f'<text x="{gx+234}" y="{py+69}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#52647a">Success rate (%) · labels show successes / trials (rate)</text>')
            for k, setting in enumerate(settings):
                for model, offset in (("SFT", 9), ("GRPO", 32)):
                    row = lookup[suite, mode, setting, model]
                    rate = row["success_rate"] * 100
                    h = max(0.8, gh * rate / 100)
                    x = gx + k * 58 + offset
                    y = gy + gh - h
                    parts.append(f'<rect x="{x}" y="{y:.2f}" width="18" height="{h:.2f}" rx="2" fill="{colors[model]}"/>')
                    pct = f'{rate:.1f}%' if row["n"] == 19 else f'{rate:.0f}%'
                    value = escape(f'{row["successes"]}/{row["n"]} ({pct})')
                    tx, ty = x + 11, y - 7
                    parts.append(f'<text x="{tx}" y="{ty:.2f}" transform="rotate(-90 {tx} {ty:.2f})" font-family="Arial,sans-serif" font-size="9" fill="#233346">{value}</text>')
                center = gx + k * 58 + 29
                for line_no, label in enumerate(labels[k]):
                    if label:
                        parts.append(f'<text x="{center}" y="{gy+gh+22+line_no*13}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#33465c">{escape(label)}</text>')
                if k == 0:
                    parts.append(f'<line x1="{gx+58}" y1="{gy-3}" x2="{gx+58}" y2="{gy+gh+2}" stroke="#98a6b6" stroke-dasharray="4 4"/>')
    parts.append('<text x="1200" y="1220" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52647a">Original: 10 tasks × 5 trials per suite, 520 steps. PLUS: ~20 variants per category and suite, suite-specific horizons. Different trials and resets; descriptive rates only.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render(root: Path, output: Path) -> tuple[Path, Path]:
    rows, hashes = build_rows(root)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "nominal_vs_plus_success_rates.png"
    svg = output / "nominal_vs_plus_success_rates.svg"
    csv_file = output / "comparison.csv"
    manifest = output / "manifest.json"
    if any(path.exists() for path in (png, svg, csv_file, manifest)):
        raise FileExistsError("Comparison output already exists")
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to render the comparison PNG")
    svg.write_text(svg_chart(rows))
    subprocess.run([converter, "-o", str(png), str(svg)], check=True)
    with csv_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("suite", "sampling_mode", "setting", "model",
                                                    "n", "successes", "success_rate"))
        writer.writeheader()
        writer.writerows(rows)
    manifest.write_text(json.dumps({"source_sha256": hashes,
                                    "figure_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                                    "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
                                    "comparison_csv_sha256": hashlib.sha256(csv_file.read_bytes()).hexdigest(),
                                    "nominal_object_status": "available: 50 trials per condition",
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
