# LIBERO-PLUS pilot generalization analysis

This CPU-only analysis reads the completed `plus-pilot-suite-nolora-v2` rollout metadata. It never starts a policy or simulator. The official 560-task pilot manifest remains unchanged. The two observed-start mismatches, `libero_spatial-t2396-i00` and `libero_goal-t2410-i00`, are validated as source records, then excluded from all four conditions. The resulting analysis contains 558 matched task IDs and 2,232 condition-episode rows.

The comparison means **matched task and verified initial observation hash**, not verified identical physical simulator state. The PLUS collector did not apply stored official initial states. A matching hash covers selected robot observations and camera pixels, not the full MuJoCo state. The 558-task sample has one missing member in each of two Light Conditions original-task cells. The original pilot inclusion probabilities do not identify complete-case population weights after these observed-start-based exclusions. The population-weighted estimate is therefore `NOT IDENTIFIABLE`; episode success rates and task macro averages are descriptive pilot results.

Original task names come from the validated official inventory and task map. The analysis checks the pilot manifest's inferred original-task names against that inventory. Difficulty is recorded separately from severity; one excluded Goal task has missing difficulty, leaving six unknown-difficulty episodes in the 558-task sample. The official metadata does not provide an independently verified severity field, so severity analysis is unavailable. The held-out 1,000-task manifest is not read by the analysis.

## Run on the evaluation server

Use the server checkout after syncing these files. The Python environment needs Matplotlib to render figures; the analysis and validation commands need only the standard library and existing repository code.

```bash
ROOT=/nfs4/jhkim/repos/rlinf
PY="$ROOT/upstream/RLinf/.venv-rlvr-liberoplus-cu121/bin/python"
export PYTHONPATH="$ROOT:$ROOT/upstream/RLinf:${PYTHONPATH:-}"
cd "$ROOT"

"$PY" -m unittest discover -s tests -p 'test_plus_*.py' -v

"$PY" -m scripts.benchmark.plus_analysis validate \
  --pilot "$ROOT/manifests/libero_plus/pilot_v1.json" \
  --run-dir "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2"

"$PY" -m scripts.benchmark.plus_analysis analyze \
  --pilot "$ROOT/manifests/libero_plus/pilot_v1.json" \
  --run-dir "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2" \
  --output "$ROOT/outputs/analysis/plus_generalization/pilot_v1"

"$PY" -m scripts.benchmark.plus_figures plot \
  --input "$ROOT/outputs/analysis/plus_generalization/pilot_v1"
```

`analyze` and `plot` refuse to overwrite existing output. `validate` is read-only. Run `--help` on either module for CLI details.

## Outputs and interpretation

The output directory contains source and output SHA256 hashes in `manifest.json`, validation details in `validation.json`, a 2,232-row `episode_table.csv`, `strata_summary.csv`, `original_task_summary.csv`, `paired_outcomes.csv`, `difficulty_summary.csv`, `concentration_summary.json`, `population_weighting_audit.json`, `nominal_compatibility.json`, and `summary.md`. `original_task_mapping.csv` maps short figure labels to full names. `figures/manifest.json` records all figure and input CSV hashes.

Deterministic and stochastic results are separate throughout. Every suite × shift stratum records successes, success rates, paired gained/lost/both outcomes, and GRPO−SFT percentage point differences. Original-task-clustered bootstrap intervals use seed 42 and resample complete SFT/GRPO pairs within suite. The 28 strata each contain roughly 20 episodes and each original-task × shift cell roughly two; cell-level changes are imprecise. The figures show descriptive distribution and concentration, not statistical significance.

Nominal results are audited from available nominal run manifests only. Distribution, task mapping, horizon, and reset compatibility are not established for a direct robustness-gap estimate; no nominal rollout is started or pooled with PLUS.
