# Stage 3 extension — action-space distribution analysis

## Inputs and scope

The CLI is `scripts/analysis/action_space_distribution.py`.
It reads validated Stage 1 logits and the matching sealed Stage 2/3 run; existing outputs and source modules are not modified.

Current local inputs:

- Stage 1: `outputs/analysis/distribution_generality/stage1-20261118T035126Z-2960548`
- Stage 2/3: `outputs/analysis/token_action/stage23-20261118-final`
- Extension result: `outputs/analysis/action_space_distribution/stage3ext-20261118`

The Stage 1 copy has 12 missing original figure files. Required numerical files and bank/worker seals pass. This partial-copy status remains explicit in the extension manifest and report.

## Implemented commands: local reproduction

Run from the local checkout. A new output directory is required. These commands perform no model inference or simulator rollout.

```bash
set -e
cd /Users/jihyekim/Desktop/rlinf
PY=/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python
CLI=scripts/analysis/action_space_distribution.py
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-stage3ext-mpl"
S1="$PWD/outputs/analysis/distribution_generality/stage1-20261118T035126Z-2960548"
S23="$PWD/outputs/analysis/token_action/stage23-20261118-final"
OUT="$PWD/outputs/analysis/action_space_distribution/stage3ext-$(date -u +%Y%m%dT%H%M%SZ)-$$"
TESTLOG="${TMPDIR:-/tmp}/rlinf-stage3ext-tests-$$.txt"

# 1. Input audit (omitted input flags search the newest eligible verified runs)
"$PY" -B "$CLI" audit --stage1 "$S1" --stage23 "$S23" --output "$OUT" --bootstrap 500 --threshold-band 0.05
# 2. Native action-bin mapping and every saved greedy action
"$PY" -B "$CLI" mapping --output "$OUT"
# 3. Scalar distribution metrics, aggregation, bootstrap, previous-result regression
"$PY" -B "$CLI" analyze --output "$OUT"
# 4. Figures A–G, each at T=1.0 and T=1.6, PNG and PDF
"$PY" -B "$CLI" figures --output "$OUT"
# 5. Fixed-rule representative cases and probability mass plots
"$PY" -B "$CLI" cases --output "$OUT"
# 6. CPU regression suite
"$PY" -B -m unittest discover -s tests -p 'test_*.py' > "$TESTLOG" 2>&1
# 7. Combined report, final seal, and full input/output verification
"$PY" -B "$CLI" finalize --output "$OUT" --test-log "$TESTLOG"
```

To verify the completed local result again:

```bash
/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python -B \
  scripts/analysis/action_space_distribution.py verify \
  --output outputs/analysis/action_space_distribution/stage3ext-20261118
```

On the server, use `/nfs4/jhkim/repos/rlinf` as the working directory and `upstream/RLinf/.venv-rlvr-libero-cu121/bin/python` as `PY`. Select the corresponding Stage 1 and Stage 2/3 run produced on that host, with consistent manifest lineage. No upstream installation/change is needed for this extension: it reuses the previously audited decoder implementation and source evidence.

`--bootstrap` and `--threshold-band` are recorded by `audit`; later stages use those recorded settings. Use a new run to change them. Source-code hashes are also fixed at audit time.

## New modules

- `action_space_core.py`: stable softmax, native support reconstruction, scalar moments, exact W1, binary gripper probabilities, weighted quantiles/Spearman, whole-task bootstrap.
- `action_space_distribution.py`: seven-stage CLI, saved-data regressions, all aggregations, representative selection and seals.
- `action_space_figures.py`: A–G and representative plots, saved-table consistency check, combined report.
- `tests/test_action_space_distribution.py`: CPU tests including SciPy W1 cross-checks and alias/nonmonotonic support.

Reused without edits: `action_schema`, `token_action_core`, `token_action_analysis.identity/check_outcomes`, `phase3b_distribution.distribution_metrics`, and common IO/integrity.

## Output index

- `audit/`: input audit, worker manifests, observations and action contract.
- `mapping/action_bins.csv`: original token order, bin index, sorted rank, normalized/decoded values, gripper command.
- `data/slot_metrics.csv`: 336,000 observation × temperature × scalar-slot records.
- `data/observation_metrics.csv`, `episode_summary.csv`, `task_summary.csv`, `suite_summary.csv`, `outcome_summary.csv`: separate grouping levels and source strata.
- `data/chunk_observations.csv`, `chunk_summary.csv`: chunk-level comparisons.
- `data/dimension_*`: seven scalar action dimensions.
- `data/gripper_binary.csv`, `gripper_*`: open/close/other mass, binary entropy, threshold mass, native greedy-command versus marginal-modal-command disagreement.
- `data/conditional_summary.csv`: same/different argmax and entropy-decrease subsets.
- `data/correlations.csv`, `quantile_bins.csv`: exact values used in figures, intervals and denominators.
- `data/regression.json`: Stage 1/2/3 numerical agreement and chunk/pooled checks.
- `figures/`: A–G, both temperatures, PNG/PDF; `plotted_values.json` and a consistency-checked seal.
- `cases/`: selection rules, selected-case CSV, actual collapsed-support probabilities, PNG/PDF.
- `summary.md`, `regression_tests.txt`, `manifest.json`, `validation.json`.

## Interpretation contracts

W1 sorts native decoded scalar values, preserving duplicate support mass. It is not joint 3-D optimal transport. Native channel units/scales differ; cross-dimension physical magnitude comparisons are not justified. Normalized-bin W1 is separately provided for common-scale diagnostics.

Group W1 is the mean scalar marginal W1 across its dimensions; greedy action difference is the existing translation/rotation vector L2 (gripper absolute difference). These distinct reductions are labeled and retained separately. `greedy_abs` preserves scalar absolute differences.

Pointwise 95% intervals use 500 whole-task draws within suite, seed 7303. Observations, origins and initial states stay with each task. Fixed JS quantile boundaries use inverse weighted empirical CDF; ties can yield fewer than ten nonempty bins. IQR bands describe within-bin dispersion, not uncertainty in an estimator. Source/outcome and per-chunk correlations are explicitly descriptive without intervals.

Binary gripper entropy is reported only when other mass is exactly zero. Native selected-token command and modal aggregated command need not agree. Threshold proximity uses a declared band of ±0.05 around decoded 0.5, not an assumed native tolerance.

All slots are predicted. Executed action identity and measured EEF movement remain UNKNOWN. Single-slot marginals do not establish cross-slot dependence or a successful-trajectory probability.
