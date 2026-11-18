# Stage 4 / R3 — outcome-conditioned trajectory diversity

This is a CPU-only, read-only analysis of the completed repeated-rollout run. It does not launch LIBERO, load checkpoints, or perform inference. The source run and its existing diversity directory remain untouched. A new output directory is required.

## Inputs and audit

- Run: `outputs/analysis/repeated_rollouts/r2a-20261117T130653Z`
- Existing summary: `outputs/analysis/repeated_rollouts/r2a-20261117T130653Z-diversity`
- The CLI verifies the run plan seal, the original diversity seal and all its recorded source hashes, then invokes `repeat_validate_state` for every one of the 100 jobs.
- The original CSV schemas are checked at runtime. The required pair columns are `episode_id,pair_type,job_a,job_b,success_a,success_b,trajectory_rms_32_m`. The required episode columns include `job_id,episode_id,policy,task_id,trial_id,policy_seed,success,steps,initial_state_sha256`. The old state summary supplies one row per initial state and the SFT, GRPO and cross-policy RMS means.
- For all 450 pairs the original `legacy task_plots.variation` is rerun on the original `robot0_eef_pos` arrays. Each recomputed value must agree with the saved CSV. The 100/100/250 pair split and published six-digit RMS means are regression gates.
- Within-policy pairs are classified as SS/SF/FF from `metadata.json` success flags. Each category's count and the weighted decomposition identity are checked per state. Undefined categories are empty CSV cells with `UNDEFINED` status, never zero-valued means.

## Server commands

Run from `/nfs4/jhkim/repos/rlinf` after copying the R3 code archive there:

```bash
cd /nfs4/jhkim/repos/rlinf
tar -xzf r3-code-20261118.tar.gz
source upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate
export PY="$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-r3-mpl"
export RUN="$PWD/outputs/analysis/repeated_rollouts/r2a-20261117T130653Z"
export DIVERSITY="$PWD/outputs/analysis/repeated_rollouts/r2a-20261117T130653Z-diversity"
export R3="$PWD/outputs/analysis/outcome_diversity/r3-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -B scripts/analysis/outcome_diversity.py audit --run "$RUN" --diversity "$DIVERSITY"
"$PY" -B scripts/analysis/outcome_diversity.py run --run "$RUN" --diversity "$DIVERSITY" --output "$R3"
```

Regenerate all figures from the same saved CSVs and original EEF arrays:

```bash
"$PY" -B scripts/analysis/outcome_diversity.py figures --output "$R3"
```

Check the saved seal, numerical decomposition, source hashes and figure count:

```bash
"$PY" -B scripts/analysis/outcome_diversity.py validate --output "$R3"
```

Read the report:

```bash
"$PY" -B scripts/analysis/outcome_diversity.py summary --output "$R3"
```

Run all CPU tests, including R1/R2 regression tests:

```bash
"$PY" -B -m unittest discover -s tests -q
```

## Outputs and reading

The output contains a manifest, validation seal, labeled 450-pair CSV, 20 state/policy outcome rows, 20 decomposition rows, 10 paired-state rows, task and policy summaries, and `summary.md`. The `figures/` directory contains five main figure types plus the summary dashboard and all 10 EEF overlays, each in PNG and PDF (30 files total). Representative overlays are `overlay_libero_10-t03-i00`, `t03-i03`, `t08-i00` and `t08-i04`.

The analysis unit for policy comparison is the matched initial state. Category means are defined only where category pairs exist. SFT and GRPO SS means over all available states have different coverage; `summary.md` also reports the subset where both are defined. The 450 pairs are dependent because each rollout appears in several pairs. Only two selected tasks and five stochastic seeds per policy/state were studied. The XYZ overlays show observed paths in world coordinates with identical axis limits for the two policies within each state. A different normalized path index does not imply a matching physical event. The figures do not establish causal effects or separate learned strategies.
