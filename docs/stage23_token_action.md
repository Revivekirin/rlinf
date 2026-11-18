# Stage 2/3: saved-token distribution and action analysis

CPU-only CLI: `scripts/analysis/token_action_analysis.py`.
Existing Stage 1/4 outputs, model checkpoints, collector schemas and CLI behavior are unchanged.
The implementation imports the existing distribution metrics and task-cluster bootstrap directly.

## Result

Validated local output: `outputs/analysis/token_action/stage23-20261118-final/`.
Read `report.md`, `audit/source_audit.md`, `audit/contract.json`, and `validation.json`.
Figures are under `stage2/figures/` and `stage3/figures/`, each in PNG and PDF.
Detailed CSV/JSON/NPZ files are under each stage's `data/` directory.

- Native 8×7 chunk-major mapping, token/bin aliasing, and Q99 unnormalization are checked against all saved records.
- Stage 1 observation and source-episode metrics are reproduced before interpreting the dimension breakdown.
- Temperature changes probabilities only. Stored greedy decoded actions are identical across T=1 and T=1.6 analyses.
- Translation/rotation values are controller commands, not measured meters/radians or EEF movement.
- Execution is UNKNOWN without a verified nominal query/slot linkage. S2-5 is NOT AVAILABLE.
- This local Stage 1 copy is missing 12 original figure files. Bank/worker seals and every required numerical input pass; the full copied analysis seal is not claimed to pass.
- New output creation is exclusive. Use a fresh `OUT` for another run. Do not overwrite sealed outputs.

## Reproduce on the server

Run from `/nfs4/jhkim/repos/rlinf`. These are the implemented commands.
The source files for the new CLI must be transferred first. The pinned upstream source tree supplies the native source audit.

```bash
set -e
cd /nfs4/jhkim/repos/rlinf
source upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate
export PY="$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-stage23-mpl-$USER"
export S1="$PWD/outputs/analysis/distribution_generality/stage1-20261118T035126Z-2960548"
export S4="$PWD/outputs/analysis/outcome_diversity/r3-20261118T053555Z"
export OUT="$PWD/outputs/analysis/token_action/stage23-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export TESTLOG="${TMPDIR:-/tmp}/rlinf-stage23-tests-$$.txt"

# 1. Input validation + native source audit
"$PY" -B scripts/analysis/token_action_analysis.py audit --stage1 "$S1" --stage4 "$S4" --output "$OUT"
# 2. Stage 2 analysis
"$PY" -B scripts/analysis/token_action_analysis.py stage2 --output "$OUT"
# 3. Stage 2 plots
"$PY" -B scripts/analysis/token_action_analysis.py plot-stage2 --output "$OUT"
# 4. Stage 3 analysis
"$PY" -B scripts/analysis/token_action_analysis.py stage3 --output "$OUT" --near-zero 0.01
# 5. Stage 3 plots
"$PY" -B scripts/analysis/token_action_analysis.py plot-stage3 --output "$OUT"
# 8. Regression tests (run before report so the log is sealed with the result)
"$PY" -B -m unittest discover -s tests -p 'test_*.py' > "$TESTLOG" 2>&1
# 6. Combined report + root output seal
"$PY" -B scripts/analysis/token_action_analysis.py report --output "$OUT" --regression-log "$TESTLOG"
# 7. Verify inputs and all new output seals
"$PY" -B scripts/analysis/token_action_analysis.py verify --output "$OUT"
cat "$OUT/report.md"
```

Execute in a shell with `set -e` (or stop immediately on failure). `report` rejects a failed regression log.
If `--stage1`/`--stage4` are omitted, audit searches eligible verified results; a lexically newest incomplete run is not accepted.
The optional legacy expert test remains skipped unless `PHASE3A_EXPERT_ROOT` points to its real legacy source checkout.

## Reproduce locally without an upstream checkout

Use `/Users/jihyekim/miniconda3/envs/basilisk-rl/bin/python`, repository-relative local Stage 1/4 paths, and a new output directory.
For audit, append:

```bash
--native-source-dir outputs/analysis/token_action/stage23-20261118-final/audit/native_sources
```

These copied source snapshots are checked against the pinned SHA256 values; they are evidence, not imported runtime code.
Use the remaining commands exactly as above with the local `PY`, `S1`, `S4`, and `OUT` variables.

## Tables and statistical interpretation

`stage2/data/slot_metrics.csv` holds all 336,000 temperature × observation × slot rows.
`stage3/data/slot_action_relations.csv` links each slot's ΔH/JS to token distance, bin distance, normalized difference and decoded difference.
`observation_action_metrics.csv` stores the 24,000 observation × chunk vector comparisons.
`episode_metrics.csv` reduces observations before comparisons; `initial_state_metrics.csv` combines paired source origins by initial-state identity.
`action_group_summary.csv`, `task_summary.csv`, `suite_summary.csv`, and `outcome_summary.csv` include separate SFT-origin, GRPO-origin, and combined strata.

Main intervals resample whole tasks within each suite, preserving observations, initial states and origins in each task. Seed=3101, 2000 draws.
Observation point clouds are descriptive; they do not treat observations or slots as independent trials.
Near-zero cosines are missing/undefined, not zero. Their conditional mean must be read with the defined fraction and sensitivity table.
Gripper command confidence is the sum of token probabilities mapping to the same prepared command.
The lost-outcome stratum is very small; an entropy change alone does not establish a success benefit.
