# Phase 3A: nominal paired behavior analysis

This pipeline reads saved episodes on CPU. It does not import a simulator, load
model weights, or run inference/training. Existing rollout files are read-only.
Each stage creates new derived files and rejects an existing destination file;
there is no `--refresh` option. A failed stage may leave partial derived output:
use a new analysis run ID, not deletion of the source rollouts.

## Inputs and definitions

The agreed SFT baseline is `RLinf-OpenVLAOFT-LIBERO-130-Base-Lora` with **no
additional adapter** (`sft_base_only_deterministic`). GRPO also uses no additional
adapter. The input paths below are explicit and never discovered by filename order.

Matching uses suite, task-local ID, task name, BDDL/problem/init-state file
identity from the audited mapping, initial-state index/hash and simulator seed.
Global LIBERO-130 ID and policy seed must agree. A missing or duplicate identity,
wrong model revision, or required alignment failure stops analysis. BDDL identity
is inherited from the mapping; this phase does not launch LIBERO or reauthenticate
the installed BDDL assets. `.partial-*` paths are excluded and reported.

Primary compares SFT deterministic with GRPO deterministic. Secondary compares
GRPO deterministic with stochastic. Additional changes both checkpoint and decoding.
Each has 150 pairs. The known aggregate deltas (66, 16, 82) are consistency gates,
not enough information to infer gained/lost counts individually.

McNemar is the two-sided exact binomial test on discordant pairs (p=1 for none).
Both 95% intervals use percentile bootstrap, 10,000 iterations, seed 3101.
Episode bootstrap samples pairs; task bootstrap resamples whole tasks separately
within each suite, retaining five trials. These are conditional descriptive
sensitivity intervals for fixed benchmark tasks/states, not coverage guarantees
for a randomly sampled task population. P-values are not causal evidence.

Long cases: at most three per category, unused task then unused initial-state
index preferred, SHA256 episode ID tie-break. Every one of the 50 Long cases is
retained. Empty categories remain empty. Trajectory patterns are inspected after
selection; they do not bias the selection rule.

## Reused expert implementation

Read-only source: `experiments/dataset_analysis` in the old `vla-rlvr` repo.
Results: `outputs/stage1_review/dataset_analysis`.

- `analysis.path_statistics`: initial/last XYZ, path length, net displacement.
- `analysis.align_path`: 32 linearly interpolated normalized-index points.
- `task_plots.variation`: all within-task pairwise XYZ RMS and initial-relative spread.
- `stage1.validate` and `semantics.sanity`: existing extracted expert validation.
- `task_plots.generate`: original expert 3D/projection plots, on a new filtered
  copy of its input schema for selected Long tasks. A derived index caption is
  corrected to describe the selected subset. Original code/results are unchanged.
- The recorded robosuite `quat2axisangle` function is extracted from its AST and
  executed with numpy/math only. This reuses the original radians conversion
  without importing robosuite or its simulation dependencies.

Local real-data regression on 2026-11-09 checked all 1,239 extracted expert episodes:
Goal 428 / 52,042 transitions, Spatial 432 / 52,970, Long 379 / 101,469.
Tolerance: `rtol=1e-7`, `atol=1e-9`. The reference CSVs and saved physical metrics
were reproduced. This is not a fresh TFRecord decode or producer authentication.

RLDS contains retained **pre-action** observations and removes no-op transitions.
RLinf contains T+1 observations for T executed actions, including the post-final
state and all policy simulator steps. Both full and pre-action-only policy path
lengths are saved. Equal endpoint definitions do not remove differing sampling
cadence. Normalized-index resampling is not physical-time alignment.

Expert initial-state hashes are not available for these policy trials. Expert
comparisons and cross-reference path distances are explicitly unpaired. Expert
plots remain separate from paired policy overlays. Distance/variation is not
reward, optimality, skill, robustness or behavior quality.

## Coordinate and event evidence

Preflight reads the active research venv's robosuite source for `eef_pos` using
`site_xpos` and `eef_quat` using body world quaternion. Evidence excerpts and hashes
are recorded. If that source is missing, A1/A2 still run, but physical trajectory
plots stop with UNKNOWN instead of silently labeling unverified values as meters.
Units follow simulator source conventions, not an external calibration.

Object distance plots require installed LIBERO object sensor source evidence.
Named object sensor keys are candidates; semantic manipulated-object/target roles
are not guessed from language. Without an explicit role map the object–target
panel displays UNKNOWN. The optional JSON `--object-roles` has exact task-name keys:

```json
{
  "EXACT_TASK_NAME": {
    "object": "EXACT_STORED_OBJECT_NAME_WITHOUT_POS_SUFFIX",
    "target": "EXACT_STORED_TARGET_NAME_WITHOUT_POS_SUFFIX",
    "evidence": "Reviewed BDDL/source path and explanation"
  }
}
```

Use `trajectories/object_role_inventory.json` to review actual available keys.
The canonical orientation is quaternion XYZW (dimensionless), with a separately
saved axis-angle radians conversion. Gripper qpos is labeled native joint units;
executed command is unitless. Numerical command transitions are not authenticated
open/close polarity or successful grasps. No contact/attachment evidence is inferred.
Stationary means step displacement <1e-4 m, a declared descriptive threshold.
No wall-clock timestamps, velocities or failure moments are fabricated.

## Replay bank

Both SFT-origin and GRPO-deterministic-origin observations are included for every
selected case. Each uses initial/20%/50%/85% query-index positions plus queries
around gripper command transitions. Duplicate query IDs are merged with reasons.
Episode coverage is balanced; different event counts can produce different source
query counts, which are reported. Future analysis must stratify/reweight by origin
and episode instead of treating all frames as independent.

Lossless NPZ copies preserve raw camera images, policy images, processed pixel
tensors, proprio, input IDs and attention masks. Array hashes include dtype/shape
and bytes. Raw NPZ hashes, language, task/init-state identity, model config, query
index, simulator step and executed-action range are recorded. Validation compares
each copy against the actual original query and rehashes all source files.
Bfloat16 model input had already been stored as float32 by the collector; the bank
does not recover unstored precision. Teacher-forced action prefixes and free
policy generation are separate future experiments. This stage runs neither.

## Server execution (Bash)

First make the new `scripts/analysis/*.py`, `tests/test_phase3a.py`, and this document
available in the server checkout. Do not replace the server collector with the
older local copy. No changes to upstream or to `vla-rlvr` are needed.

```bash
conda activate base
export RESEARCH_ROOT=/nfs4/jhkim/repos/rlinf
cd "$RESEARCH_ROOT"
source "$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-phase3a-mpl-$USER"
python -c 'import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)'

export EXPERT_ROOT=/nfs4/jhkim/repos/vla-rlvr
export EXPERT_RESULTS="$EXPERT_ROOT/outputs/dataset_analysis"
# Read-only existence checks; if absent, obtain the existing source/archive at these paths.
test -f "$EXPERT_ROOT/experiments/dataset_analysis/analysis.py"
test -f "$EXPERT_RESULTS/trajectory_data/long.json"

export ANALYSIS_OUT="$RESEARCH_ROOT/outputs/analysis/nominal_behavior/phase3a-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python scripts/analysis/nominal_behavior.py --help
PHASE3A_EXPERT_ROOT="$EXPERT_ROOT" python -B -m unittest discover -s tests -p test_phase3a.py -v

# Fresh input audit; hashes all 450 source episodes and inventories actual arrays.
python -B scripts/analysis/nominal_behavior.py preflight \
  --research-root "$RESEARCH_ROOT" \
  --sft "$RESEARCH_ROOT/outputs/nominal/nominal-sft_base_only_deterministic-v1" \
  --grpo "$RESEARCH_ROOT/outputs/nominal/nominal-grpo_deterministic-v1" \
  --stochastic "$RESEARCH_ROOT/outputs/nominal/nominal-grpo_stochastic-v1" \
  --mapping "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" \
  --expert-root "$EXPERT_ROOT" --expert-results "$EXPERT_RESULTS" \
  --seed 3101 --iterations 10000 --output "$ANALYSIS_OUT"
```

Run each next stage only after the preceding command exits successfully. Fixed
input paths and source hashes are loaded from the new analysis manifest.

```bash
python -B scripts/analysis/nominal_behavior.py paired --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py cases --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py trajectories --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py expert --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py replay --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py validate --output "$ANALYSIS_OUT"
python -B scripts/analysis/nominal_behavior.py report --output "$ANALYSIS_OUT"
```

For unattended sequential execution after preflight, this loop stops at the first error:

```bash
for stage in paired cases trajectories expert replay validate report; do
  python -B scripts/analysis/nominal_behavior.py "$stage" --output "$ANALYSIS_OUT" || break
done
```

Inspection:

```bash
cat "$ANALYSIS_OUT/summary.md"
cat "$ANALYSIS_OUT/paired/statistics.json"
cat "$ANALYSIS_OUT/cases/case_selection_manifest.json"
cat "$ANALYSIS_OUT/replay_bank/validation.json"
cat "$ANALYSIS_OUT/completion.json"
find "$ANALYSIS_OUT/trajectories/figures" "$ANALYSIS_OUT/expert_reference/figures" -name '*.png' -print

git status --short
git diff --check
git diff -- scripts/analysis tests/test_phase3a.py docs/PHASE3A.md
# Optional, after transferring/reviewing just these new files:
# git add scripts/analysis/phase3a_core.py scripts/analysis/nominal_behavior.py tests/test_phase3a.py docs/PHASE3A.md
# git commit -m "Add saved-rollout paired behavior analysis"
```

`expert-audit` can run independently when nominal raw files are not on the machine:

```bash
python -B scripts/analysis/nominal_behavior.py expert-audit \
  --expert-root "$EXPERT_ROOT" --expert-results "$EXPERT_RESULTS" \
  --output "$RESEARCH_ROOT/outputs/analysis/nominal_behavior/expert-check-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

## Completion and limitations

`completion.json` exists only after all stages and source preservation validation
succeed. Outputs include schema inventory, all pair IDs, suite/task statistics,
all 50 Long cases, selected figures, canonical measured arrays, expert regression,
and a lossless bank. Individual outputs inherit provenance from manifest.json;
validation.json inventories output checksums, and completion.json binds the final
summary and validation. Actual nominal counts, IDs, plots and bank sizes must be
reported from the server output, never inferred from the three success totals.

Local implementation checks: 15 unit tests passed. The complete CLI pipeline
also passed on a **synthetic** 450-episode fixture, producing 9 policy figures,
8 real-expert reference figures and a 54-observation fixture replay bank.
1,867 input file hashes were unchanged. These fixture counts/outcomes are not
nominal research results. Actual server analysis remains to be run by the user.
Matplotlib was installed only into `/tmp/phase3a-plot-deps` for local visual QA;
no existing project environment was upgraded.
