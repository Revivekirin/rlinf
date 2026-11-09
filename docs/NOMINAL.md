# LIBERO-130 nominal rollout collection

This is an evaluation-only data collection protocol for the pinned RLinf
checkout `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d`. It uses the
LIBERO-130 SFT checkpoint, with the adapter-loading choice recorded explicitly,
and the separate GRPO final policy.
The same policy is evaluated on the Goal (`libero_goal`), Spatial
(`libero_spatial`), and Long (`libero_10`) subsets. The mapping audit checks
the exact task identity and five initial-state byte hashes against
`libero_130`; it does not infer index offsets.

The existing one-episode SFT smoke in `manifests/libero_130_smoke.json` was
**stochastic**, despite the CLI setting `do_sample=false`: the pinned RLinf
eval worker derives sampling from `temperature_eval > 0`. Its success/failure
must not be pooled with the deterministic conditions below. This collector
passes `do_sample` and temperature directly to the native model. In the
pinned model, `top_k` and temperature affect stochastic sampling; `top_p` is
passed but is not used by `predict_action_batch`.

## Server preflight and mapping audit

Run each block on `server4`. The research repo's local changes must first be
committed and made available to the server checkout. Stop if any command fails.
The server's `conda info --envs` listing contains no RLinf-specific conda
environment. Activate conda `base` as the host shell, then activate the
RLinf project venv. Do not use either `tdmpc2-ms2` or the nonexistent
`rlinf-libero-cu121-v2` as the conda environment for this protocol. The
interpreter path below selects the actual RLinf Python packages. Check CUDA
with that interpreter before the next smoke run.

```bash
conda activate base
export RESEARCH_ROOT=/nfs4/jhkim/repos/rlinf
cd "$RESEARCH_ROOT"
git pull --ff-only origin main
source "$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
mkdir -p "$RESEARCH_ROOT/logs"
git status --short
git rev-parse HEAD
git -C upstream/RLinf rev-parse HEAD
git -C upstream/RLinf status --short
export EMBODIED_PATH="$RESEARCH_ROOT/upstream/RLinf/examples/embodiment"
export PYTHONPATH="$RESEARCH_ROOT:$RESEARCH_ROOT/upstream/RLinf:${PYTHONPATH:-}"
export LIBERO_TYPE=standard
VENV_PY="$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
"$VENV_PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvidia-smi
df -h "$RESEARCH_ROOT"
```

The process requires an idle GPU and at least 10 GiB free in the collector's
preflight. The 450-episode output may require substantially more than this;
check disk usage during execution. Do not stop other users' Ray processes.

```bash
"$VENV_PY" "$RESEARCH_ROOT/scripts/diagnostics/audit_libero_mapping.py" \
  --output "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" --trials 5
"$VENV_PY" -m json.tool "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" >/dev/null
```

If mapping fails, stop. Do not guess task IDs. `manifests/local/` is ignored
by Git because this is a machine-specific audit result.

## Bounded smoke

Each invocation runs one initial state of the first task in each of the three
subsets: three episodes per condition. Each run has a unique ID. An existing
run is never overwritten.

```bash
cd "$RESEARCH_ROOT"
for condition in sft_deterministic grpo_deterministic grpo_stochastic; do
  "$VENV_PY" scripts/evaluate/nominal_collect.py \
    --research-root "$RESEARCH_ROOT" \
    --mapping "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" \
    --run-id "smoke-${condition}-v1" --condition "$condition" --smoke \
    2>&1 | tee "$RESEARCH_ROOT/logs/nominal-smoke-${condition}.log"
  test "${PIPESTATUS[0]}" -eq 0 || break
done
```

The loop is Bash, not zsh. Inspect all three `summary.json` files before
expanding. Confirm completed 3/3, no missing episodes, task IDs and init-state
hashes, 8-token chunks, real EEF pose, aligned state/action lengths, and image
reload. A failed task outcome is valid data; a collector exception is not.
Run the structural validator on every smoke run:

```bash
for condition in sft_deterministic grpo_deterministic grpo_stochastic; do
  "$VENV_PY" "$RESEARCH_ROOT/scripts/evaluate/nominal_validate.py" \
    "$RESEARCH_ROOT/outputs/nominal/smoke-${condition}-v1" || break
done
```

On 2026-11-08 the user ran and structurally validated all three bounded GPU
smokes in the RLinf venv. Each completed 3/3 episodes with no missing IDs:

| Condition | Goal | Spatial | Long | Total |
| --- | ---: | ---: | ---: | ---: |
| SFT deterministic | 0/1 | 0/1 | 0/1 | 0/3 |
| GRPO deterministic | 1/1 | 1/1 | 0/1 | 2/3 |
| GRPO stochastic, temperature 1.6 | 1/1 | 1/1 | 1/1 | 3/3 |

On 2026-11-09 the full `sft_deterministic` collection completed 150/150
episodes, all failures at 520 steps (Goal, Spatial, Long: 0/50 each). This is
**the SFT checkpoint with the additional LoRA adapter applied**, not a
validated reproduction of the published SFT baseline. The published unified
SFT model reports 49.40% Goal, 51.61% Spatial, and 11.90% Long across 500
trials per suite ([RLinf benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)).
The difference requires investigation before using this collection as a
baseline.

The adapter is active: inference on the stored first observation reproduces
the recorded 56 action tokens exactly; disabling it changes 9 tokens. In
official RLinf evaluation, Goal task 0/trial 0 fails at 520 steps both with
and without the adapter. Goal task 0/trial 25 fails at 520 with the adapter
but succeeds at 128 with the base weights alone. This paired result makes
adapter loading a stronger lead than a collector-only success-count bug, but
does not establish the correct loading mode for every task. The pinned
upstream `libero_130_grpo_openvlaoft.yaml` points at the `Base-Lora`
checkpoint while inheriting `is_lora=false` from `model/openvla_oft.yaml`.

The GRPO deterministic smoke was replayed under conda `base` plus the RLinf
venv. For all three episodes, metadata (except run ID) and every array in
`actions.npz`, `states.npz`, and `observations.npz` matched exactly. This
checks fixed-state reproducibility, not performance or independently
instrumented action equivalence to the official worker.

An uninstrumented official RLinf `libero_goal` evaluation of the GRPO model
with the same task/trial (0/0), seed, normalization, and deterministic
sampling reported success and `episode_len=104`. The collector recorded the
first terminal action at step 101. The pinned upstream `LiberoEnv.chunk_step`
executes all eight slots even after termination; its `episode_len` metric is
read from the final slot's elapsed step. Thus the official reported length
is `ceil(101 / 8) * 8 = 104`. The official worker does not persist per-step
actions, so exact action equality between the two implementations has not
been measured. Full runs must preserve this distinction between valid-prefix
length and upstream chunk-end length.

## Full collection after smoke validation

The historical command below produced the adapter-on SFT 0/150 run. Do not
interpret it as a validated SFT baseline. The collector now also accepts
`sft_base_only_deterministic`; use a new run ID for that loading control and
validate its summary before proceeding to full GRPO comparisons.

Run the base-only control under a new ID; `sft_deterministic` retains its
historical adapter-on meaning for provenance. The initial three-episode
smoke may still all fail, so inspect a broader subset before deciding whether
the published result has been reproduced.

```bash
conda activate base
export RESEARCH_ROOT=/nfs4/jhkim/repos/rlinf
cd "$RESEARCH_ROOT"
git pull --ff-only origin main
source "$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
export EMBODIED_PATH="$RESEARCH_ROOT/upstream/RLinf/examples/embodiment"
export PYTHONPATH="$RESEARCH_ROOT:$RESEARCH_ROOT/upstream/RLinf:${PYTHONPATH:-}"
export LIBERO_TYPE=standard
python scripts/evaluate/nominal_collect.py \
  --research-root "$RESEARCH_ROOT" \
  --mapping "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" \
  --run-id nominal-sft_base_only_deterministic-v1 \
  --condition sft_base_only_deterministic
python scripts/evaluate/nominal_validate.py \
  "$RESEARCH_ROOT/outputs/nominal/nominal-sft_base_only_deterministic-v1"
```

Each command targets exactly 10 tasks × 5 audited initial states per suite,
or 150 episodes per condition. The required three conditions total 450
episodes. The optional fourth `sft_stochastic` would add 150. Run serially on
the single A5000.

```bash
cd "$RESEARCH_ROOT"
for condition in sft_deterministic grpo_deterministic grpo_stochastic; do
  "$VENV_PY" scripts/evaluate/nominal_collect.py \
    --research-root "$RESEARCH_ROOT" \
    --mapping "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" \
    --run-id "nominal-${condition}-v1" --condition "$condition" \
    2>&1 | tee "$RESEARCH_ROOT/logs/nominal-${condition}.log"
  test "${PIPESTATUS[0]}" -eq 0 || break
done
```

The collector never invokes the RLinf training runner or any optimizer. It
loads each policy once and runs one simulator process at a time. `--smoke`
changes the expected episode set and therefore cannot be used to resume a
full run.

## Status, resume, and validation

```bash
ps -u "$USER" -o pid,etime,stat,cmd | grep -E '[n]ominal_collect.py|[O]ffScreenRenderEnv'
tail -80 "$RESEARCH_ROOT/logs/nominal-grpo_stochastic.log"
du -sh "$RESEARCH_ROOT/outputs/nominal"/*
```

Resume only the same run ID, condition, mapping, code version, and smoke/full
mode. A partial directory stays hidden from the summary. An existing complete
episode is skipped; an incomplete visible episode is an error that needs
inspection.

```bash
"$VENV_PY" "$RESEARCH_ROOT/scripts/evaluate/nominal_collect.py" \
  --research-root "$RESEARCH_ROOT" \
  --mapping "$RESEARCH_ROOT/manifests/local/libero_130_mapping.json" \
  --run-id nominal-grpo_stochastic-v1 --condition grpo_stochastic --resume
```

For each full run, inspect `outputs/nominal/<run-id>/summary.json`,
`episodes.csv`, and `manifest.json`. The summary is rebuilt after every
episode from validated directories, so it records the count, success rate,
mean length, missing IDs, per-suite counts, and total stored bytes. The
success rates are empirical for the selected 5 states per task, not the
published 130-task benchmark. Never combine deterministic with stochastic
successes.

```bash
for condition in sft_deterministic grpo_deterministic grpo_stochastic; do
  "$VENV_PY" "$RESEARCH_ROOT/scripts/evaluate/nominal_validate.py" \
    "$RESEARCH_ROOT/outputs/nominal/nominal-${condition}-v1" || break
done
```

## Data interpretation

`states.npz` contains one simulator observation at reset plus one after each
executed action: `T+1` states for `T` actions. `robot0_eef_pos` is the
simulator's world-frame XYZ; `robot0_eef_quat` is its raw quaternion. Other
numeric simulator observation keys, including joints and available object
states, are kept by name. The position unit is the simulator's world unit
(meters in LIBERO/robosuite); do not assume it matches an external expert
dataset frame. Settling actions during the standard 15-step reset are not
counted as policy actions.

`actions.npz` contains every eight-slot decoded chunk and raw token chunk,
plus the actions actually sent to the simulator. The latter have RLinf's
LIBERO gripper postprocessing. `executed_query_index`,
`executed_action_slot`, and `simulator_step` align each executed action to a
query and the `T+1` states. If success or horizon occurs mid-chunk, remaining
slots are *not* recorded as executed. This differs from the upstream
`chunk_step` implementation, which still calls the simulator on later slots
after an already-done signal and masks them for metrics. This collector stops
at the first terminal action, preserving the same valid prefix and official
`success_once` judgment. Before interpreting rates as identical to the
official evaluator, compare paired bounded runs and investigate differences.

`observations.npz` contains query-step indices, raw main-camera image,
180-degree rotated policy main image, proprio input, processor pixel tensor,
input IDs, and attention mask. The wrist image is stored only if the loaded
model actually uses more than one image; the pinned LIBERO-130 model uses one.
The language string is in `metadata.json`. NPZ uses lossless compression; the
processor pixel values are saved as float32 if the in-memory type is bfloat16.
Full logits and hidden states are intentionally outside this phase.

The collector aborts on source/checkpoint mismatch, missing normalization
statistics, mapping mismatch, invalid EEF/action values, or state/action
alignment failure. Files are written into a `.partial-*` directory and
promoted to a complete episode directory only after they reload successfully.
Remote GPU smoke remains the gate before the 450-episode run.
