# LIBERO-PLUS stratified pilot protocol

## Scope and source

The pilot uses the pinned official classification and ordered task map at LIBERO-PLUS commit `4976dc30028e805ff8094b55501d532c48fec182`. `scripts/benchmark/plus_inventory.py` validates the 10,030 variants before sampling. The split is selected from metadata alone; policy results and action-distribution measures never enter the selection algorithm.

The default pilot selects 20 variants in each of 28 suite × category strata. Within each stratum it reserves two variants per original-task name group. The resulting 560 variants cover all 10 original tasks in every stratum. The 1,000 held-out variants are drawn from the remaining population with proportional allocation by stratum and largest-remainder rounding. The split is variant-disjoint but **not** original-task-disjoint. Seed 42 and stable sorted input order make the task content reproducible. `created_at` is excluded from `manifest_content_sha256`.

`difficulty_level` is recorded but never used to exclude a task. The 121 official null values remain unknown. Exact default-design pilot inclusion probability for a variant is `2 / population(original task × suite × category)`. Held-out conditional inclusion probability is `heldout quota / remaining stratum population`; the unconditional probability also accounts for pilot selection. A fallback with original-task quotas too small records null exact probabilities rather than a fabricated estimate. The 560 and 1,000 sizes are design choices, not an optimal sample-size claim.

## Generate and verify manifests (CPU only)

```bash
python3 -m unittest tests.test_plus_inventory tests.test_plus_sampling
python3 -m scripts.benchmark.plus_manifest
```

The command creates `manifests/libero_plus/pilot_v1.json`, `manifests/libero_plus/heldout_v1.json`, and four reports in `outputs/benchmark/libero_plus/sampling/`. It refuses to overwrite existing files. Use `--heldout-size`, `--seed`, `--manifest-dir`, and `--report-dir` when creating a distinct design. Manifest task IDs are official one-based classification IDs; the evaluator converts them to zero-based registry indices.

## Evaluation protocol

The existing PLUS collector default remains `rlinf-520` over all 10,030 tasks. The new pilot requires both `--task-manifest` and explicit `--horizon-protocol lerobot-suite`. Horizon is set separately in both `max_episode_steps` and `max_steps_per_rollout_epoch`: Spatial 280, Object 280, Goal 300, Long 520. `nominal_collect.collect_episode` still performs the same action-chunk inference (8 actions per query), executes individual actions, stops immediately on success, and stops on the suite action-step cap. RLinf's 15 zero-action reset settling steps happen before the collector's episode action count. These horizons match the public LeRobot LIBERO-PLUS guide; other reset and wrapper details may differ from LeRobot.

All four conditions share the same manifest IDs, simulator seed 0, reset path, suite horizon, success signal, and policy seed formula `100000*suite_index+zero_based_task_id`. The condition-specific decoding is `do_sample=False, temperature=1.0` or `do_sample=True, temperature=1.6`, both with `top_k=-1`, `top_p=1.0`. Both published checkpoints are evaluated directly with `is_lora=False`, as the RLinf model card directs. The SFT checkpoint contains an adapter training artifact, but loading it again changes the evaluation policy. Checkpoint identity, file inventory, normalization key, and camera count are checked before a new run. The earlier `plus-pilot-suite-v1` SFT run used `is_lora=True`; preserve it as a separate exploratory run and do not mix its episodes with this corrected protocol.

The environment reset still does **not** apply the saved `.pruned_init` file. The recorded initial-state hash covers selected EEF/gripper observations and camera pixels, not complete simulator state. Run `check_plus_initial_state.py` separately before interpreting four-way paired physical starts. That smoke test requires simulator access and is not run by manifest generation.

On the evaluation server, with the pinned checkout and dependencies available:

```bash
ROOT=/nfs4/jhkim/repos/rlinf
PY="$ROOT/upstream/RLinf/.venv-rlvr-liberoplus-cu121/bin/python"
export PYTHONPATH="$ROOT:$ROOT/upstream/RLinf:${PYTHONPATH:-}"
export EMBODIED_PATH="$ROOT/upstream/RLinf/examples/embodiment"
export LIBERO_TYPE=plus LIBERO_SUFFIX=none MUJOCO_GL=egl
export MAGICK_HOME="$ROOT/conda-envs/libmagick-plus"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2/logs"

nohup "$PY" "$ROOT/scripts/evaluate/plus_collect.py" --research-root "$ROOT" --run-id plus-pilot-suite-nolora-v2 --condition sft_deterministic --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json" --horizon-protocol lerobot-suite > "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2/logs/sft_deterministic.log" 2>&1 &
nohup "$PY" "$ROOT/scripts/evaluate/plus_collect.py" --research-root "$ROOT" --run-id plus-pilot-suite-nolora-v2 --condition sft_stochastic --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json" --horizon-protocol lerobot-suite > "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2/logs/sft_stochastic.log" 2>&1 &
nohup "$PY" "$ROOT/scripts/evaluate/plus_collect.py" --research-root "$ROOT" --run-id plus-pilot-suite-nolora-v2 --condition grpo_deterministic --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json" --horizon-protocol lerobot-suite > "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2/logs/grpo_deterministic.log" 2>&1 &
nohup "$PY" "$ROOT/scripts/evaluate/plus_collect.py" --research-root "$ROOT" --run-id plus-pilot-suite-nolora-v2 --condition grpo_stochastic --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json" --horizon-protocol lerobot-suite > "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2/logs/grpo_stochastic.log" 2>&1 &
```

These are **four example launch commands**, not a recommendation to run them simultaneously on one 24 GiB GPU. Run sequentially or schedule separately; the active full evaluation must finish or remain isolated. To resume a partial condition, repeat its command with `--resume`. A new pilot run rejects an existing condition directory; resume requires the saved protocol hash and selected manifest to match. Pilot runs use a distinct run ID and condition directories.

## Reset reproducibility smoke test (requires explicit simulator authorization)

```bash
"$PY" "$ROOT/scripts/diagnostics/check_plus_initial_state.py" \
  --research-root "$ROOT" \
  --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json" \
  --repeats 3 \
  --output "$ROOT/outputs/benchmark/libero_plus/sampling/initial_state_smoke.json"
```

The script selects one manifest task per suite, repeats the collector's reset path under all four condition labels, compares the worker's flattened simulator state, observable physical fields, and camera hash, then separately applies the saved init state and repeats the 15-step settling phase. The RLinf worker API does not expose qpos and qvel as separate fields, so it reports their combined flattened simulator state instead of claiming separate checks. It does not load a policy or prove four model-loading paths; any inaccessible fields are reported as such. The smoke test has not yet been run.

## Pilot result analysis

After all four conditions complete:

```bash
"$PY" "$ROOT/scripts/evaluate/plus_validate.py" \
  "$ROOT/outputs/libero_plus/plus-pilot-suite-nolora-v2" \
  --task-manifest "$ROOT/manifests/libero_plus/pilot_v1.json"
```

The pilot validator checks every episode against its manifest, condition protocol hash, horizon, checkpoint revision, sampling mode, and observed-start hash. It creates `pilot_validation.json`, `pilot_episodes.csv`, and `pilot_paired.csv` without overwriting existing files. It reports suite, category, and suite × category rates; four-way paired outcomes and SFT/GRPO differences; inverse-inclusion-probability benchmark estimates; and exploratory 95% original-task cluster bootstrap intervals within strata. Only one stochastic policy seed per variant is collected, so stochastic-repeat variance is not measured. With metadata-only rollouts, `action_trajectory_path` is null; trajectory-level links to Stage 1–3 require raw actions to be saved in a later, explicitly scoped run.

The bootstrap is exploratory: each suite × category has only 10 original-task clusters, and variants within an original task can be correlated. Do not interpret its intervals as definitive category-level significance. A matched horizon by itself does not establish full equivalence to LeRobot's evaluation protocol.
