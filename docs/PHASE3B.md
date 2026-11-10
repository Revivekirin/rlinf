# Phase 3B: action distributions, EEF stagnation, stochastic pilot

## Status and scope

The local research checkout is `main`. It has no `upstream/RLinf`, original nominal rollout directories, or CUDA environment. The copied Phase 3A directory is available at `outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246`. Local checks validate all 765 sealed derived files, summary/validation hashes, 450 canonical paths and 260 replay observations (137 SFT-origin, 123 GRPO-origin). They do **not** reauthenticate remote raw arrays or checkpoints. The server `audit` stage does that afresh, including all raw episode validation, original source hashes and checkpoint LFS/Git-blob hashes. GPU status is recorded without stopping or changing any process. A later SFT base-only stochastic run was separately validated by the user at 150/150 episodes: Goal 35/50, Spatial 35/50, Long 24/50, total 94/150, with `do_sample=True`, T=1.6, no second adapter, revision `a4ea91b3897648f3c1395e5221edb40b8f53d1a4`, collector SHA256 `cbd26f446dd5dc19db5e737b618ac8787563b2292a588dff98c4be6cdae6d7f0`. Its paired outcome/stagnation results are not yet available locally.

The pinned reference source was retrieved from:

`https://github.com/RLinf/RLinf/blob/e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d/rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`

Its exact SHA256 is `22a5b7864bc125bb8ca757b1ed14e901972eea96ab243e9b5d663f61b3f219e9`. This is reference-source inspection, **not** a claim that the remote checkout was accessed. Server audit checks the live file against this digest and requires a clean upstream at `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d`.

### Native policy contract

`OpenVLAOFTForRLActionPrediction.predict_action_batch` does one language-model forward for 8×7 action slots. `_prepare_input_for_action_prediction` appends placeholders; `_build_embedding` zeros the action embeddings. Action tokens are chosen in parallel with argmax or `torch.multinomial`. There is **no autoregressive action prefix** in this path. Common-prefix teacher forcing is therefore marked `NOT_APPLICABLE_NATIVE_PARALLEL`; no attention/caching or prefix implementation is substituted.

A read-only `language_model.register_forward_hook` copies the native action-position logits. Native masking admits token IDs `[vocab_size - n_action_bins, vocab_size)`. KL/JS use the entire admitted support in float64 log space, not top-k approximations. Metrics at T=1 and T=1.6 are entropy/top-1/top-5/top-10 mass, greedy disagreement, forward/reverse KL and JS, by all 56 positions. T=1 categorical entropy is **not** entropy of the deterministic argmax rollout policy. The native token-to-bin clipping map is recorded; token entropy is not equivalent to useful continuous-action diversity.

Each model runs in a separate process, sequentially on the chosen GPU, with `eval()`, disabled parameter gradients and `torch.inference_mode()`. The native `predict_action_batch(env_obs=...)` path receives the exact saved policy image/proprio arrays and language. Its own processor and normalization are retained. Processed tensor hashes are compared with the saved inputs and across models; differences are explicit. The smoke compares hooked and unhooked native greedy and stochastic actions, action tokens and processed inputs at the same seed. Full workers require that completed smoke.

SFT is the previously validated Base-Lora checkpoint at `a4ea91b3897648f3c1395e5221edb40b8f53d1a4`, **without a second adapter application**. `is_lora=False`, `lora_path=None`. GRPO revision is `473954241c8148bc9a1bf9ed547a74523e4a7d41`. This preserves the agreed SFT baseline despite the request's shorthand “Base + learned SFT LoRA”.

Reused local entry points: `scripts/evaluate/nominal_collect.py:collect_episode`, `nominal_validate.py:validate_episode`, `nominal_storage.py` episode format; `scripts/analysis/phase3a_core.py` hashing and legacy loader. Pilot diversity reuses `experiments/dataset_analysis/analysis.py:path_statistics` and `task_plots.py:variation` from the configured, unmodified `vla-rlvr` checkout.

## Stagnation

CPU-only. The primary definition is Euclidean displacement **< 0.0001 m per executed simulator step** for **>= 50%** of steps. T+1 measured EEF states define T displacements. No wall-clock sampling rate is inferred. The older ad-hoc comparison used fraction **> 50%**; both classifications and equality cases are recorded.

The same invocation computes the 0.05/0.1/0.2/0.5 mm × 25/50/75% sensitivity grid, all 450 per-episode metrics, primary and secondary 150-pair changes, task/suite/outcome tables, path/episode/step displacement distributions, and stationary run lengths. `--raw` additionally requires the original states/actions/observations/metadata to match Phase 3A hashes, and computes native translation command norms, command-motion correlation, zero/nonzero command counts during stationary steps, finger qpos motion and gripper transitions. Commands are not assumed to be world-space meters. Object failure labels stay UNKNOWN. These diagnostics do not authenticate saturation/contact/scaling or classify all stationary intervals as failures.

Local real-data result: SFT 61/150, GRPO deterministic 8/150, GRPO stochastic 0/150. Primary changes: 56 lose the stagnation classification, 3 gain it, 5 retain it. Gained subset: SFT 46/67, GRPO 0/67. Exactly-half boundary episodes: 0. Command-motion analysis on real raw data remains pending server execution.

The optional CPU-only `sft-sampling` stage validates the new fourth run afresh, checks its protocol against the same-checkpoint SFT deterministic baseline, pairs all 150 identical task/initial-state IDs, and reports gained/lost outcomes, EEF stagnation, native command magnitudes, and threshold sensitivity. The observed 94/150 versus 64/150 is a 30-episode net gain; the gained and lost counts require this paired stage. The existing Phase 3B primary and secondary comparisons retain their original definitions.

## Pilot A

Zero-based Long tasks 8 and 3; initial states 0–4, matched against task identity/BDDL and initial-state hashes. Seeds are:

`200000 + 100*task_id + trial_id + 1000000*replicate`, replicate 1–5.

These 50 seeds do not overlap the existing nominal seed for each state. The first two jobs are `t08-i00`, seeds 1200800 and 2200800. They constitute the bounded smoke and are included in Pilot A, so the full budget is **2 smoke + 48 additional = 50**. Pilot B is not implemented or launched.

The collector is imported and its **unchanged** `collect_episode` function performs resets, native inference, action postprocessing, terminal checks and valid-prefix step counting. No original collector or upstream file is patched. Each new episode has its own run/manifest, full config hash, raw token/action/state/observation files and checksums. A completed smoke is referenced read-only with `reused_smoke` provenance. Existing output directories are rejected unless the explicit Python `run --resume` option is used; incomplete job directories are never silently overwritten.

Environment config seed is fixed at 0 and task/init-state selection is explicit. The inherited collector seeds Python/NumPy/torch/CUDA with policy seed **after reset**. This does not establish full independence of environment CPU RNG and policy RNG. Therefore results are a sampling-seed intervention under this protocol, not pure policy sampling variance. Validation requires byte-identical initial observations and initial recorded states across the two smoke seeds, agreement with the original nominal first observation, distinct sampled token sequences, full saved-array structure, expected identities/seeds and hashes. The unchanged collector function's source digest is stored. A new end-to-end comparison against a separately launched official evaluator is not claimed.

Diversity uses original 32-point normalized-index world RMS per same state. Successful/failed/mixed pairs are reported separately; success-conditioned pairwise results are UNDEFINED for <=1 success. Endpoint position/quaternion angles, gripper transitions, path length variation and common-query-ordinal token/action differences are descriptive. Equal query indices across diverged trajectories do not imply equal observations or semantic events. Paths with different shapes do not establish multiple useful strategies.

## Deployment and exact server commands

Transfer `phase3b-code-v2.tar.gz` from the local research repository's `outputs/analysis/` into `/nfs4/jhkim/repos/rlinf/` using your normal transfer method. The archive is not automatically present on server4. It contains only the new Phase 3B scripts, tests and this document. Do not overwrite the nominal collector or Phase 3A code. If you already extracted the older `phase3b-code.tar.gz`, stop before extraction: `tar -xkzf` deliberately refuses to replace those files. Use a clean checkout or review the v2 diff and copy only the changed Phase 3B files.

```bash
cd /nfs4/jhkim/repos/rlinf
# Only after the archive has actually been transferred:
tar -xkzf phase3b-code-v2.tar.gz
conda activate base
source "$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
export RESEARCH_ROOT="$PWD"
export PHASE3A="$RESEARCH_ROOT/outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246"
export PHASE3B_RUN="phase3b-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-phase3b-mpl-$USER"
echo "PHASE3B_RUN=$PHASE3B_RUN"
python -B -m unittest discover -s tests -p test_phase3b.py -v
bash -n scripts/analysis/run_phase3b.sh
```

Keep the same `PHASE3B_RUN` for all following stages. Run each only after the prior command succeeds. Do not run an all-stage script: none is provided. The SFT stochastic run completed before Phase 3B deployment; check current GPU occupancy before the GPU stages.

```bash
# 2. Current upstream/checkpoints/raw inputs/replay/GPU audit (checkpoint hashing reads ~30 GB).
bash scripts/analysis/run_phase3b.sh audit
# Supplementary fourth-condition comparison, 150 SFT deterministic/stochastic pairs.
bash scripts/analysis/run_phase3b.sh sft-sampling
# 3. All 150 primary + 150 secondary pairs, with raw commanded/actual motion.
bash scripts/analysis/run_phase3b.sh stagnation
# 4. Display the sensitivity grid already computed by the same analysis.
bash scripts/analysis/run_phase3b.sh sensitivity
# 5. Fresh original-observation reference/readback and native source preflight.
bash scripts/analysis/run_phase3b.sh replay-preflight
# 6. One identical observation; separate SFT/GRPO workers; hooked/unhooked comparisons.
bash scripts/analysis/run_phase3b.sh model-smoke
# 7. Full 260-observation bank; explicit user launch after model smoke PASS.
bash scripts/analysis/run_phase3b.sh distributions
# 8. Full-support KL/JS, token/action tables, grouped summaries and plots.
bash scripts/analysis/run_phase3b.sh distribution-summary
# 9. Verify task/state/seed grid, then collect only the two bounded smoke episodes.
bash scripts/analysis/run_phase3b.sh pilot-plan
bash scripts/analysis/run_phase3b.sh pilot-smoke
# 10. Emits the exact 50-episode Pilot A command ONLY after smoke PASS.
# It does not launch Pilot A. Run the printed command manually.
bash scripts/analysis/run_phase3b.sh pilot-command
# 11. After the manually launched Pilot A finishes:
bash scripts/analysis/run_phase3b.sh pilot-validate
# 12. Same-state diversity, including UNDEFINED success-conditioned pairs.
bash scripts/analysis/run_phase3b.sh diversity
# 13. Report from validated outputs.
bash scripts/analysis/run_phase3b.sh report
# 14. Review only the new implementation; commit is optional.
git status --short
git diff --check
git diff --stat
# git add scripts/analysis/phase3b.py scripts/analysis/phase3b_distribution.py scripts/analysis/phase3b_pilot.py scripts/analysis/phase3b_sft_sampling.py scripts/analysis/run_phase3b.sh tests/test_phase3b.py docs/PHASE3B.md
# git commit -m "Add native policy distribution and bounded stochastic pilot analyses"
```

For an unattended full inference stage after successful model smoke:

```bash
mkdir -p "$RESEARCH_ROOT/logs"
# Choose a fresh log filename; shell noclobber prevents replacing an existing log.
( set -C; nohup bash scripts/analysis/run_phase3b.sh distributions > "$RESEARCH_ROOT/logs/${PHASE3B_RUN}-distributions.log" 2>&1 < /dev/null & )
```

The direct Python CLIs (`--help` validated) are available for restarting a **missing** worker or resuming an interrupted pilot. Completed files are never overwritten. Example worker recovery: invoke `phase3b_distribution.py worker --model grpo` with the same preflight/output/smoke-result paths, rather than rerunning the SFT worker. Partial worker directories require a new output run; partial pilot job directories also require a new run. `run --resume` reuses only completed, hash-validated pilot entries.

Outputs are under `outputs/analysis/{phase3b_audit,stagnation,sft_sampling,policy_distribution,stochastic_pilot,stochastic_diversity,phase3b_report}/`. The SFT stochastic run was validated by the user on server4, but its new paired comparison has **NOT RUN locally**. GPU distribution results and actual pilot smoke are **NOT RUN locally**. CPU unit tests and synthetic fixtures do not substitute for GPU smoke.
