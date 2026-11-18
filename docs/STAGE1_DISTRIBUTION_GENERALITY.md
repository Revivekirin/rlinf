# Stage 1: common-observation distribution generality

This stage prepares a primary bank from all 150 deterministic SFT and 150 deterministic GRPO nominal episodes across Goal, Spatial and Long. It selects at most ten uniform query observations per source episode. Every selected observation is replayed independently through both pinned models. The old 18-episode / 260-observation selected Long bank and outputs are read only. GPU full replay is **manual**; the code below does not start it on its own.

The CPU bank preflight checks all four nominal run directories and their protocol, 150 completed episode directories per condition, query counts, the 30-task mapping, deterministic paired outcomes and state hashes. The fourth SFT stochastic condition is inventoried; a secondary bank can later be made with `--origins sft_stochastic grpo_stochastic` in a separate run. Primary and secondary origins cannot be mixed in one bank.

The local archive `outputs/analysis/stage1-code-20261118.tar.gz` contains `phase3b_distribution.py`, `phase3b_generality.py`, the full-run shell script, the CPU test and this runbook. Transfer it to the server repository root, then inspect and extract it. Server HEAD `bdd07c9` is compatible; check any source differences before extraction.

```bash
cd /nfs4/jhkim/repos/rlinf
git status --short
tar -tzf stage1-code-20261118.tar.gz
tar -xzf stage1-code-20261118.tar.gz
git diff --check
git diff --stat
source upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate
export PY="$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
export PYTHONPATH="$PWD:$PWD/upstream/RLinf:${PYTHONPATH:-}"
export EMBODIED_PATH="$PWD/upstream/RLinf/examples/embodiment"
export LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-stage1-mpl-$USER"
export PHASE3A="$PWD/outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246"
export AUDIT="$PWD/outputs/analysis/phase3b_audit/phase3b-20261117T111837Z-2822692"
export RUN="$PWD/outputs/analysis/distribution_generality/stage1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -B -m unittest discover -s tests -p 'test_phase3b_generality.py' -q
"$PY" -B -m unittest discover -s tests -p 'test_phase3b.py' -q
bash -n scripts/analysis/run_stage1_full.sh
df -h "$PWD/outputs"
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
```

## A. CPU preflight and bank

```bash
"$PY" -B scripts/analysis/phase3b_generality.py bank \
  --phase3a "$PHASE3A" --audit "$AUDIT" --research-root "$PWD" \
  --origins sft grpo --max-queries 10 --output "$RUN"
"$PY" -B scripts/analysis/phase3b_generality.py bank-validate --run "$RUN"
"$PY" -B -c 'import json,sys; m=json.load(open(sys.argv[1])); print("episodes",len(m["source_episodes"]),"observations",len(m["observations"]),"inventory",m["nominal_inventory"])' "$RUN/bank/manifest.json"
```

Expected: 300 source episodes; at most 3,000 distinct observations. The actual observation count depends on short successful episodes. The bank records raw and copied file hashes, array hashes, query indices, outcome labels and source policy. It checks selected arrays against each original `observations.npz`. `bank-validate` rereads original observations. The bank seal hashes every derived file.

## B. Bounded GPU smoke, five observations per model

Run the models sequentially, as separate processes. Smoke writes under `RUN/smoke/` and does not count toward full worker completion.

```bash
"$PY" -u -B scripts/analysis/phase3b_generality.py worker \
  --run "$RUN" --model sft --smoke --research-root "$PWD"
"$PY" -u -B scripts/analysis/phase3b_generality.py worker \
  --run "$RUN" --model grpo --smoke --research-root "$PWD"
"$PY" -B scripts/analysis/phase3b_generality.py worker-validate --run "$RUN" --smoke
```

Each worker checks the native 56-slot logits shape, finite values, argmax/token alignment, model revision, action support and full preprocessing hashes. It checks hook equivalence for greedy and sampled native inference on the first observation. The smoke log prints free GPU memory after checkpoint load. If a smoke stage stops, rerun that worker with `--resume` and the same `RUN`.

## C. Manual full GPU replay

**Run only after both smoke workers and smoke validation pass.** This is a server command for the operator; preparing this runbook does not start the full replay.

```bash
export SESSION="stage1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export LOG="$PWD/logs/${SESSION}.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "bash scripts/analysis/run_stage1_full.sh '$RUN' '$LOG'"
printf 'SESSION=%s\nLOG=%s\nRUN=%s\n' "$SESSION" "$LOG" "$RUN"
```

The shell script runs SFT and GRPO in separate sequential processes, each loading its checkpoint once for all remaining observations. The worker commits each completed observation as one directory. `--resume` checks every completed logits checksum and the bank, code, checkpoint and action-support contracts before continuing. An interrupted temporary directory is moved to `RUN/interrupted/`; no committed observation is overwritten. Optional bounded chunks use `worker --run "$RUN" --model sft --resume --max-new 250` (similarly for GRPO), at the cost of another checkpoint load per chunk.

To inspect, stop, and resume:

```bash
tmux ls
tail -n 40 "$LOG"
tmux send-keys -t "$SESSION" C-c
# For a new session, set a new SESSION and LOG, then rerun the same tmux command above.
```

Do not start concurrent workers for the same model/run. The full worker seal is written only after all expected observations finish. On normal completion the log ends with `worker-validate` PASS and `exit_status=0`.

## D. CPU summary, figures and report

```bash
"$PY" -B scripts/analysis/phase3b_generality.py worker-validate --run "$RUN"
"$PY" -B scripts/analysis/phase3b_generality.py summarize --run "$RUN"
"$PY" -B scripts/analysis/phase3b_generality.py figures --run "$RUN"
"$PY" -B scripts/analysis/phase3b_generality.py report --run "$RUN"
"$PY" -B scripts/analysis/phase3b_generality.py verify --run "$RUN"
cat "$RUN/analysis/summary.md"
"$PY" -B -c 'import json,sys; v=json.load(open(sys.argv[1])); print({k:v[k] for k in ("status","source_episodes","observations","figures","scope")})' "$RUN/analysis/validation.json"
```

Outputs: `bank/{manifest.json,observation_index.csv,observations/,validation.json}`, `workers/{sft,grpo}/`, `analysis/{observations,episodes,tasks,suites,outcomes,delta_entropy}.csv`, `analysis/metrics.json`, `analysis/figures/` with six figure types at each of two temperatures in PNG and PDF, `analysis/summary.md`, `analysis/validation.json`, and `run_manifest.json`.

Metrics average the 56 action slots within a selected observation, selected observations within their source episode, and then source episodes equally. Macro task means weight each task equally. The bootstrap resamples whole tasks within each suite. Outcome groups use the paired deterministic SFT/GRPO result for the same initial state; SFT-origin and GRPO-origin observations may differ later in the trajectory. T=1 and T=1.6 use offline softmax of one captured raw-logit forward per model/observation. These fixed nominal tasks do not support out-of-distribution or causal claims.
