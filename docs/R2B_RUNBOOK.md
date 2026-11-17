# R2-B repeated rollout server runbook

Status at preparation: server GPU smoke incomplete; 100-job collection has not been launched by this checkout. The operator chose to proceed without waiting for smoke. The server must receive the updated `phase3b_pilot.py` and `run_r2b_overnight.sh` before using the new progress and diversity behavior. The completed nominal and Phase 3A/3B outputs are read only.

The local deliverable archive is `outputs/analysis/r2b-code-20261117.tar.gz`. Transfer it to the server repository root through the user's existing transfer path. Before extraction, check that `git rev-parse HEAD` is `357e23b` and `git status --short` shows no source changes; otherwise review the server differences first. Inspect `tar -tzf r2b-code-20261117.tar.gz`, then extract with `tar -xzf r2b-code-20261117.tar.gz` and review `git diff --check` plus `git diff --stat`. The archive contains only the runner, overnight script, CPU test, this runbook and the selection manifest.

## 1. Activate and identify the existing run

```bash
cd /nfs4/jhkim/repos/rlinf
source "$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
export RESEARCH_ROOT="$PWD"
export PHASE3A="$PWD/outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246"
export AUDIT="$PWD/outputs/analysis/phase3b_audit/phase3b-20261117T111837Z-2822692"
export PYTHONPATH="$PWD:$PWD/upstream/RLinf:${PYTHONPATH:-}"
export EMBODIED_PATH="$PWD/upstream/RLinf/examples/embodiment"
export LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PY="$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
tmux ls
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
df -h "$PWD/outputs"
```

Use the *existing* partial smoke run directory as `RUN`; it already references the 100-job plan. Find candidates with:

```bash
"$PY" -B - <<'PY'
import json
from pathlib import Path
for path in Path('outputs/analysis').rglob('manifest.json'):
    try: m=json.loads(path.read_text())
    except (ValueError, OSError): continue
    if m.get('experiment_hash')=='735e7c57b3d238786c1ebd331421dfa69cb7de2287d965349aa0f1bb9282513f' and 'plan' in m:
        print(path.parent.resolve(), 'entries=',len(list((path.parent/'entries').glob('*.json'))))
PY
```

Set `RUN` to the unique partial run directory printed above. If there is more than one, inspect their entries and choose the intended existing run. Then derive its exact plan:

```bash
export RUN=/nfs4/jhkim/repos/rlinf/outputs/analysis/REPLACE_WITH_EXISTING_RUN
export PLAN="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"])' "$RUN/manifest.json")"
"$PY" -B -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["expected"]==100 and p["experiment"]["repeats"]==5; print("PLAN PASS",p["expected"],p["experiment_hash"])' "$PLAN/manifest.json"
```

If no partial run exists, create a fresh plan and use a fresh run path instead:

```bash
export R2B_ID="r2b-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export PLAN="$PWD/outputs/analysis/repeated_rollouts/${R2B_ID}-plan"
export RUN="$PWD/outputs/analysis/repeated_rollouts/${R2B_ID}-run"
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-plan \
  --phase3a "$PHASE3A" --suite libero_10 --task-ids 3 8 \
  --trial-ids 0 1 2 3 4 --policies sft grpo --repeats 5 --output "$PLAN"
```

## 2. Preflight and overnight 100-job pilot

```bash
"$PY" -B -m unittest tests.test_repeated_rollout
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-preflight \
  --plan "$PLAN" --audit "$AUDIT" --research-root "$PWD"
bash -n scripts/analysis/run_r2b_overnight.sh
command -v tmux
export SESSION="r2b-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export LOG="$PWD/logs/${SESSION}.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "bash scripts/analysis/run_r2b_overnight.sh '$PLAN' '$AUDIT' '$RUN' '$LOG'"
printf 'SESSION=%s\nLOG=%s\nRUN=%s\n' "$SESSION" "$LOG" "$RUN"
```

The script logs the experiment ID, UTC start/end, main/upstream commits, free disk, GPU process list, preflight, each job's policy/task/seed, checkpoint load memory, and final validation. It appends to a dedicated log. It does not terminate other GPU processes or download checkpoints. It has no automatic retry loop.

Progress and log inspection while it runs:

```bash
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-progress --plan "$PLAN" --output "$RUN"
tail -n 40 "$LOG"
tmux ls
```

Stop only this session's foreground job with `tmux send-keys -t "$SESSION" C-c`. Completed episode directories remain committed. To resume, rerun `run_r2b_overnight.sh` with the same `PLAN`, `AUDIT`, and `RUN` and a new `LOG`. If interruption left an incomplete job directory, inspect it and add `--retry-incomplete` to a direct `repeat-run --resume` command. That option archives the incomplete directory under `RUN/interrupted/` and permits one retry of the job; a second retry is refused.

## 3. Complete validation and read-only analysis

```bash
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-validate --output "$RUN"
export DIVERSITY="$PWD/outputs/analysis/repeated_rollouts/$(basename "$RUN")-diversity"
"$PY" -B scripts/analysis/phase3b_pilot.py diversity --pilot "$RUN" --output "$DIVERSITY"
cat "$DIVERSITY/summary.md"
```

`repeat-validate` must return `COMPLETE_VALID`, 100 completed, 0 pending, 10 initial-state groups. The `diversity` stage writes 100 per-episode rows, 450 within/cross-policy EEF pairs, ten state rows and a Markdown report. It never edits the source rollout directory. Conditional diversity and branching metrics remain for R3.

## 4. Prepared expansions, do not run

The 20-seed Long plan can be created after the pilot. It contains 400 planned jobs and the pilot's 100 job IDs as a subset. The current runner intentionally refuses plans above 100 because it cannot yet import the 100 prior episodes without re-executing them. No expansion collection is authorized here.

```bash
export LONG_PLAN="$PWD/outputs/analysis/repeated_rollouts/long20-$(date -u +%Y%m%dT%H%M%SZ)-plan"
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-plan \
  --phase3a "$PHASE3A" --suite libero_10 --task-ids 3 8 \
  --trial-ids 0 1 2 3 4 --policies sft grpo --repeats 20 --output "$LONG_PLAN"
```

Cross-suite selection is fixed in `docs/R2B_SELECTION.json`: tasks 3 and 8, states 0 and 1 in Goal, Spatial, and Long, with ten seeds. Each plan has 80 jobs, 240 total. These are preparation commands only:

```bash
for suite in libero_goal libero_spatial libero_10; do
  "$PY" -B scripts/analysis/phase3b_pilot.py repeat-plan \
    --phase3a "$PHASE3A" --suite "$suite" --task-ids 3 8 \
    --trial-ids 0 1 --policies sft grpo --repeats 10 \
    --output "$PWD/outputs/analysis/repeated_rollouts/cross-${suite}-$(date -u +%Y%m%dT%H%M%SZ)-plan"
done
```

Future expansion decisions require the 100-job pilot's complete validation, seed/state controls, GPU and disk capacity, and a working reuse path for overlapping jobs. Outcome direction is not an expansion gate.
