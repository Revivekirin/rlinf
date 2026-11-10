#!/usr/bin/env bash
# Explicit stages only. No all-stage or automatic long-GPU launch.
set -euo pipefail
export RESEARCH_ROOT="${RESEARCH_ROOT:-/nfs4/jhkim/repos/rlinf}"
: "${PHASE3B_RUN:?Set a fresh PHASE3B_RUN once and retain it for all stages}"
[[ "$PHASE3B_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid PHASE3B_RUN'; exit 1; }
cd "$RESEARCH_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$RESEARCH_ROOT/upstream/RLinf/examples/embodiment"
export PYTHONPATH="$RESEARCH_ROOT:$RESEARCH_ROOT/upstream/RLinf:${PYTHONPATH:-}"
export LIBERO_TYPE=standard
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-phase3b-mpl-$USER"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
P3A="${PHASE3A:-$RESEARCH_ROOT/outputs/analysis/nominal_behavior/phase3a-20261109T082746Z-1415246}"
AUDIT="$RESEARCH_ROOT/outputs/analysis/phase3b_audit/$PHASE3B_RUN"
STAG="$RESEARCH_ROOT/outputs/analysis/stagnation/$PHASE3B_RUN"
PRE="$RESEARCH_ROOT/outputs/analysis/policy_distribution/${PHASE3B_RUN}-preflight"
SMOKE="$RESEARCH_ROOT/outputs/analysis/policy_distribution/${PHASE3B_RUN}-smoke"
DIST="$RESEARCH_ROOT/outputs/analysis/policy_distribution/$PHASE3B_RUN"
PLAN="$RESEARCH_ROOT/outputs/analysis/stochastic_pilot/${PHASE3B_RUN}-plan"
PSMOKE="$RESEARCH_ROOT/outputs/analysis/stochastic_pilot/${PHASE3B_RUN}-smoke"
PILOT="$RESEARCH_ROOT/outputs/analysis/stochastic_pilot/${PHASE3B_RUN}-pilot-a"
DIVERSITY="$RESEARCH_ROOT/outputs/analysis/stochastic_diversity/$PHASE3B_RUN"
SFT_SAMPLING="$RESEARCH_ROOT/outputs/analysis/sft_sampling/$PHASE3B_RUN"
case "${1:-help}" in
  audit)
    "$PY" -B scripts/analysis/phase3b.py audit --phase3a "$P3A" --output "$AUDIT" ;;
  sft-sampling)
    CUDA_VISIBLE_DEVICES="" "$PY" -B scripts/analysis/phase3b_sft_sampling.py --research-root "$RESEARCH_ROOT" --output "$SFT_SAMPLING" ;;
  stagnation)
    CUDA_VISIBLE_DEVICES="" "$PY" -B scripts/analysis/phase3b.py stagnation --raw --phase3a "$P3A" --output "$STAG" ;;
  sensitivity)
    "$PY" -B - "$STAG" <<'PY'
import sys,csv
from pathlib import Path
sys.path.insert(0,'scripts/analysis')
from phase3b import verify_seal
p=Path(sys.argv[1]); verify_seal(p)
for row in csv.DictReader((p/'threshold_sensitivity.csv').open()): print(row)
PY
    ;;
  replay-preflight)
    "$PY" -B scripts/analysis/phase3b_distribution.py preflight --phase3a "$P3A" --audit "$AUDIT" --output "$PRE" ;;
  model-smoke)
    for model in sft grpo; do
      "$PY" -B scripts/analysis/phase3b_distribution.py worker --preflight "$PRE" --output "$SMOKE" --model "$model" --smoke
    done
    "$PY" -B scripts/analysis/phase3b_distribution.py summarize --preflight "$PRE" --output "$SMOKE" ;;
  distributions)
    for model in sft grpo; do
      "$PY" -B scripts/analysis/phase3b_distribution.py worker --preflight "$PRE" --output "$DIST" --model "$model" --smoke-result "$SMOKE"
    done ;;
  distribution-summary)
    "$PY" -B scripts/analysis/phase3b_distribution.py summarize --preflight "$PRE" --output "$DIST" ;;
  pilot-plan)
    "$PY" -B scripts/analysis/phase3b_pilot.py plan --phase3a "$P3A" --audit "$AUDIT" --output "$PLAN" ;;
  pilot-smoke)
    "$PY" -B scripts/analysis/phase3b_pilot.py run --plan "$PLAN" --mode smoke --output "$PSMOKE"
    "$PY" -B scripts/analysis/phase3b_pilot.py validate --output "$PSMOKE" ;;
  pilot-command)
    "$PY" -B - "$PSMOKE" "$PHASE3B_RUN" "$RESEARCH_ROOT" <<'PY'
import sys,shlex
from pathlib import Path
sys.path.insert(0,'scripts/analysis')
from phase3b import verify_seal, require
from nominal_behavior import read
p=Path(sys.argv[1]); v=verify_seal(p)
require(read(p/'manifest.json')['mode']=='smoke' and v['seed_divergence_verified'], 'Pilot smoke did not pass')
print('Smoke PASS. Run manually:')
print('RESEARCH_ROOT='+shlex.quote(sys.argv[3])+' PHASE3B_RUN='+shlex.quote(sys.argv[2])+' bash '+shlex.quote(sys.argv[3]+'/scripts/analysis/run_phase3b.sh')+' pilot-a')
PY
    ;;
  pilot-a)
    "$PY" -B scripts/analysis/phase3b_pilot.py run --plan "$PLAN" --mode pilot --smoke "$PSMOKE" --output "$PILOT" ;;
  pilot-validate)
    "$PY" -B scripts/analysis/phase3b_pilot.py validate --output "$PILOT" ;;
  diversity)
    CUDA_VISIBLE_DEVICES="" "$PY" -B scripts/analysis/phase3b_pilot.py diversity --pilot "$PILOT" --output "$DIVERSITY" ;;
  report)
    "$PY" -B scripts/analysis/phase3b.py report --stagnation "$STAG" --sft-sampling "$SFT_SAMPLING" --distribution "$DIST" --diversity "$DIVERSITY" --output "$RESEARCH_ROOT/outputs/analysis/phase3b_report/$PHASE3B_RUN" ;;
  help)
    echo 'Stages: audit sft-sampling stagnation sensitivity replay-preflight model-smoke distributions distribution-summary pilot-plan pilot-smoke pilot-command pilot-a pilot-validate diversity report' ;;
  *) echo 'Unknown stage'; exit 1 ;;
esac
