#!/usr/bin/env bash
# Resume an existing repeated-rollout run. Run inside a dedicated tmux session.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: run_r2b_overnight.sh PLAN AUDIT RUN LOG' >&2
  exit 2
fi
PLAN="$1"
AUDIT="$2"
RUN="$3"
LOG="$4"
RESEARCH_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$RESEARCH_ROOT"
PY="$RESEARCH_ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -f "$PLAN/validation.json" && -f "$AUDIT/validation.json" ]] || {
  echo 'Missing sealed plan or audit' >&2; exit 1;
}
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$RESEARCH_ROOT/upstream/RLinf/examples/embodiment"
export PYTHONPATH="$RESEARCH_ROOT:$RESEARCH_ROOT/upstream/RLinf:${PYTHONPATH:-}"
export LIBERO_TYPE=standard
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-r2b-mpl-$USER"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "experiment_id=$(basename "$RUN")"
echo "start_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit=$(git rev-parse HEAD)"
echo "upstream_commit=$(git -C upstream/RLinf rev-parse HEAD)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
df -h "$RESEARCH_ROOT/outputs"
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-preflight \
  --plan "$PLAN" --audit "$AUDIT" --research-root "$RESEARCH_ROOT"
"$PY" -u -B scripts/analysis/phase3b_pilot.py repeat-run \
  --plan "$PLAN" --audit "$AUDIT" --output "$RUN" \
  --research-root "$RESEARCH_ROOT" --resume
"$PY" -B scripts/analysis/phase3b_pilot.py repeat-validate --output "$RUN"
echo "end_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
