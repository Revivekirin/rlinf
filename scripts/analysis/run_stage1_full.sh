#!/usr/bin/env bash
# Explicit, sequential full common-observation replay. Invoke only after both GPU smokes pass.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: run_stage1_full.sh RUN LOG' >&2
  exit 2
fi
RUN="$1"
LOG="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
[[ -x "$PY" && -f "$RUN/bank/validation.json" ]] || {
  echo 'Missing server Python or sealed bank' >&2
  exit 1
}
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
finish() {
  status=$?
  trap - EXIT
  echo "end_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_status=$status"
  exit "$status"
}
trap finish EXIT
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream/RLinf:${PYTHONPATH:-}"
export EMBODIED_PATH="$ROOT/upstream/RLinf/examples/embodiment"
export LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rlinf-stage1-mpl-$USER"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "run=$RUN"
echo "start_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit=$(git rev-parse HEAD)"
echo "upstream_commit=$(git -C upstream/RLinf rev-parse HEAD)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
df -h "$RUN"
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
"$PY" -u -B scripts/analysis/phase3b_generality.py worker --run "$RUN" --model sft --resume
"$PY" -u -B scripts/analysis/phase3b_generality.py worker --run "$RUN" --model grpo --resume
"$PY" -B scripts/analysis/phase3b_generality.py worker-validate --run "$RUN"
