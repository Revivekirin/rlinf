# R1 — Camera Pair & Tensor Capture

**Overall · BLOCKED** · 2026-11-27 KST

SFT `Base-Lora@a4ea91b38976` · GRPO `LIBERO-130@473954241c81`. The R0-1 through R0-3 evidence and saved R0 logits were revalidated before R1 preflight. [R0 report](../R0/REPORT.md)

| Gate | Status | Evidence |
| --- | --- | --- |
| R1-1 · Pair alignment | BLOCKED | Source plan verified, local `robosuite` import unavailable. [Plan](../../../outputs/research/R1/r1-preflight-final-20261127/plan.json) · [Status](../../../outputs/research/R1/r1-preflight-final-20261127/status.json) |
| R1-2 · Tensor capture | BLOCKED by alignment | SFT 0 and GRPO 0 captured observations |
| R1-3 · Action regression | BLOCKED by alignment | R0 archived outputs verified, live SFT/GRPO replay pending |
| R1-4 · Offline validation | BLOCKED by incomplete evidence | Offline command correctly rejected the blocked run. [CPU checks](../../../outputs/research/R1/r1-cpu-final-20261127/pytest.log) |

**Dataset** · Target 3 original tasks (Spatial, Object, Goal), 3 saved init states each, 36 distinct views, and 27 pairs. Locally measured counts are 0 valid pairs and 0 captures. Scene selection uses the pinned PLUS classification and three severity levels per task.

**CPU tests** · 143 passed, 1 skipped, and 15 subtests passed.

PLUS parses `_view_` task-name parameters into runtime arena camera position and orientation, including a distance scale. The smoke reads the compiled camera pose and vertical FOV and replays them in a single canonical simulator after 15 zero-action settling steps. These are same-state camera diagnostics using official PLUS geometry. The official evaluator has its own reset semantics. The dataset verifies exact equality of simulator time, robot/object state, and controller and requires different RGB. Policy input uses RLinf’s 180-degree image rotation and the native OFT processor. No figure was generated.

## Server4 reproduction

Sync the R1 sources to the server checkout with the existing PLUS environment and asset files.

```bash
cd /nfs6/jhkim/repos/rlinf
conda activate base
PY="$PWD/upstream/RLinf/.venv-rlvr-liberoplus-cu121/bin/python"
test -x "$PY" || PY=/nfs4/jhkim/repos/rlinf/upstream/RLinf/.venv-rlvr-liberoplus-cu121/bin/python
export PYTHONPATH="$PWD:$PWD/upstream/RLinf:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MAGICK_HOME=/nfs4/jhkim/repos/rlinf/conda-envs/libmagick-plus
export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"

run_id="r1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -B -m scripts.research.cli.capture \
  --r0-3 outputs/research/R0/r0-3-20261127T055218Z-3098626 \
  --checkpoint-root /nfs4/jhkim/repos/rlinf/checkpoints --run-id "$run_id" &&
"$PY" -B -m scripts.research.cli.validate "outputs/research/R1/$run_id"
```

Use `--assets /path/to/installed/PLUS/assets` if the installed asset directory is in a nonstandard location. For simulator-only preflight, add `--pairs-only` and use a separate run ID. This yields PARTIAL after successful pair validation.

Server attempt `r1-20261127T061448Z-3098626` stopped before rendering when PLUS resolved its scene XML relative to the source checkout. The retry stages a temporary copy of the pinned Python package beside a link to the installed assets, which supplies the layout required by the scene and object loaders.

Server retry `r1-pairs-20261127T061948Z-3098626` reached the first physical-state check and failed with zero valid pairs. The next retry synchronizes derived poses before freezing the baseline and uses MuJoCo's camera and light kinematics for each camera update. State equality remains exact, with separate checks after camera update and RGB rendering. The revised path passed 25 local research tests and awaits server verification.

**Next** · Verify the full server run and its offline gate before judging R2 readiness. The smoke tests artifact and inference validity for its selected cases.
