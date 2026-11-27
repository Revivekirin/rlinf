# R1 — Camera Pair & Tensor Capture

**Overall · PARTIAL** · 2026-11-27 KST

SFT `Base-Lora@a4ea91b38976` · GRPO `LIBERO-130@473954241c81`. The R0-1 through R0-3 evidence and saved R0 logits were revalidated before R1 preflight. [R0 report](../R0/REPORT.md)

| Gate | Status | Evidence |
| --- | --- | --- |
| R1-1 · Pair alignment | PASS on server4 | Run `r1-pairs-20261127T062618Z-3098626` reported 27 valid pairs and 36 observations. [Local plan](../../../outputs/research/R1/r1-preflight-final-20261127/plan.json) · [Status](../../../outputs/research/R1/r1-preflight-final-20261127/status.json) |
| R1-2 · Tensor capture | Pending server GPU run | SFT 0 and GRPO 0 captured observations |
| R1-3 · Action regression | Pending server GPU run | R0 archived outputs verified, live SFT/GRPO replay pending |
| R1-4 · Offline validation | BLOCKED by incomplete evidence | Offline command correctly rejected the blocked run. [CPU checks](../../../outputs/research/R1/r1-cpu-final-20261127/pytest.log) |

**Dataset** · Target 3 original tasks (Spatial, Object, Goal), 3 saved init states each, 36 distinct views, and 27 pairs. The supplied server log reports all 27 pairs and 36 views validated. The GPU capture rechecks the parent artifact hashes and state alignment before using those same files. Scene selection uses the pinned PLUS classification and three severity levels per task.

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

Server retry `r1-pairs-20261127T062331Z-3098626` used the standard LIBERO interpreter and was blocked during PLUS asset discovery. Invoke the PLUS interpreter explicitly with `"$PY"` for both the full run and `--pairs-only`, even if the shell prompt shows a PLUS environment.

Server retry `r1-pairs-20261127T062447Z-3098626` used the PLUS interpreter and reproduced the earlier traceback byte-for-byte. The error text comes from the earlier simulator source. Before retrying, verify that the imported `scripts.research.data.simulator_state` resolves to this checkout and contains the `mj_camlight` update.

**Next** · Reuse the validated observations for GPU capture. Use the Python environment from the passing R0-3 run, set the repository and RLinf upstream on `PYTHONPATH`, then execute:

```bash
cd /nfs6/jhkim/repos/rlinf
pairs=outputs/research/R1/r1-pairs-20261127T062618Z-3098626
run_id="r1-capture-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python -B -m scripts.research.cli.capture \
  --r0-3 outputs/research/R0/r0-3-20261127T055218Z-3098626 \
  --pairs-from "$pairs" --run-id "$run_id" &&
python -B -m scripts.research.cli.validate "outputs/research/R1/$run_id"
```

Verify the full server run and its offline gate before judging R2 readiness. The smoke tests artifact and inference validity for its selected cases.
