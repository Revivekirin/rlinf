# R2 — Module-wise Parameter Intervention

**Overall · BLOCKED** · 2026-11-27 KST

| Stage | Status |
| --- | --- |
| R0/R1 evidence and group inventory | Verified locally from the sealed runs |
| CPU implementation tests | 32 research tests passed. Repository regression run recorded 150 passed, 1 skipped, and 15 subtests passed before the two unit-evidence tests were added. |
| GPU smoke | BLOCKED locally because CUDA is unavailable. 0 of 21 units executed. |
| Full sweep | Pending GPU smoke. 0 of 1,983 units executed. |
| Offline outcome validation | CPU fixture validation passed. No measured units to verify. |

| Parameter region | Direct-owner groups | Aligned scalar elements |
| --- | ---: | ---: |
| DINOv2 | 195 | 303,230,976 |
| SigLIP | 172 | 427,680,704 |
| Visual projector | 3 | 71,385,600 |
| Transformer | 288 | 6,476,267,520 |
| LM input embedding and final norm | 2 | 131,338,240 |
| Shared LM output projection | 1 | 131,334,144 |
| **Total** | **661** | **7,541,237,184** |

R0-2 accounts for 982 aligned runtime tensors. The 661 direct-owner groups contain 608 groups with changed values and 53 `NO_UPDATE` groups. At the stored-value level, 1,870,875,515 elements differ and 5,670,361,669 are identical. Five GRPO-only value-head tensors (2,164,352 elements) are outside the loaded native policy and excluded as `UNSAFE`. Live storage alias checks and owner-forward coverage await GPU execution. [Inventory and checkpoint provenance](../../../outputs/research/R2A/r2a-preflight-20261127/manifest.json) · [R1 evidence](../R1/REPORT.md)

Each unit preserves raw action-bin logits, tokens, and decoded commands. Intermediate activations retain tensor hashes and L2/max-absolute differences to bound output size. Unit records include direct canonical/OOD effects and the within-pair response change in decoded command units. Stored R1 SFT results supply the unmodified SFT control. Live GRPO must match sealed R0 and R1 controls. Selected weights are checked against the original GRPO shards after restoration, and standard in-place changes outside the selection are checked with PyTorch version counters. **Effective parameter, representation, and action effects remain unmeasured.** [Local smoke attempt](../../../outputs/research/R2A/r2a-local-smoke-20261127/attempts) · [Research tests](../../../outputs/research/R2A/r2a-cpu-20261127/final-research.log) · [Repository regressions](../../../outputs/research/R2A/r2a-regressions-20261127/pytest.log)

## Server4 reproduction

```bash
cd /nfs6/jhkim/repos/rlinf
conda activate base
PY="$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python"
test -x "$PY" || PY=/nfs4/jhkim/repos/rlinf/upstream/RLinf/.venv-rlvr-libero-cu121/bin/python
test -x "$PY"
export PYTHONPATH="$PWD:$PWD/upstream/RLinf${PYTHONPATH:+:$PYTHONPATH}"
r1=outputs/research/R1/r1-capture-20261127T063211Z-3098626
run_id="r2a-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -B -m scripts.research.cli.intervene --r1 "$r1" --run-id "$run_id" --mode smoke &&
"$PY" -B -m scripts.research.cli.validate_intervention "outputs/research/R2A/$run_id" --r1 "$r1"
```

After `SMOKE_PASS`, use a new run ID with `--mode full` for all 661 groups and all 36 observations. To run in batches, add `--max-units N`; resume with the same mode and ID plus `--resume "outputs/research/R2A/$run_id"`. The coarse mode covers six parent groups and cannot satisfy the leaf sweep. R2-B is outside this run.
