# R0 checkpoint and inference audit

**Overall state — PARTIAL** · 2026-11-27 KST

| Stage | Status | Evidence |
| --- | --- | --- |
| R0-1 · Model structure | PASS on server4 | [Status](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/status.json) · [SFT registry](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/sft-modules.json) · [GRPO registry](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/grpo-modules.json) |
| R0-2 · Checkpoint differences | BLOCKED locally · server run pending | [CPU suite](../../../outputs/research/R0/r0-2-cpu-20261127/pytest.log) · [Run status](../../../outputs/research/R0/r0-2-local-20261127/status.json) · [Provenance](../../../outputs/research/R0/r0-2-local-20261127/provenance.json) |
| R0-3 · Hook equivalence | Pending R0-2 PASS | Hook and real-model inference comparisons unmeasured |

The passing R0-1 smoke identifies SFT `Base-Lora@a4ea91b38976` and GRPO `LIBERO-130@473954241c81`. Both use adapter-disabled inference and the shared language-model output for 56 parallel action-token slots, decoded to 8 × 7 actions. The SFT release contains a separate LoRA payload. Merge history and training-time optimizer membership remain unresolved. [Structure evidence](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/structure.json) · [Checkpoint inventory](../../../manifests/models_130.json)

The copied registries agree on names, shapes and runtime dtypes for 982 parameter tensors and 7,541,237,184 elements per policy. This includes 730,911,680 vision elements (DINOv2 303,230,976 and SigLIP 427,680,704), 71,385,600 projector elements, 6,476,267,520 elements across 32 transformer blocks, and 131,334,144 in the shared LM output projection. These are runtime structure counts. **Weight differences and hook equivalence remain unmeasured.** A zero vision delta would establish equality of the compared tensors. Camera sensitivity requires a separate experiment.

The local CPU suite recorded 134 tests passed, 1 skipped, and 15 subtests passed. The local comparison stopped after checking R0-1 provenance because the checkpoint storage is unavailable.

## Reproduce R0-2 on server4

With the new R0-2 sources in the server checkout and the existing RLinf Python environment active:

```bash
cd /nfs6/jhkim/repos/rlinf
python -B -m scripts.research.cli.compare_checkpoints \
  --r0-1 outputs/research/R0/r0-1-20261127T050947Z-3098626 \
  --checkpoint-root /nfs4/jhkim/repos/rlinf/checkpoints \
  --run-id "r0-2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

Add `--dry-run` with a fresh run ID to authenticate files and validate key, shape, and dtype correspondence without computing deltas. The full run streams CPU chunks and classifies changes at zero tolerance on stored values, with float64 reductions. JSON and CSV results go under the run ID, and the measured figure goes under `figures/`. R0-3 implementation starts after verification of the R0-2 PASS evidence.

**Next experiment** — R1 Camera pair construction, after R0 passes.

**2026-11-27** — Recorded server4 R0-1 PASS and added the R0-2 comparison, CPU tests, and server command.
