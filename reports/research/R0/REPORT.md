# R0 checkpoint and inference audit

**Overall state — PARTIAL** · 2026-11-27 KST

| Stage | Status | Evidence |
| --- | --- | --- |
| R0-1 · Model structure | PASS on server4 | [Status](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/status.json) · [Structure and identities](../../../outputs/research/R0/r0-1-20261127T050947Z-3098626/structure.json) |
| R0-2 · Checkpoint differences | PASS on server4, evidence verified locally | [Status](../../../outputs/research/R0/r0-2-20261127T052515Z-3098626/status.json) · [Tensor and module results](../../../outputs/research/R0/r0-2-20261127T052515Z-3098626/comparison.json) · [Module CSV](../../../outputs/research/R0/r0-2-20261127T052515Z-3098626/modules.csv) |
| R0-3 · Hook equivalence | BLOCKED locally · server CUDA run pending | [CPU suite](../../../outputs/research/R0/r0-3-cpu-20261127/pytest.log) · [Replay preflight](../../../outputs/research/R0/r0-3-preflight-20261127/provenance.json) · [Local status](../../../outputs/research/R0/r0-3-local-20261127/status.json) |

SFT `Base-Lora@a4ea91b38976` and GRPO `LIBERO-130@473954241c81` use adapter-disabled native inference. The shared LM projection supplies 56 action-token slots, decoded to 8 × 7 actions. SFT’s published LoRA payload and historical trainable masks require separate provenance.

Across 982 aligned tensors at zero tolerance on stored values, 844 differed and 138 were identical. Five GRPO-only value-head tensors fell outside the loaded policy registry. Overall, 24.81% of 7.541 billion aligned elements differed. Changed fractions were 19.43% in vision (DINOv2 20.07%, SigLIP 18.98%), 40.52% in the projector, 26.24% in transformer blocks, and 0.183% in the shared LM projection. Weight changes establish snapshot differences. Training attribution and camera sensitivity remain separate questions.

![R0-2 changed parameter fractions and relative L2 differences](figures/r0-2-20261127T052515Z-3098626-modules-local.png)

*Figure regenerated locally from the verified server comparison. Fractions use aligned elements in each labeled module. Relative L2 uses the corresponding SFT norm.*

R0-3 authenticated saved observation `sft-libero_10-t05-i03-q0000`. Real hook equivalence remains unmeasured. The CUDA run uses an uninstrumented prediction, the Phase 3B reference, and two registry-instrumented predictions per policy. Tokens, actions, and processed inputs must match exactly. Native inference exposes no uninstrumented raw logits, so the raw action-bin slice is checked against the independent Phase 3B LM-output hook.

Local CPU tests recorded 138 passed, 1 skipped, and 15 subtests passed.

## Reproduce on server4

```bash
cd /nfs6/jhkim/repos/rlinf
conda activate base
source "$PWD/upstream/RLinf/.venv-rlvr-libero-cu121/bin/activate"
export PYTHONPATH="$PWD/upstream/RLinf:$PWD"

r0_1=outputs/research/R0/r0-1-20261127T050947Z-3098626
r0_2=outputs/research/R0/r0-2-20261127T052515Z-3098626
checkpoints=/nfs4/jhkim/repos/rlinf/checkpoints

python -B -m scripts.research.cli.check_hook_equivalence \
  --r0-1 "$r0_1" --r0-2 "$r0_2" --checkpoint-root "$checkpoints" \
  --run-id "r0-3-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

To reproduce the CPU comparison, use `python -B -m scripts.research.cli.compare_checkpoints --r0-1 "$r0_1" --checkpoint-root "$checkpoints" --run-id "r0-2-$(date -u +%Y%m%dT%H%M%SZ)-$$"`.

**Next experiment** — R1 Camera pair construction, after verified R0-3 PASS.

**2026-11-27** — Verified the server R0-2 results and added the R0-3 replay and hook checks.
