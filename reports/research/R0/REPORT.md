# R0 checkpoint and inference audit

**Overall state — BLOCKED** · 2026-11-27 KST

| Stage | Status | Evidence |
| --- | --- | --- |
| R0-1 Model structure | BLOCKED | [CPU tests](../../../outputs/research/R0/r0-1-local-20261127/cpu-tests.log) · [Preflight](../../../outputs/research/R0/r0-1-preflight-20261127/status.json) · [Smoke attempt](../../../outputs/research/R0/r0-1-smoke-20261127/status.json) · [Source and protocol evidence](../../../outputs/research/R0/r0-1-smoke-20261127/structure.json) |
| R0-2 Checkpoint difference | BLOCKED by R0-1 | Pending implementation after R0-1 passes |
| R0-3 Hook equivalence | BLOCKED by R0-2 | Pending implementation after R0-2 passes |

The repository inventory and saved Phase 3A protocol agree on the SFT baseline
`RLinf-OpenVLAOFT-LIBERO-130-Base-Lora@a4ea91b3897648f3c1395e5221edb40b8f53d1a4`
and GRPO `RLinf-OpenVLAOFT-LIBERO-130@473954241c8148bc9a1bf9ed547a74523e4a7d41`.
Both protocols disable the inference adapter. The SFT inventory includes a separate LoRA payload. Its merge history and the historical optimizer mask remain unresolved. [Inventory](../../../manifests/models_130.json) · [Checkpoint notes](../../../docs/MODELS.md)

The installed upstream checkout matched the Phase 3B revision and native action-source hash. The pinned source selects 56 parallel action slots from shared LM logits and decodes them into an 8 × 7 action chunk under the recorded policy configuration. Runtime registry resolution, parameter metadata, and action shapes still require a real load. Both checkpoint directories are absent on this host, and the local PyTorch environment reports CUDA unavailable. [Phase 3B reference](../../../scripts/analysis/phase3b_distribution.py)

**CPU verification** — 129 tests passed, one skipped, and 15 subtests passed using the existing `basilisk-rl` environment. **Not measured** — parameter changes and hook-equivalence differences.

## Reproduce

From the repository root on the checkpoint server, with the documented native environment and the saved Phase 3A manifest:

```bash
PYTHONPATH="$PWD/upstream/RLinf:$PWD" upstream/RLinf/.venv-rlvr-libero-cu121/bin/python -B \
  -m scripts.research.cli.audit --checkpoint-root /nfs4/jhkim/repos/rlinf/checkpoints \
  --run-id "r0-1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

Add `--dry-run` to verify checkpoint inventory, digests, configs, and shard indexes without loading the model. The smoke loads one checkpoint at a time and runs one deterministic query per checkpoint.

**R1** — Pending R0 completion. Use the verified modules and hook-equivalent inference on a fixed saved-observation bank.

**2026-11-27** — Added the R0-1 adapter, registry, and CLI. Recorded CPU verification and the local checkpoint and CUDA blockers.
