# RLinf VLA-RLVR Research

A separate research workspace for evaluating public OpenVLA-OFT SFT and GRPO checkpoints with the official RLinf runtime. The existing `vla-rlvr` repository is not used or modified.

## Current status

- Original LIBERO runtime installed in an isolated virtual environment.
- LIBERO-130 SFT and GRPO checkpoints downloaded and audited.
- Both checkpoints completed a one-episode LIBERO-130 runtime smoke test.
- Full-suite nominal evaluation, LIBERO-PLUS OOD evaluation, and training have not been run.

The smoke results in `manifests/libero_130_smoke.json` verify execution only. One episode per model is insufficient for a performance comparison.

## Layout

- `upstream/RLinf/`: independent clone of official RLinf, pinned by commit SHA
- `checkpoints/`: local model weights; excluded from Git
- `manifests/`: checkpoint provenance, package inventory, and smoke results
- `docs/`: installation, model, evaluation, and experiment protocols
- `configs/`, `scripts/`, `analysis/`: research extensions kept separate from upstream
- `datasets/`, `outputs/`, `logs/`: local data and results; excluded from Git

## Documentation

- [Installation](docs/INSTALL.md)
- [Models](docs/MODELS.md)
- [Evaluation](docs/EVALUATION.md)
- [Experiment plan](docs/EXPERIMENTS.md)

## Upstream

RLinf: https://github.com/RLinf/RLinf

Pinned checkout: `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d`

The upstream clone is recorded by SHA rather than tracked as a Git submodule. This keeps the research repository independent of upstream working-tree changes while allowing an exact checkout to be recreated.
