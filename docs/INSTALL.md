# Installation and environment record

## Server used for the initial smoke tests

- Workspace: `/nfs4/jhkim/repos/rlinf`
- GPU: NVIDIA RTX A5000, 24 GiB
- Driver: 535.230.02
- Python: 3.11.14
- RLinf commit: `e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d`
- Environment: `upstream/RLinf/.venv-rlvr-libero-cu121`

The virtual environment name contains `cu121`, but the installed PyTorch reports **2.5.1+cu124** and CUDA build **12.4**. Use the reported package versions, not the directory name, when interpreting results. A bounded CUDA tensor calculation passed on the A5000.

## Recreate the upstream checkout

After cloning this research repository:

```bash
cd /nfs4/jhkim/repos/rlinf
git clone https://github.com/RLinf/RLinf.git upstream/RLinf
git -C upstream/RLinf checkout e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d
