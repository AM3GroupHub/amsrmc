# Prerequisites

Python 3.10 or later.

## Pure RMC (CPU)

```
pip install numpy ase
```

ASE: https://ase-lib.org

## Hybrid RMC (GPU)

1. PyTorch with CUDA, for your system: https://pytorch.org/get-started/locally
2. MACE:
   ```
   pip install mace-torch
   ```
   Documentation: https://mace-docs.readthedocs.io
3. MACE-MH-1 foundation model (multi-head; the paper uses head `matpes_r2scan`).
   Download the checkpoint from the MACE foundation-model page,
   https://github.com/ACEsuit/mace-foundations, and save it as `models/mace-mh-1.model`
   (or change `MODEL` in the example's `run.py`).
4. cuEquivariance (optional; speeds up MACE on NVIDIA GPUs):
   ```
   pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
   ```
   Use `cu11` instead of `cu12` for CUDA 11. Then set `ENABLE_CUEQ = True` in `run.py`.
   Details: https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html
