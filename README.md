# hrmc

Reverse Monte Carlo (RMC) and hybrid RMC (HRMC) refinement of amorphous structures against a
neutron total pair distribution function G(r). It accompanies our paper on amorphous
Li0.9NbO0.9Cl4.1 refined with the foundation potential MACE-MH-1.

Three methods share one Monte Carlo walker (single-atom displacement moves, hard-sphere cutoffs,
incremental partial histograms):

- `rmc.run` – pure RMC, with an optional coordination-number (CN) constraint
  (`cn=None`, `cn="input"` for the input structure's distribution, or an explicit distribution);
- `hrmc.inline` – the MACE energy enters every acceptance, exp(-dX2/2σ² - dE/kBT);
- `hrmc.alternating` – pure-RMC moves, with N FIRE steps under MACE after every M accepted moves.

The package is system-agnostic: hard-sphere cutoffs, scattering lengths, the fitting window and any
CN target are arguments. All other defaults are the paper's settings.

## Install

See [INSTALL.md](INSTALL.md). Pure RMC needs only numpy and ASE and runs on a CPU; HRMC also
needs PyTorch, mace-torch and the MACE-MH-1 checkpoint. cuEquivariance is optional.

## Data and model

- Experimental G(r): Butenko et al., *Small Methods* 10, e00947 (2026),
  doi:10.1002/smtd.202500947. Save the data as a two-column file (r in Å, G(r)) at `data/exp_Gr.dat`.
- MACE-MH-1: from the MACE foundation-model page (see [INSTALL.md](INSTALL.md)).

## Examples

Each folder in `examples/` starts from the same melt-quench glass (2236 atoms) and holds the input,
a `run.py` with every parameter at the top, and the output of one paper chain
(`final_structure.xyz`, `trace.csv`, `traj.xyz`).

```
python examples/rmc_no_cn/run.py
```

| example | method |
|---|---|
| `rmc_no_cn` | pure RMC |
| `rmc_cn` | pure RMC with the Nb-O coordination constraint |
| `hrmc_inline` | inline HRMC, T = 175 K, 1.2 × 10⁶ moves |
| `hrmc_alternating` | alternating HRMC, M = 500, N = 5, 2 × 10⁷ moves |

**HRMC is a GPU job.** Pure RMC runs on a CPU.

## Citation

If you use this code, please cite: [paper reference to be added].
