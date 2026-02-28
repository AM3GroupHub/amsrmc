# Amorphous LiNbOCl4 Structure Generation

Melt-quench workflow to generate amorphous LiNbOCl4 using LAMMPS + MACE ML potential.
Starts from crystalline LiVOF4 template, does chemical substitution (V->Nb, F->Cl), then melt-quench MD.

## Scripts (run in order)

1. `1_prepare_and_relax.py` - Build supercell, NPT relax, apply disorder & density matching (HPC)
2. `2_melt_test.py` - Melt test at 1800 K, track MSD (HPC)
3. `3_plot_msd.py` - Plot MSD from melt test (local)
4. `4_melt_quench.py` - Melt -> quench -> equilibrate (HPC)
5. `5_analyze_structure.py` - Coordination, bond distances, bond angles (local)

Steps 1/2/4 need LAMMPS with MLIAP+Kokkos GPU. Steps 3/5 just need Python.

## Key parameters

Hardcoded at top of each script:
- `SUPERCELL = [9, 3, 3]` (~1600 atoms)
- `TARGET_DENSITY = 1.636` g/cm3
- `MELT_TEMP = 1800` K
- `LI_REMOVE_FRAC = 0.10`, `O_TO_CL_FRAC = 0.10`

## Output folders

- `melt_quench_step1/` `step2/` `step4/` - HPC run outputs
- `analysis_output/` - plots and summary from step 5