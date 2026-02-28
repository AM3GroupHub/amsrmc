# Amorphous LiNbOCl4 Structure Generation

Generate an amorphous solid electrolyte structure (LiNbOCl4) via melt-quench molecular dynamics using a MACE machine-learning interatomic potential.

Starting from a crystalline LiVOF4 template, the workflow performs chemical substitution, supercell construction, NPT relaxation, melt-quench MD, and structural analysis.

## Workflow

The scripts are numbered in execution order. Steps 1, 2, and 4 require LAMMPS+MACE and should be run on a GPU-equipped machine (HPC).

| Step | Script | Description | Run on |
|------|--------|-------------|--------|
| 1 | `1_prepare_and_relax.py` | Build supercell, relax, apply disorder and density strain | HPC |
| 2 | `2_melt_test.py` | NVT melt test at 1800 K to verify melting | HPC |
| 3 | `3_plot_msd.py` | Plot MSD from melt test to confirm diffusive behavior | Local |
| 4 | `4_melt_quench.py` | Full melt-quench-equilibrate to generate amorphous structure | HPC |
| 5 | `5_analyze_structure.py` | Analyze coordination, bond distances, and bond angles | Local |

### Step 1: Prepare and Relax (`1_prepare_and_relax.py`)

1. Reads `LiVOF4.cif` crystal template
2. Chemical substitution: V -> Nb, F -> Cl to create LiNbOCl4
3. Builds a 9x3x3 supercell (~1600 atoms)
4. NPT relaxation at 300 K via LAMMPS + MACE (1000 steps, 2 ps)
5. Applies disorder: removes 10% of Li, replaces 10% of O with Cl
6. Scales cell to match experimental density of 1.636 g/cm3
7. Outputs: `disordered.data`, `disordered.cif`

### Step 2: Melt Test (`2_melt_test.py`)

1. Reads `disordered.data` from Step 1
2. NVT simulation at 1800 K for 200 ps
3. Computes per-species MSD (Li, Nb, O, Cl) during melting
4. Outputs: `msd_data.txt`, `lammps.log`

### Step 3: Plot MSD (`3_plot_msd.py`)

1. Parses MSD data from the LAMMPS log
2. Plots MSD vs time for each species
3. Linear MSD growth confirms diffusive motion (system is melted)
4. Output: `msd_melt_test.pdf`

### Step 4: Melt-Quench (`4_melt_quench.py`)

1. Reads `disordered.data` from Step 1
2. Three-phase MD simulation:
   - Melt at 1800 K for 200 ps (NVT)
   - Quench from 1800 K to 300 K over 150 ps (Langevin + NVE, ~10^13 K/s)
   - Equilibrate at 300 K for 50 ps (NVT)
3. Outputs: `amorphous.data`, `amorphous_LiNbOCl.cif`

### Step 5: Analyze Structure (`5_analyze_structure.py`)

1. Reads `amorphous_LiNbOCl.cif` from Step 4
2. Computes partial pair distribution functions g(r) for Li-O, Li-Cl, Nb-O, Nb-Cl
3. Determines CN cutoffs from first minimum of g(r)
4. Computes coordination number distributions, bond distance distributions, and bond angle distributions
5. Outputs saved to `analysis_output/`:
   - `coordination_numbers.pdf`
   - `bond_distances.pdf`
   - `bond_angles.pdf`
   - `analysis_summary.txt`

## Input File

- `LiVOF4.cif` -- Crystallographic template (space group Cmc2_1, orthorhombic). LiVOF4 is isostructural to the target LiNbOCl4 family.

## Example HPC Run Outputs

Each `melt_quench_step*/` directory contains a complete HPC run for that step, including the Python script, generated LAMMPS input, PBS submission script, and output files:

- `melt_quench_step1/` -- Step 1 outputs (supercell, relaxed structure, disordered structure)
- `melt_quench_step2/` -- Step 2 outputs (melt test log with MSD data)
- `melt_quench_step4/` -- Step 4 outputs (final amorphous structure)

## Dependencies

- Python 3.10+
- [pymatgen](https://pymatgen.org/) -- crystal structure manipulation, CIF I/O, substitution
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/) -- peak finding for g(r) analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/) -- structure I/O (used in Step 5)
- [LAMMPS](https://www.lammps.org/) -- MD engine, compiled with MLIAP and Kokkos GPU support
- [MACE](https://github.com/ACEsuit/mace) -- machine-learning interatomic potential

## LAMMPS + MACE Setup

The scripts use MACE via LAMMPS's MLIAP interface. The MACE model is automatically converted to LAMMPS format on first run. Key requirements:

- LAMMPS built with `MLIAP` and `KOKKOS` packages
- `pair_style mliap unified <model.pt> 0`
- GPU execution flags: `-k on g 1 -sf kk -pk kokkos newton on neigh half`
- `LD_LIBRARY_PATH` must include `$CONDA_PREFIX/lib` (handled automatically in the scripts)

## Configuration

All scripts use hardcoded parameters at the top of each file (no command-line arguments). Key parameters to modify:

| Parameter | Script | Default | Description |
|-----------|--------|---------|-------------|
| `SUPERCELL` | Step 1 | `[9, 3, 3]` | Supercell dimensions |
| `TARGET_DENSITY` | Step 1 | 1.636 g/cm3 | Experimental density |
| `LI_REMOVE_FRAC` | Step 1 | 0.10 | Fraction of Li sites removed |
| `O_TO_CL_FRAC` | Step 1 | 0.10 | Fraction of O sites replaced by Cl |
| `MELT_TEMP` | Steps 2, 4 | 1800 K | Melting temperature |
| `MELT_STEPS` | Steps 2, 4 | 100000 | Melting duration (200 ps) |
| `QUENCH_STEPS` | Step 4 | 75000 | Quench duration (150 ps) |

Paths to the MACE model and LAMMPS binary are auto-detected based on whether the script runs on HPC or locally.
