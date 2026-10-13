"""Inline HRMC with MACE-MH-1 from melt-quench glass b1 (GPU job)."""
import sys
from pathlib import Path
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hrmc import read_gr, hrmc  # noqa: E402

GR_FILE = ROOT / "data" / "exp_Gr.dat"
INPUT = HERE / "input" / "mace_b1.data"
TYPES = {1: 17, 2: 3, 3: 41, 4: 8}          # LAMMPS type -> atomic number (Cl, Li, Nb, O)

WINDOW = (1.58, 15.0, 482)                   # r_min, r_max (A), bins
B = {"Cl": 9.577, "Li": -1.9, "Nb": 7.054, "O": 5.803}   # neutron scattering lengths (fm)
HARD_SPHERE = {("Cl", "Cl"): 2.76, ("Cl", "Li"): 1.93, ("Cl", "Nb"): 2.03, ("Cl", "O"): 2.44,
               ("Li", "Li"): 2.39, ("Li", "Nb"): 2.69, ("Li", "O"): 1.59, ("Nb", "Nb"): 2.84,
               ("Nb", "O"): 1.58, ("O", "O"): 2.25}

MODEL = ROOT / "models" / "mace-mh-1.model"
HEAD = "matpes_r2scan"
ENABLE_CUEQ = False                          # True: cuEquivariance speed-up (see INSTALL.md)

if __name__ == "__main__":
    atoms = read(INPUT, format="lammps-data", atom_style="atomic", Z_of_type=TYPES)
    gr = read_gr(GR_FILE)
    print(hrmc.inline(atoms, gr, HARD_SPHERE, B, WINDOW, HERE / "output", MODEL,
                      head=HEAD, enable_cueq=ENABLE_CUEQ))
