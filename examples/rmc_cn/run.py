"""Pure RMC with the Nb-O coordination-number constraint, from melt-quench glass b1."""
import sys
from pathlib import Path
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hrmc import read_gr, rmc  # noqa: E402

GR_FILE = ROOT / "data" / "exp_Gr.dat"
INPUT = HERE / "input" / "mace_b1.data"
TYPES = {1: 17, 2: 3, 3: 41, 4: 8}          # LAMMPS type -> atomic number (Cl, Li, Nb, O)

WINDOW = (1.58, 15.0, 482)                   # r_min, r_max (A), bins
B = {"Cl": 9.577, "Li": -1.9, "Nb": 7.054, "O": 5.803}   # neutron scattering lengths (fm)
HARD_SPHERE = {("Cl", "Cl"): 2.76, ("Cl", "Li"): 1.93, ("Cl", "Nb"): 2.03, ("Cl", "O"): 2.44,
               ("Li", "Li"): 2.39, ("Li", "Nb"): 2.69, ("Li", "O"): 1.59, ("Nb", "Nb"): 2.84,
               ("Nb", "O"): 1.58, ("O", "O"): 2.25}

# Fraction of Nb with 0..7 O within 2.92 A: mean over five MACE-MH-1 melt-quench glasses.
# cn="input" would instead take the distribution of the input structure.
CN_CUTOFF = {("Nb", "O"): 2.92}
CN_TARGET = {("Nb", "O"): [(0, 0.0981), (1, 0.3636), (2, 0.3741), (3, 0.1451),
                           (4, 0.0179), (5, 0.0012), (6, 0.0), (7, 0.0)]}
CN_WEIGHT = 0.9

if __name__ == "__main__":
    atoms = read(INPUT, format="lammps-data", atom_style="atomic", Z_of_type=TYPES)
    gr = read_gr(GR_FILE)
    print(rmc.run(atoms, gr, HARD_SPHERE, B, WINDOW, HERE / "output",
                  cn=CN_TARGET, cn_cutoff=CN_CUTOFF, cn_weight=CN_WEIGHT))
