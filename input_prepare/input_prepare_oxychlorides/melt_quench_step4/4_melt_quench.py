import os
import re
import subprocess
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice

# ---- Parameters ----

INPUT_DATA = "disordered.data"
OUTPUT_CIF = "amorphous_LiNbOCl.cif"
MELT_TEMP = 1800.0  # K (change to 2000 if melt test shows insufficient melting)
QUENCH_TEMP = 300.0
TIMESTEP = 0.002  # ps (2 fs)

MELT_STEPS = 100000    # 200 ps
QUENCH_STEPS = 75000   # 150 ps (quench rate ~10^13 K/s)
EQUIL_STEPS = 25000    # 50 ps

THERMO_INTERVAL = 500
DUMP_INTERVAL = 5000
SEED = 42

# ---- Paths ----

HPC_BASE = "/home/svu/xintian.wang/project"
LOCAL_BASE = "/home/xintianw/Desktop/amorphousSSE"

if Path(HPC_BASE).exists():
    BASE = HPC_BASE + "/amorphousSSE"
    LMP_BASE = HPC_BASE
else:
    BASE = LOCAL_BASE
    LMP_BASE = LOCAL_BASE

MACE_MODEL = f"{BASE}/models/MACE-matpes-r2scan-omat-ft.model"
LMP_BINARY = f"{LMP_BASE}/mliap-mace-lammps/build-mliap/lmp"

ATOMIC_MASSES = {
    "Cl": 35.453, "Li": 6.941, "Nb": 92.906, "O": 15.999,
}

# Element-to-type mapping (alphabetical)
# Type 1: Cl, Type 2: Li, Type 3: Nb, Type 4: O
ELEMENTS = ["Cl", "Li", "Nb", "O"]


def convert_mace_model(model_path):
    out_path = model_path + "-mliap_lammps.pt"
    if os.path.exists(out_path):
        print(f"Converted model already exists: {out_path}")
        return out_path
    print(f"Converting MACE model for LAMMPS MLIAP...")
    model_dir = os.path.dirname(model_path)
    original_dir = os.getcwd()
    os.chdir(model_dir)
    try:
        subprocess.run(
            ["python", "-m", "mace.cli.create_lammps_model",
             model_path, "--format", "mliap", "--dtype", "float32"],
            check=True,
        )
    finally:
        os.chdir(original_dir)
    print(f"Saved: {out_path}")
    return out_path


def read_lammps_data(filename, inv_type_map):
    with open(filename) as f:
        content = f.read()
    n_atoms = int(re.search(r"(\d+)\s+atoms", content).group(1))
    xlo, xhi = [float(x) for x in re.search(r"([\d.e+-]+)\s+([\d.e+-]+)\s+xlo xhi", content).groups()]
    ylo, yhi = [float(x) for x in re.search(r"([\d.e+-]+)\s+([\d.e+-]+)\s+ylo yhi", content).groups()]
    zlo, zhi = [float(x) for x in re.search(r"([\d.e+-]+)\s+([\d.e+-]+)\s+zlo zhi", content).groups()]
    lattice = Lattice.from_parameters(xhi - xlo, yhi - ylo, zhi - zlo, 90, 90, 90)
    atoms_section = content.split("Atoms")[1]
    if "Velocities" in atoms_section:
        atoms_section = atoms_section.split("Velocities")[0]
    positions = np.zeros((n_atoms, 3))
    species = [""] * n_atoms
    for line in atoms_section.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            aid = int(parts[0])
            tid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            positions[aid - 1] = [x, y, z]
            species[aid - 1] = inv_type_map[tid]
        except (ValueError, KeyError):
            continue
    stru = Structure(lattice, species, positions, coords_are_cartesian=True)
    print(f"Read back: {len(stru)} atoms, cell: {lattice.a:.3f} x {lattice.b:.3f} x {lattice.c:.3f}")
    return stru


def run_lammps(lmp_binary, input_file):
    cmd = [
        lmp_binary,
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
        "-in", input_file,
    ]
    print(f"Running LAMMPS: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open("lammps.log", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    if result.returncode != 0:
        print(f"LAMMPS STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"LAMMPS failed with exit code {result.returncode}")
    print("LAMMPS completed successfully.")


def main():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        conda_lib = f"{conda_prefix}/lib"
        if not ld_path.startswith(conda_lib):
            os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"

    print("=== Melt-Quench-Equilibrate ===")
    print(f"  Melt: {MELT_TEMP} K for {MELT_STEPS * TIMESTEP:.0f} ps")
    print(f"  Quench: {MELT_TEMP} -> {QUENCH_TEMP} K over {QUENCH_STEPS * TIMESTEP:.0f} ps")
    print(f"  Equilibrate: {QUENCH_TEMP} K for {EQUIL_STEPS * TIMESTEP:.0f} ps")

    mace_pt = convert_mace_model(MACE_MODEL)

    elem_str = " ".join(ELEMENTS)
    mq_script = f"""units           metal
atom_style      atomic
newton          on
boundary        p p p
read_data       {INPUT_DATA}

pair_style      mliap unified {mace_pt} 0
pair_coeff      * * {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        {TIMESTEP}
thermo          {THERMO_INTERVAL}
thermo_style    custom step temp pe ke etotal press vol

dump            1 all custom {DUMP_INTERVAL} melt_quench.dump id type x y z

# Phase 1: Melt at {MELT_TEMP} K
velocity        all create {MELT_TEMP} {SEED} dist gaussian
fix             1 all nvt temp {MELT_TEMP} {MELT_TEMP} 0.1
run             {MELT_STEPS}
unfix           1

# Phase 2: Quench from {MELT_TEMP} K to {QUENCH_TEMP} K
fix             2 all langevin {MELT_TEMP} {QUENCH_TEMP} 0.1 42
fix             3 all nve
run             {QUENCH_STEPS}
unfix           2
unfix           3

# Phase 3: Equilibrate at {QUENCH_TEMP} K
fix             4 all nvt temp {QUENCH_TEMP} {QUENCH_TEMP} 0.1
run             {EQUIL_STEPS}
unfix           4

write_data      amorphous.data
"""
    with open("melt_quench.in", "w") as f:
        f.write(mq_script)
    print("Saved: melt_quench.in")

    run_lammps(LMP_BINARY, "melt_quench.in")

    print("\nReading final structure...")
    type_map = {elem: i + 1 for i, elem in enumerate(ELEMENTS)}
    inv_type_map = {v: k for k, v in type_map.items()}
    final = read_lammps_data("amorphous.data", inv_type_map)
    final.to(filename=OUTPUT_CIF)
    print(f"Saved: {OUTPUT_CIF}")

    counts = {}
    for site in final:
        s = str(site.specie)
        counts[s] = counts.get(s, 0) + 1
    total_mass_amu = sum(ATOMIC_MASSES[s] * n for s, n in counts.items())
    total_mass_g = total_mass_amu * 1.66054e-24
    vol_cm3 = final.lattice.volume * 1e-24
    density = total_mass_g / vol_cm3

    print(f"\nFinal composition: {counts}, total: {len(final)}")
    print(f"Final density: {density:.4f} g/cm3")
    print(f"Final cell: {final.lattice.a:.3f} x {final.lattice.b:.3f} x {final.lattice.c:.3f} A")

    print("\nDone.")


if __name__ == "__main__":
    main()
