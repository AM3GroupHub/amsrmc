import os
import re
import subprocess
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element

# ---- Parameters ----

INPUT_CIF = "LiVOF4.cif"
SUPERCELL = [9, 3, 3]
TARGET_DENSITY = 1.636  # g/cm3
LI_REMOVE_FRAC = 0.10
O_TO_CL_FRAC = 0.10

TIMESTEP = 0.002  # ps (2 fs)
TEMPERATURE = 300.0
NPT_STEPS = 1000
TDAMP = 0.1
PDAMP = 1.0
PRESSURE = 0.0
THERMO_INTERVAL = 100
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
    "Cl": 35.453, "F": 18.998, "Li": 6.941, "Nb": 92.906,
    "O": 15.999, "V": 50.942, "H": 1.008, "C": 12.011,
    "N": 14.007, "S": 32.065, "P": 30.974, "Na": 22.990,
    "K": 39.098, "Ca": 40.078, "Si": 28.086, "Al": 26.982,
}


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


def get_element_types(structure):
    elements = sorted(set(str(site.specie) for site in structure))
    type_map = {elem: i + 1 for i, elem in enumerate(elements)}
    return elements, type_map


def write_lammps_data(structure, filename, type_map):
    cart = structure.cart_coords
    lattice = structure.lattice
    with open(filename, "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{len(structure)} atoms\n")
        f.write(f"{len(type_map)} atom types\n\n")
        f.write(f"0.0 {lattice.a:.6f} xlo xhi\n")
        f.write(f"0.0 {lattice.b:.6f} ylo yhi\n")
        f.write(f"0.0 {lattice.c:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for elem, tid in sorted(type_map.items(), key=lambda x: x[1]):
            mass = ATOMIC_MASSES.get(elem, 1.0)
            f.write(f"{tid} {mass:.3f}  # {elem}\n")
        f.write("\nAtoms # atomic\n\n")
        for i, site in enumerate(structure):
            tid = type_map[str(site.specie)]
            x, y, z = cart[i]
            f.write(f"{i + 1} {tid} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved: {filename}")


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


def build_substituted_supercell(cif_path, supercell):
    stru = Structure.from_file(cif_path)
    print(f"Original unit cell: {len(stru)} atoms")
    print(f"  Cell: {stru.lattice.a:.3f} x {stru.lattice.b:.3f} x {stru.lattice.c:.3f} A")

    sub_map = {"V": "Nb", "F": "Cl"}
    new_species = []
    for site in stru:
        elem = str(site.specie)
        if elem in sub_map:
            new_species.append(sub_map[elem])
        elif elem.rstrip("+-0123456789") in sub_map:
            clean = elem.rstrip("+-0123456789")
            new_species.append(sub_map[clean])
        else:
            clean = elem.rstrip("+-0123456789")
            new_species.append(clean)

    stru_clean = Structure(stru.lattice, new_species, stru.frac_coords)
    stru_clean.to(filename="LiNbOCl4_unitcell.cif")
    print(f"Saved unit cell: LiNbOCl4_unitcell.cif")

    stru_clean.make_supercell(supercell)
    stru_clean.to(filename="1_ordered.cif")
    print(f"Saved: 1_ordered.cif (supercell {supercell}, {len(stru_clean)} atoms)")
    print(f"  Cell: {stru_clean.lattice.a:.3f} x {stru_clean.lattice.b:.3f} x {stru_clean.lattice.c:.3f} A")
    report_composition(stru_clean)
    return stru_clean


def apply_disorder(structure, li_remove_frac, o_to_cl_frac, seed=42):
    rng = np.random.default_rng(seed)

    li_indices = [i for i, site in enumerate(structure) if str(site.specie) == "Li"]
    n_remove = int(round(len(li_indices) * li_remove_frac)) # randomly remove the Li
    remove_indices = rng.choice(li_indices, size=n_remove, replace=False)
    print(f"Removing {n_remove} Li atoms (from {len(li_indices)})")

    o_indices = [i for i, site in enumerate(structure) if str(site.specie) == "O"] 
    n_replace = int(round(len(o_indices) * o_to_cl_frac))
    replace_indices = rng.choice(o_indices, size=n_replace, replace=False)
    print(f"Replacing {n_replace} O atoms with Cl (from {len(o_indices)})")

    new_species = []
    new_coords = []
    for i, site in enumerate(structure):# randomly select O to replace with Cl
        if i in set(remove_indices):
            continue
        elem = str(site.specie)
        if i in set(replace_indices):
            elem = "Cl"
        new_species.append(elem)
        new_coords.append(site.frac_coords)

    disordered = Structure(structure.lattice, new_species, new_coords)
    print(f"After disorder: {len(disordered)} atoms")
    report_composition(disordered)
    return disordered


def apply_density_strain(structure, target_density):
    species_list = [str(site.specie) for site in structure]
    total_mass_amu = sum(ATOMIC_MASSES[s] for s in species_list)
    total_mass_g = total_mass_amu * 1.66054e-24

    target_vol_cm3 = total_mass_g / target_density
    target_vol_A3 = target_vol_cm3 * 1e24

    current_vol = structure.lattice.volume
    scale_factor = (target_vol_A3 / current_vol) ** (1.0 / 3.0)
    print(f"Current volume: {current_vol:.1f} A^3, target: {target_vol_A3:.1f} A^3")
    print(f"Scale factor: {scale_factor:.6f}")

    new_lattice = Lattice(structure.lattice.matrix * scale_factor)
    strained = Structure(new_lattice, [str(s.specie) for s in structure],
                         structure.frac_coords)

    actual_density = total_mass_g / (strained.lattice.volume * 1e-24)
    print(f"Final density: {actual_density:.4f} g/cm3 (target: {target_density:.4f})")
    print(f"Final cell: {strained.lattice.a:.3f} x {strained.lattice.b:.3f} x {strained.lattice.c:.3f} A")
    return strained


def report_composition(structure):
    counts = {}
    for site in structure:
        s = str(site.specie)
        counts[s] = counts.get(s, 0) + 1
    vol_A3 = structure.lattice.volume
    print(f"  Composition: {counts}, total: {len(structure)}")
    print(f"  Number densities (atoms/A^3):")
    for elem in sorted(counts.keys()):
        print(f"    {elem}: {counts[elem] / vol_A3:.6f}")


def main():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        conda_lib = f"{conda_prefix}/lib"
        if not ld_path.startswith(conda_lib):
            os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"

    np.random.seed(SEED)

    print("=== Step 1: Build substituted supercell ===")
    supercell = build_substituted_supercell(INPUT_CIF, SUPERCELL)

    print("\n=== Step 2: Convert MACE model ===")
    mace_pt = convert_mace_model(MACE_MODEL)

    elements, type_map = get_element_types(supercell)
    inv_type_map = {v: k for k, v in type_map.items()}
    print(f"Element mapping: {type_map}")

    print("\n=== Step 3: NPT relaxation ===")
    write_lammps_data(supercell, "supercell.data", type_map)

    elem_str = " ".join(elements)
    npt_script = f"""units           metal
atom_style      atomic
newton          on
boundary        p p p
read_data       supercell.data

pair_style      mliap unified {mace_pt} 0
pair_coeff      * * {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        {TIMESTEP}
thermo          {THERMO_INTERVAL}
thermo_style    custom step temp pe ke etotal press vol lx ly lz

velocity        all create {TEMPERATURE} {SEED} dist gaussian
fix             1 all npt temp {TEMPERATURE} {TEMPERATURE} {TDAMP} iso {PRESSURE} {PRESSURE} {PDAMP}
run             {NPT_STEPS}
unfix           1

write_data      relaxed.data
"""
    with open("npt_relax.in", "w") as f:
        f.write(npt_script)
    print("Saved: npt_relax.in")

    run_lammps(LMP_BINARY, "npt_relax.in")

    print("\n=== Step 4: Read relaxed structure ===")
    relaxed = read_lammps_data("relaxed.data", inv_type_map)

    print("\n=== Step 5: Apply disorder ===")
    disordered = apply_disorder(relaxed, LI_REMOVE_FRAC, O_TO_CL_FRAC, seed=SEED)

    print("\n=== Step 6: Apply density strain ===")
    final = apply_density_strain(disordered, TARGET_DENSITY)
    report_composition(final)

    elements_final, type_map_final = get_element_types(final)
    write_lammps_data(final, "disordered.data", type_map_final)
    final.to(filename="2_substituted.cif")
    print("Saved: 2_substituted.cif")

    print("\nDone.")


if __name__ == "__main__":
    main()
