import os
import re
import subprocess
import numpy as np
from pathlib import Path

# ---- Parameters ----

INPUT_DATA = "disordered.data"
MELT_TEMP = 1800.0  # K
TIMESTEP = 0.002  # ps (2 fs)
MELT_STEPS = 100000  # 200 ps
THERMO_INTERVAL = 100
SEED = 42

# ---- Paths ----

# this is used to switch between HPC and local paths --> bc i submit the job to vanda to run
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



# Element-to-type mapping (alphabetical, consistent across all scripts)
# Type 1: Cl, Type 2: Li, Type 3: Nb, Type 4: O
ELEMENTS = ["Cl", "Li", "Nb", "O"]


# this is a function to convert the MACE model to LAMMPS MLIAP format if not already done
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


def parse_msd_from_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()

    header_pattern = re.compile(r"^\s*Step\s+Temp\s+PotEng")
    end_pattern = re.compile(r"^\s*Loop time")

    data_lines = []
    in_thermo = False
    for line in lines:
        if header_pattern.match(line):
            in_thermo = True
            continue
        if in_thermo:
            if end_pattern.match(line):
                break
            parts = line.split()
            if len(parts) >= 10:
                try:
                    float(parts[0])
                    data_lines.append(parts)
                except ValueError:
                    continue

    if not data_lines:
        raise RuntimeError("No thermo data found in log file")

    data = np.array(data_lines, dtype=float)
    # Columns: step, temp, pe, ke, etotal, press, msd_li, msd_nb, msd_ox, msd_cl
    steps = data[:, 0]
    time_ps = steps * TIMESTEP
    msd_li = data[:, 6]
    msd_nb = data[:, 7]
    msd_ox = data[:, 8]
    msd_cl = data[:, 9]

    output = np.column_stack([time_ps, msd_li, msd_nb, msd_ox, msd_cl])
    header = "time_ps  MSD_Li  MSD_Nb  MSD_O  MSD_Cl"
    np.savetxt("msd_data.txt", output, header=header, fmt="%.6f")
    print(f"Saved: msd_data.txt ({len(output)} data points)")


def main():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        conda_lib = f"{conda_prefix}/lib"
        if not ld_path.startswith(conda_lib):
            os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"

    print("=== Melt test at {:.0f} K ===".format(MELT_TEMP))

    mace_pt = convert_mace_model(MACE_MODEL)

    elem_str = " ".join(ELEMENTS)
    melt_script = f"""units           metal
atom_style      atomic
newton          on
boundary        p p p
read_data       {INPUT_DATA}

pair_style      mliap unified {mace_pt} 0
pair_coeff      * * {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Groups: type 1=Cl, 2=Li, 3=Nb, 4=O
group           cl type 1
group           li type 2
group           nb type 3
group           ox type 4

compute         msd_li li msd
compute         msd_nb nb msd
compute         msd_ox ox msd
compute         msd_cl cl msd

timestep        {TIMESTEP}
thermo          {THERMO_INTERVAL}
thermo_style    custom step temp pe ke etotal press c_msd_li[4] c_msd_nb[4] c_msd_ox[4] c_msd_cl[4]

velocity        all create {MELT_TEMP} {SEED} dist gaussian
fix             1 all nvt temp {MELT_TEMP} {MELT_TEMP} 0.1
run             {MELT_STEPS}
unfix           1

write_data      melted.data
"""
    with open("melt_test.in", "w") as f:
        f.write(melt_script)
    print("Saved: melt_test.in")

    run_lammps(LMP_BINARY, "melt_test.in")

    print("\nParsing MSD from log...")
    parse_msd_from_log("lammps.log")

    print("\nDone.")


if __name__ == "__main__":
    main()
