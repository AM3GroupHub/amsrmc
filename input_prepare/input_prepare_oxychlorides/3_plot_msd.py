import re
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = "lammps_melt.log"
MSD_FILE = "msd_data.txt"
OUTPUT_PDF = "msd_melt_test.pdf"
TIMESTEP = 0.002  # ps


def parse_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    header_re = re.compile(r"^\s*Step\s+Temp\s+PotEng")
    end_re = re.compile(r"^\s*(Loop time|WARNING)")
    data_lines = []
    in_thermo = False
    for line in lines:
        if header_re.match(line):
            in_thermo = True
            continue
        if in_thermo:
            if end_re.match(line) or not line.strip():
                break
            parts = line.split()
            if len(parts) >= 10:
                try:
                    float(parts[0])
                    data_lines.append(parts[:10])
                except ValueError:
                    continue
    data = np.array(data_lines, dtype=float)
    time_ps = data[:, 0] * TIMESTEP
    output = np.column_stack([time_ps, data[:, 6], data[:, 7], data[:, 8], data[:, 9]])
    np.savetxt(MSD_FILE, output, header="time_ps  MSD_Li  MSD_Nb  MSD_O  MSD_Cl", fmt="%.6f")
    print(f"Parsed {len(output)} data points from {log_file}")
    return output


def main():
    data = parse_log(LOG_FILE)
    time_ps = data[:, 0]
    msd_li = data[:, 1]
    msd_nb = data[:, 2]
    msd_o = data[:, 3]
    msd_cl = data[:, 4]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(time_ps, msd_li, label="Li", linewidth=2)
    ax.plot(time_ps, msd_nb, label="Nb", linewidth=2)
    ax.plot(time_ps, msd_o, label="O", linewidth=2)
    ax.plot(time_ps, msd_cl, label="Cl", linewidth=2)

    ax.set_xlabel("Time [ps]", fontsize=18)
    ax.set_ylabel("MSD [A$^2$]", fontsize=18)
    ax.set_title("Melt Test: MSD vs Time", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")

    print("\nInterpretation:")
    print("  Linear MSD growth = diffusive motion = system is melted")
    print("  Plateau or sub-linear = atoms are trapped = not fully melted")
    for label, msd in [("Li", msd_li), ("Nb", msd_nb), ("O", msd_o), ("Cl", msd_cl)]:
        final_msd = msd[-1]
        print(f"  {label}: final MSD = {final_msd:.1f} A^2")


if __name__ == "__main__":
    main()
