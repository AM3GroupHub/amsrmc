import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations_with_replacement
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

CIF_FILE = "melt_quench_step4/amorphous_LiNbOCl.cif"
OUTPUT_DIR = Path("analysis_output")

CENTERS = ["Li", "Nb"]
ANIONS = ["O", "Cl"]
BOND_PAIRS = [("Li", "O"), ("Li", "Cl"), ("Nb", "O"), ("Nb", "Cl")]
ANGLE_TRIPLETS = [
    ("O", "Li", "O"), ("Cl", "Li", "Cl"), ("O", "Li", "Cl"),
    ("O", "Nb", "O"), ("Cl", "Nb", "Cl"), ("O", "Nb", "Cl"),
]


# ============================================================
# PBC utilities (from utilities.py)
# ============================================================

def minimum_image(delta, box_lengths):
    return delta - box_lengths * np.round(delta / box_lengths)


def distance_pbc_vectorized(pos, all_positions, box_lengths):
    delta = minimum_image(pos - all_positions, box_lengths)
    return np.linalg.norm(delta, axis=1)


# ============================================================
# Structure loading
# ============================================================

def load_structure(cif_file):
    from ase.io import read
    atoms = read(cif_file)
    positions = atoms.get_positions().copy()
    species = np.array(atoms.get_chemical_symbols())
    cell = atoms.get_cell()
    box_lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
    return positions, species, box_lengths


# ============================================================
# g(r) computation and peak finding
# ============================================================

def compute_partial_gr(positions, species, box_lengths, pair, r_min, r_max, n_bins):
    elem_a, elem_b = pair
    idx_a = np.where(species == elem_a)[0]
    idx_b = np.where(species == elem_b)[0]
    n_a, n_b = len(idx_a), len(idx_b)
    n_total = len(positions)
    box_volume = np.prod(box_lengths)
    dr = (r_max - r_min) / n_bins
    r_bins = np.linspace(r_min + dr / 2, r_max - dr / 2, n_bins)

    histogram = np.zeros(n_bins)
    for i in idx_a:
        dists = distance_pbc_vectorized(positions[i], positions[idx_b], box_lengths)
        if elem_a == elem_b:
            dists[idx_b == i] = np.inf
        mask = (dists >= r_min) & (dists < r_max)
        bin_indices = ((dists[mask] - r_min) / dr).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(histogram, bin_indices, 1)

    n_pairs = n_a * (n_a - 1) if elem_a == elem_b else n_a * n_b
    rho0 = n_total / box_volume
    gr = np.zeros(n_bins)
    for i, r in enumerate(r_bins):
        shell_vol = 4.0 * np.pi * r ** 2 * dr
        ideal = rho0 * shell_vol * n_pairs / n_total
        if ideal > 0:
            gr[i] = histogram[i] / ideal
    return r_bins, gr


def find_first_peak_and_minimum(r, gr, sigma=3):
    gr_smooth = gaussian_filter1d(gr, sigma=sigma)
    peaks, _ = find_peaks(gr_smooth, height=0.1)
    if len(peaks) == 0:
        return None, None

    first_peak_idx = peaks[0]
    first_peak_r = r[first_peak_idx]

    search_region = gr_smooth[first_peak_idx:]
    minima, _ = find_peaks(-search_region)
    first_min_r = r[-1]
    for m in minima:
        m_global = first_peak_idx + m
        if gr_smooth[m_global] < 1.0:
            first_min_r = r[m_global]
            break

    return first_peak_r, first_min_r


# ============================================================
# Coordination numbers
# ============================================================

def compute_coordination(positions, species, box_lengths, center_elem, ligand_elem, cutoff):
    idx_center = np.where(species == center_elem)[0]
    idx_ligand = np.where(species == ligand_elem)[0]
    cn_list = []
    for i in idx_center:
        dists = distance_pbc_vectorized(positions[i], positions[idx_ligand], box_lengths)
        cn = np.sum(dists < cutoff)
        cn_list.append(cn)
    return np.array(cn_list)


def compute_total_coordination(positions, species, box_lengths, center_elem, ligand_cutoffs):
    idx_center = np.where(species == center_elem)[0]
    cn_list = np.zeros(len(idx_center), dtype=int)
    for ligand_elem, cutoff in ligand_cutoffs.items():
        idx_ligand = np.where(species == ligand_elem)[0]
        for k, i in enumerate(idx_center):
            dists = distance_pbc_vectorized(positions[i], positions[idx_ligand], box_lengths)
            cn_list[k] += np.sum(dists < cutoff)
    return cn_list


# ============================================================
# Bond distances within cutoff
# ============================================================

def collect_bond_distances(positions, species, box_lengths, center_elem, ligand_elem, cutoff):
    idx_center = np.where(species == center_elem)[0]
    idx_ligand = np.where(species == ligand_elem)[0]
    distances = []
    for i in idx_center:
        dists = distance_pbc_vectorized(positions[i], positions[idx_ligand], box_lengths)
        bond_dists = dists[dists < cutoff]
        distances.extend(bond_dists.tolist())
    return np.array(distances)


# ============================================================
# Bond angle distributions
# ============================================================

def build_neighbor_lists(positions, species, box_lengths, cn_cutoffs):
    neighbors = defaultdict(list)
    n = len(positions)
    for center_elem in CENTERS:
        idx_center = np.where(species == center_elem)[0]
        for anion in ANIONS:
            key = (center_elem, anion)
            if key not in cn_cutoffs:
                continue
            cutoff = cn_cutoffs[key]
            idx_ligand = np.where(species == anion)[0]
            for i in idx_center:
                dists = distance_pbc_vectorized(positions[i], positions[idx_ligand], box_lengths)
                for j_local, d in enumerate(dists):
                    if d < cutoff:
                        j_global = idx_ligand[j_local]
                        neighbors[i].append(j_global)
    return neighbors


def compute_angle_distribution(positions, species, box_lengths, neighbors, triplet):
    elem_a, elem_center, elem_b = triplet
    angles = []
    idx_center = np.where(species == elem_center)[0]

    for center_idx in idx_center:
        nbrs = neighbors.get(center_idx, [])
        nbrs_a = [n for n in nbrs if species[n] == elem_a]
        nbrs_b = [n for n in nbrs if species[n] == elem_b]
        if elem_a == elem_b:
            for ii in range(len(nbrs_a)):
                for jj in range(ii + 1, len(nbrs_a)):
                    vec_a = minimum_image(positions[nbrs_a[ii]] - positions[center_idx], box_lengths)
                    vec_b = minimum_image(positions[nbrs_a[jj]] - positions[center_idx], box_lengths)
                    norm_prod = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
                    if norm_prod < 1e-12:
                        continue
                    cos_angle = np.clip(np.dot(vec_a, vec_b) / norm_prod, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cos_angle)))
        else:
            for ia in nbrs_a:
                for ib in nbrs_b:
                    vec_a = minimum_image(positions[ia] - positions[center_idx], box_lengths)
                    vec_b = minimum_image(positions[ib] - positions[center_idx], box_lengths)
                    norm_prod = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
                    if norm_prod < 1e-12:
                        continue
                    cos_angle = np.clip(np.dot(vec_a, vec_b) / norm_prod, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cos_angle)))
    return np.array(angles)


# ============================================================
# Plotting
# ============================================================

def plot_coordination_numbers(cn_data, total_cn_data, output_dir):
    n_panels = len(cn_data) + len(total_cn_data)
    n_cols = 3
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    idx = 0
    for (center, ligand), cn_arr in sorted(cn_data.items()):
        ax = axes[idx]
        bins = np.arange(cn_arr.min() - 0.5, cn_arr.max() + 1.5, 1)
        ax.hist(cn_arr, bins=bins, density=True, color='steelblue', edgecolor='black', alpha=0.8)
        mean_cn = cn_arr.mean()
        std_cn = cn_arr.std()
        ax.axvline(mean_cn, color='red', linestyle='--', linewidth=2,
                   label=f'mean={mean_cn:.2f}+/-{std_cn:.2f}')
        ax.set_xlabel(f'CN({center}-{ligand})', fontsize=22)
        ax.set_ylabel('Fraction', fontsize=22)
        ax.set_title(f'{center}-{ligand} coordination', fontsize=24)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
        idx += 1

    for center, cn_arr in sorted(total_cn_data.items()):
        ax = axes[idx]
        bins = np.arange(cn_arr.min() - 0.5, cn_arr.max() + 1.5, 1)
        ax.hist(cn_arr, bins=bins, density=True, color='coral', edgecolor='black', alpha=0.8)
        mean_cn = cn_arr.mean()
        std_cn = cn_arr.std()
        ax.axvline(mean_cn, color='red', linestyle='--', linewidth=2,
                   label=f'mean={mean_cn:.2f}+/-{std_cn:.2f}')
        ligand_str = "+".join(ANIONS)
        ax.set_xlabel(f'CN({center}-[{ligand_str}])', fontsize=22)
        ax.set_ylabel('Fraction', fontsize=22)
        ax.set_title(f'{center} total coordination ({ligand_str})', fontsize=24)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
        idx += 1

    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "coordination_numbers.pdf")
    plt.close()
    print("  Saved coordination_numbers.pdf")


def plot_bond_distances(dist_data, output_dir):
    n_panels = len(dist_data)
    n_cols = min(n_panels, 2)
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.atleast_2d(axes).flatten()

    for idx, ((center, ligand), dists) in enumerate(sorted(dist_data.items())):
        ax = axes[idx]
        if len(dists) == 0:
            ax.text(0.5, 0.5, 'No bonds found', transform=ax.transAxes,
                    ha='center', fontsize=20)
            ax.set_title(f'{center}-{ligand}', fontsize=24)
            continue
        bins = np.linspace(dists.min() - 0.05, dists.max() + 0.05, 60)
        ax.hist(dists, bins=bins, density=True, color='steelblue', edgecolor='black', alpha=0.8)
        mean_d = dists.mean()
        ax.axvline(mean_d, color='red', linestyle='--', linewidth=2,
                   label=f'mean={mean_d:.3f} A')
        ax.set_xlabel(r'Bond length [$\mathrm{\AA}$]', fontsize=22)
        ax.set_ylabel('Probability density', fontsize=22)
        ax.set_title(f'{center}-{ligand} bond distance', fontsize=24)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=18)

    for i in range(len(dist_data), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "bond_distances.pdf")
    plt.close()
    print("  Saved bond_distances.pdf")


def plot_bond_angles(angle_data, output_dir):
    valid = {k: v for k, v in angle_data.items() if len(v) > 0}
    n_panels = len(valid)
    if n_panels == 0:
        print("  No angle data to plot.")
        return
    n_cols = min(n_panels, 3)
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.atleast_2d(axes).flatten()

    for idx, (triplet, angs) in enumerate(sorted(valid.items())):
        ax = axes[idx]
        bins = np.linspace(0, 180, 60)
        ax.hist(angs, bins=bins, density=True, color='steelblue', edgecolor='black', alpha=0.8)
        mean_a = angs.mean()
        ax.axvline(mean_a, color='red', linestyle='--', linewidth=2,
                   label=f'mean={mean_a:.1f} deg')
        ax.axvline(109.5, color='green', linestyle=':', linewidth=1.5, label='Tetrahedral (109.5)')
        ax.axvline(90.0, color='orange', linestyle=':', linewidth=1.5, label='Octahedral (90)')
        label = f'{triplet[0]}-{triplet[1]}-{triplet[2]}'
        ax.set_xlabel(r'Angle [$\degree$]', fontsize=22)
        ax.set_ylabel('Probability density', fontsize=22)
        ax.set_title(f'{label} angle', fontsize=24)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=14)

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "bond_angles.pdf")
    plt.close()
    print("  Saved bond_angles.pdf")


# ============================================================
# Text summary
# ============================================================

def write_summary(cn_data, total_cn_data, dist_data, angle_data, cn_cutoffs, output_dir):
    lines = []
    lines.append("=" * 60)
    lines.append("Amorphous LiNbOCl Structure Analysis Summary")
    lines.append("=" * 60)
    lines.append("")

    lines.append("--- CN Cutoffs (first minimum of g(r)) ---")
    for (center, ligand), cutoff in sorted(cn_cutoffs.items()):
        lines.append(f"  {center}-{ligand}: {cutoff:.2f} A")
    lines.append("")

    lines.append("--- Coordination Numbers ---")
    for (center, ligand), cn_arr in sorted(cn_data.items()):
        lines.append(f"  CN({center}-{ligand}): mean={cn_arr.mean():.2f} +/- {cn_arr.std():.2f} "
                     f"(min={cn_arr.min()}, max={cn_arr.max()})")
    lines.append("")

    lines.append("--- Total Coordination Numbers ---")
    for center, cn_arr in sorted(total_cn_data.items()):
        ligand_str = "+".join(ANIONS)
        lines.append(f"  CN({center}-[{ligand_str}]): mean={cn_arr.mean():.2f} +/- {cn_arr.std():.2f} "
                     f"(min={cn_arr.min()}, max={cn_arr.max()})")
    lines.append("")

    lines.append("--- Bond Distances ---")
    for (center, ligand), dists in sorted(dist_data.items()):
        if len(dists) > 0:
            lines.append(f"  {center}-{ligand}: mean={dists.mean():.3f} +/- {dists.std():.3f} A "
                         f"(n={len(dists)})")
    lines.append("")

    lines.append("--- Bond Angle Distributions ---")
    for triplet, angs in sorted(angle_data.items()):
        if len(angs) > 0:
            label = f"{triplet[0]}-{triplet[1]}-{triplet[2]}"
            lines.append(f"  {label}: mean={angs.mean():.1f} +/- {angs.std():.1f} deg "
                         f"(n={len(angs)})")
    lines.append("")

    summary = "\n".join(lines)
    with open(output_dir / "analysis_summary.txt", 'w') as f:
        f.write(summary)
    print("  Saved analysis_summary.txt")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading structure from {CIF_FILE}...")
    positions, species, box_lengths = load_structure(CIF_FILE)
    n_atoms = len(positions)
    elements = sorted(np.unique(species))
    print(f"  {n_atoms} atoms, box = {box_lengths[0]:.2f} x {box_lengths[1]:.2f} x {box_lengths[2]:.2f} A")
    for elem in elements:
        print(f"  {elem}: {np.sum(species == elem)}")

    r_min = 0.5
    r_max_safe = np.min(box_lengths) / 2 * 0.9
    r_max = min(10.0, r_max_safe)
    n_bins = 500

    # Step 1: Compute g(r) for bond pairs and determine CN cutoffs
    print("\nComputing partial g(r) for bond pairs...")
    cn_cutoffs = {}
    for pair in BOND_PAIRS:
        r, gr = compute_partial_gr(positions, species, box_lengths, pair, r_min, r_max, n_bins)
        peak_r, min_r = find_first_peak_and_minimum(r, gr)
        if peak_r is not None and min_r is not None:
            cn_cutoffs[pair] = min_r
            print(f"  {pair[0]}-{pair[1]}: peak={peak_r:.2f} A, cutoff={min_r:.2f} A")
        else:
            print(f"  {pair[0]}-{pair[1]}: no clear peak found, skipping")

    # Step 2: Coordination numbers
    print("\nComputing coordination numbers...")
    cn_data = {}
    for pair in BOND_PAIRS:
        if pair not in cn_cutoffs:
            continue
        cn = compute_coordination(positions, species, box_lengths, pair[0], pair[1], cn_cutoffs[pair])
        cn_data[pair] = cn
        print(f"  CN({pair[0]}-{pair[1]}): mean={cn.mean():.2f} +/- {cn.std():.2f}")

    total_cn_data = {}
    for center in CENTERS:
        ligand_cutoffs = {}
        for anion in ANIONS:
            key = (center, anion)
            if key in cn_cutoffs:
                ligand_cutoffs[anion] = cn_cutoffs[key]
        if ligand_cutoffs:
            cn = compute_total_coordination(positions, species, box_lengths, center, ligand_cutoffs)
            total_cn_data[center] = cn
            print(f"  CN({center}-[{'+'.join(ANIONS)}]): mean={cn.mean():.2f} +/- {cn.std():.2f}")

    plot_coordination_numbers(cn_data, total_cn_data, OUTPUT_DIR)

    # Step 3: Bond distance distributions
    print("\nComputing bond distance distributions...")
    dist_data = {}
    for pair in BOND_PAIRS:
        if pair not in cn_cutoffs:
            continue
        dists = collect_bond_distances(positions, species, box_lengths, pair[0], pair[1], cn_cutoffs[pair])
        dist_data[pair] = dists
        if len(dists) > 0:
            print(f"  {pair[0]}-{pair[1]}: mean={dists.mean():.3f} A, n={len(dists)}")

    plot_bond_distances(dist_data, OUTPUT_DIR)

    # Step 4: Bond angle distributions
    print("\nBuilding neighbor lists for angle computation...")
    neighbors = build_neighbor_lists(positions, species, box_lengths, cn_cutoffs)

    print("Computing bond angle distributions...")
    angle_data = {}
    for triplet in ANGLE_TRIPLETS:
        angs = compute_angle_distribution(positions, species, box_lengths, neighbors, triplet)
        angle_data[triplet] = angs
        if len(angs) > 0:
            label = f"{triplet[0]}-{triplet[1]}-{triplet[2]}"
            print(f"  {label}: mean={angs.mean():.1f} +/- {angs.std():.1f} deg (n={len(angs)})")

    plot_bond_angles(angle_data, OUTPUT_DIR)

    # Step 5: Write summary
    print("\nWriting summary...")
    summary = write_summary(cn_data, total_cn_data, dist_data, angle_data, cn_cutoffs, OUTPUT_DIR)
    print("\n" + summary)

    print(f"\nAll output saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
