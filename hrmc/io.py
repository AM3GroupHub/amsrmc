"""Reading G(r) data and writing extended-xyz frames."""
import numpy as np


def read_gr(path):
    """Two-column r, G(r) file (whitespace or comma separated, '#' comments)."""
    r, g = [], []
    for line in open(path):
        parts = line.replace(",", " ").split()
        if len(parts) < 2 or line.lstrip().startswith("#"):
            continue
        try:
            r.append(float(parts[0]))
            g.append(float(parts[1]))
        except ValueError:
            continue
    return np.array(r), np.array(g)


def write_xyz(path, positions, species, cell, step, mode="w"):
    lattice = " ".join(f"{v:.6f}" for v in np.asarray(cell).flatten())
    with open(path, mode) as f:
        f.write(f"{len(positions)}\n")
        f.write(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 step={step}\n')
        for s, p in zip(species, positions):
            f.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
