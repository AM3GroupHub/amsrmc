"""Periodic-boundary geometry and hard-sphere overlap repair."""
import numpy as np


def perpendicular_widths(cell):
    a, b, c = np.asarray(cell)
    v = abs(float(np.linalg.det(cell)))
    return np.array([v / np.linalg.norm(np.cross(b, c)),
                     v / np.linalg.norm(np.cross(c, a)),
                     v / np.linalg.norm(np.cross(a, b))])


def check_mic_safe(cell, r_max):
    half = 0.5 * float(perpendicular_widths(cell).min())
    if r_max >= half:
        raise ValueError(f"r_max={r_max:.3f} A must be below half the smallest "
                         f"perpendicular cell width ({half:.3f} A)")


def wrap(positions, cell, inv_cell):
    frac = np.asarray(positions) @ inv_cell
    frac -= np.floor(frac)
    return frac @ cell


def distances(pos, all_positions, cell, inv_cell):
    """Minimum-image distances from one point to many."""
    frac = (np.asarray(pos) - np.asarray(all_positions)) @ inv_cell
    frac -= np.round(frac)
    return np.linalg.norm(frac @ cell, axis=1)


def symmetric(table):
    out = {}
    for (a, b), v in table.items():
        out[(a, b)] = out[(b, a)] = v
    return out


def overlapping_atoms(positions, species, cutoffs, cell, inv_cell):
    cut = symmetric(cutoffs)
    n = len(positions)
    bad = set()
    for i in range(n):
        d = distances(positions[i], positions, cell, inv_cell)
        d[i] = np.inf
        lim = np.array([cut.get((species[i], species[j]), 0.0) for j in range(n)])
        hit = np.where(d < lim)[0]
        if hit.size:
            bad.add(i)
            bad.update(hit.tolist())
    return sorted(bad)


def repair_overlaps(positions, species, cutoffs, cell, inv_cell,
                    margin=1.02, max_iter=500):
    """Push overlapping pairs apart, half the deficit each, until none remain."""
    cut = symmetric(cutoffs)
    pos = np.array(positions, dtype=float, copy=True)
    n = len(pos)
    for it in range(1, max_iter + 1):
        shift = np.zeros_like(pos)
        n_bad = 0
        for i in range(n):
            delta = pos - pos[i]
            frac = delta @ inv_cell
            delta = (frac - np.round(frac)) @ cell
            d = np.linalg.norm(delta, axis=1)
            d[i] = np.inf
            lim = np.array([cut.get((species[i], species[j]), 0.0) for j in range(n)])
            for j in np.where(d < lim)[0]:
                n_bad += 1
                need = lim[j] * margin - d[j]
                u = delta[j] / max(d[j], 1e-9)
                shift[i] -= 0.5 * need * u
                shift[j] += 0.5 * need * u
        if n_bad == 0:
            return pos, it - 1
        pos = pos + shift
    return pos, max_iter
