"""Hard-sphere and coordination-number constraints."""
import numpy as np
from .geometry import distances


class HardSphere:
    """Rejects a move that brings an atom closer than its pair cutoff."""

    def __init__(self, cutoffs, species, cell, inv_cell):
        self.cut = {tuple(sorted(k)): v for k, v in cutoffs.items()}
        self.cell, self.inv_cell = cell, inv_cell
        self.elements = sorted(np.unique(species))
        self.index = {e: np.where(species == e)[0] for e in self.elements}

    def violated(self, i, new, positions, species):
        for e in self.elements:
            c = self.cut.get(tuple(sorted([species[i], e])), 0.0)
            if c <= 0:
                continue
            idx = self.index[e]
            d = distances(new, positions[idx], self.cell, self.inv_cell)
            if np.any(d[idx != i] < c):
                return True
        return False


class Coordination:
    """Penalises the distance between the CN histogram and a target distribution.

    cutoffs: {(centre, neighbour): r}; targets: {(centre, neighbour): [(cn, fraction), ...]}
    or None to take the distribution of the given positions.
    """

    def __init__(self, cutoffs, positions, species, cell, inv_cell,
                 targets=None, sigma=0.05):
        self.sigma, self.cell, self.inv_cell = sigma, cell, inv_cell
        self.cutoffs = cutoffs
        self.pairs = {}
        for (c, n), r in cutoffs.items():
            ic, inb = np.where(species == c)[0], np.where(species == n)[0]
            cn = np.zeros(len(ic), dtype=int)
            for k, i in enumerate(ic):
                d = distances(positions[i], positions[inb], cell, inv_cell)
                if c == n:
                    d[inb == i] = np.inf
                cn[k] = np.sum(d < r)
            t = targets[(c, n)] if targets and (c, n) in targets else self.distribution(cn)
            self.pairs[(c, n)] = dict(ic=ic, inb=inb, r=r, cn=cn, targets=t)

    @staticmethod
    def distribution(cn):
        """Observed fractions over 0..max+2, so new CN values are penalised."""
        u, k = np.unique(cn, return_counts=True)
        obs = {int(a): int(b) for a, b in zip(u, k)}
        return [(c, obs.get(c, 0) / len(cn)) for c in range(max(obs) + 3)]

    def targets(self):
        return {k: list(v["targets"]) for k, v in self.pairs.items()}

    def delta(self, i, old, new, positions, species):
        e = species[i]
        out = {}
        for (c, n), p in self.pairs.items():
            dcn = {}
            if e == n:
                d0 = distances(old, positions[p["ic"]], self.cell, self.inv_cell)
                d1 = distances(new, positions[p["ic"]], self.cell, self.inv_cell)
                if c == n:
                    d0[p["ic"] == i] = d1[p["ic"] == i] = np.inf
                w0, w1 = d0 < p["r"], d1 < p["r"]
                for k in np.where(~w0 & w1)[0]:
                    dcn[k] = dcn.get(k, 0) + 1
                for k in np.where(w0 & ~w1)[0]:
                    dcn[k] = dcn.get(k, 0) - 1
            if e == c:
                k = np.where(p["ic"] == i)[0][0]
                d0 = distances(old, positions[p["inb"]], self.cell, self.inv_cell)
                d1 = distances(new, positions[p["inb"]], self.cell, self.inv_cell)
                if c == n:
                    d0[p["inb"] == i] = d1[p["inb"] == i] = np.inf
                change = int(np.sum(d1 < p["r"])) - int(np.sum(d0 < p["r"]))
                if change:
                    dcn[k] = dcn.get(k, 0) + change
            if dcn:
                out[(c, n)] = dcn
        return out

    def chi2(self, changes=None):
        x = 0.0
        for key, p in self.pairs.items():
            cn = p["cn"]
            if changes and key in changes:
                cn = cn.copy()
                for k, v in changes[key].items():
                    cn[k] += v
            for c, f in p["targets"]:
                x += ((np.sum(cn == c) / len(cn) - f) / self.sigma) ** 2
        return x

    def apply(self, changes):
        for key, dcn in changes.items():
            for k, v in dcn.items():
                self.pairs[key]["cn"][k] += v
