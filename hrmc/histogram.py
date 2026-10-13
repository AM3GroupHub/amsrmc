"""Partial pair histograms with single-atom incremental updates; neutron G(r)."""
import numpy as np
from itertools import combinations_with_replacement
from .geometry import distances


class Histograms:

    def __init__(self, r_min, r_max, n_bins, positions, species, cell, b):
        self.r_min, self.r_max, self.n_bins = r_min, r_max, n_bins
        self.dr = (r_max - r_min) / n_bins
        self.r = np.linspace(r_min + self.dr / 2, r_max - self.dr / 2, n_bins)
        self.cell = np.asarray(cell, dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)
        n = len(positions)
        rho0 = n / abs(float(np.linalg.det(self.cell)))
        self.elements = sorted(np.unique(species))
        self.pairs = list(combinations_with_replacement(self.elements, 2))
        self.index = {e: np.where(species == e)[0] for e in self.elements}
        count = {e: len(self.index[e]) for e in self.elements}
        conc = {e: count[e] / n for e in self.elements}
        self.key = {a: {c: tuple(sorted([a, c])) for c in self.elements}
                    for a in self.elements}

        shell = 4.0 * np.pi * self.r ** 2 * self.dr
        avg_b_sq = sum(conc[e] * b[e] for e in self.elements) ** 2
        rows = []
        for p, q in self.pairs:
            n_pairs = count[p] * (count[p] - 1) if p == q else count[p] * count[q]
            ideal = rho0 * shell * n_pairs / n
            norm = np.where(ideal > 0, 1.0 / ideal, 0.0)
            w = conc[p] * conc[q] * b[p] * b[q]
            if p != q:
                w *= 2
            rows.append((w / avg_b_sq) * norm)
        self.coeff = np.array(rows)
        self.prefactor = 4.0 * np.pi * self.r * rho0
        self.full(positions)

    def _bin(self, d):
        m = (d >= self.r_min) & (d < self.r_max)
        return np.clip(((d[m] - self.r_min) / self.dr).astype(int), 0, self.n_bins - 1)

    def full(self, positions):
        self.h = {p: np.zeros(self.n_bins) for p in self.pairs}
        for p, q in self.pairs:
            iq = self.index[q]
            for i in self.index[p]:
                d = distances(positions[i], positions[iq], self.cell, self.inv_cell)
                if p == q:
                    d[iq == i] = np.inf
                np.add.at(self.h[(p, q)], self._bin(d), 1)

    def delta(self, i, old, new, positions, species):
        """Histogram changes for moving atom i from old to new."""
        d_old = distances(old, positions, self.cell, self.inv_cell)
        d_new = distances(new, positions, self.cell, self.inv_cell)
        d_old[i] = d_new[i] = np.inf
        e = species[i]
        out = {}
        for other in self.elements:
            idx = self.index[other]
            dh = (np.bincount(self._bin(d_new[idx]), minlength=self.n_bins)[:self.n_bins]
                  - np.bincount(self._bin(d_old[idx]), minlength=self.n_bins)[:self.n_bins]
                  ).astype(float)
            if other == e:
                dh *= 2
            out[self.key[e][other]] = dh
        return out

    def apply(self, changes):
        for p, dh in changes.items():
            self.h[p] += dh

    def G(self, changes=None):
        """Neutron-weighted total G(r) = 4 pi r rho0 (g(r) - 1)."""
        stack = np.array([self.h[p] + (changes[p] if changes and p in changes else 0)
                          for p in self.pairs])
        return self.prefactor * (np.einsum('ij,ij->j', self.coeff, stack) - 1.0)
