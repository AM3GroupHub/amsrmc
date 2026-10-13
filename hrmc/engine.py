"""The Monte Carlo walker shared by pure RMC, inline HRMC and alternating HRMC.

Acceptance: log_alpha = -dX2 / (2 sigma^2) - dE / (kB T), with
X2 = sum_n [G(r_n) - G_exp(r_n)]^2 (+ w * chi2_CN). Moves are single-atom displacements.
"""
import time
import numpy as np
from pathlib import Path
from .geometry import wrap, check_mic_safe, overlapping_atoms, repair_overlaps
from .histogram import Histograms
from .constraints import HardSphere, Coordination
from .io import write_xyz

KB_EV = 8.617333262e-5
TRACE_HEADER = "step,X2,E_per_atom,acc_frac,hs_frac,n_fire\n"
FIRE_HEADER = ("cycle,step,X2_pre,X2_post,E_pre_eVat,E_post_eVat,"
               "fire_steps,rmsd_A,wall_s\n")


def mace_calculator(model, head=None, dtype="float32", enable_cueq=False, device="cuda"):
    from mace.calculators import MACECalculator
    kw = dict(model_paths=str(model), device=device, default_dtype=dtype)
    if head is not None:
        kw["head"] = head
    if enable_cueq:
        kw["enable_cueq"] = True
    return MACECalculator(**kw)


def energy(calc, positions, species, cell):
    from ase import Atoms
    a = Atoms(symbols=list(species), positions=positions, cell=cell, pbc=True)
    a.calc = calc
    return float(a.get_potential_energy())


def start_positions(atoms, rattle, seed):
    """Input positions plus Gaussian noise of width rattle (A)."""
    rng = np.random.default_rng(4000 + seed)
    return atoms.get_positions() + rng.normal(0, rattle, (len(atoms), 3))


class Engine:

    def __init__(self, positions, species, cell, gr, hard_sphere, scattering_lengths,
                 window, cn=None, cn_cutoff=None, cn_weight=0.9, cn_sigma=0.05,
                 calc=None, seed=11):
        r_min, r_max, n_bins = window
        self.species = np.asarray(species)
        self.cell = np.asarray(cell, dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)
        self.n = len(positions)
        check_mic_safe(self.cell, r_max)
        self.pos = np.array(positions, dtype=float)
        if overlapping_atoms(self.pos, self.species, hard_sphere, self.cell, self.inv_cell):
            self.pos, n_it = repair_overlaps(self.pos, self.species, hard_sphere,
                                             self.cell, self.inv_cell)
            print(f"hard-sphere overlaps repaired in {n_it} passes")

        self.hist = Histograms(r_min, r_max, n_bins, self.pos, self.species,
                               self.cell, scattering_lengths)
        self.G_exp = np.interp(self.hist.r, *gr)
        self.hs = HardSphere(hard_sphere, self.species, self.cell, self.inv_cell)
        self.cn = None
        self.cn_weight = cn_weight
        if cn is not None:
            self.cn = Coordination(cn_cutoff, self.pos, self.species, self.cell,
                                   self.inv_cell, None if cn == "input" else cn, cn_sigma)
        self.calc = calc
        self.E = energy(calc, self.pos, self.species, self.cell) if calc else None

        self.x2_gr = float(np.sum((self.hist.G() - self.G_exp) ** 2))
        self.x2_cn = self.cn.chi2() if self.cn else 0.0
        self.x2 = self.x2_gr + cn_weight * self.x2_cn if self.cn else self.x2_gr
        self.generated = self.accepted = self.hs_rejected = 0
        self.rng = np.random.default_rng(seed)

    def G(self):
        return self.hist.r, self.hist.G()

    def step(self, d, sigma, kT):
        """One displacement move; kT=None means no energy term."""
        self.generated += 1
        i = self.rng.integers(0, self.n)
        old = self.pos[i].copy()
        dd = d[self.species[i]]
        new = wrap(old + self.rng.uniform(-dd, dd, 3), self.cell, self.inv_cell)
        if self.hs.violated(i, new, self.pos, self.species):
            self.hs_rejected += 1
            return
        dh = self.hist.delta(i, old, new, self.pos, self.species)
        x2_gr = float(np.sum((self.hist.G(dh) - self.G_exp) ** 2))
        if self.cn:
            dcn = self.cn.delta(i, old, new, self.pos, self.species)
            x2_cn = self.cn.chi2(dcn)
            x2 = x2_gr + self.cn_weight * x2_cn
        else:
            x2_cn, x2 = 0.0, x2_gr
        dx = (x2 - self.x2) / (2.0 * sigma * sigma)
        if kT is not None:
            trial = self.pos.copy()
            trial[i] = new
            e_new = energy(self.calc, trial, self.species, self.cell)
            log_alpha = -(dx + (e_new - self.E) / kT)
        else:
            log_alpha = -dx
        if log_alpha > 0.0 or self.rng.random() < np.exp(log_alpha):
            self.pos[i] = new
            self.hist.apply(dh)
            if self.cn:
                self.cn.apply(dcn)
            if kT is not None:
                self.E = e_new
            self.x2_gr, self.x2_cn, self.x2 = x2_gr, x2_cn, x2
            self.accepted += 1

    def fire(self, n_steps, fmax):
        """Partial FIRE relaxation under the potential, then rebuild the fit state."""
        from ase import Atoms
        from ase.optimize import FIRE
        a = Atoms(symbols=list(self.species), positions=self.pos.copy(),
                  cell=self.cell, pbc=True)
        a.calc = self.calc
        e_b = a.get_potential_energy()
        opt = FIRE(a, logfile=None)
        opt.run(fmax=fmax, steps=n_steps)
        e_a = a.get_potential_energy()
        self.pos = wrap(a.get_positions(), self.cell, self.inv_cell)
        self.hist.full(self.pos)
        self.x2_gr = float(np.sum((self.hist.G() - self.G_exp) ** 2))
        if self.cn:
            self.cn = Coordination(self.cn.cutoffs, self.pos, self.species, self.cell,
                                   self.inv_cell, self.cn.targets(), self.cn.sigma)
            self.x2_cn = self.cn.chi2()
            self.x2 = self.x2_gr + self.cn_weight * self.x2_cn
        else:
            self.x2 = self.x2_gr
        return e_b, e_a, opt.nsteps

    def run(self, n_steps, d, sigma, out, T=None, fire=None, step0=0,
            save_every=20_000, frame_every=0, conv=None):
        """Walk n_steps moves, writing trace.csv, traj.xyz and final_structure.xyz.

        T: inline temperature (K). fire: (M, N, fmax) for alternating HRMC.
        conv: (burn_in, window, tol, hits) arms the X2-plateau exit.
        step0 > 0 continues the files of a previous call.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if step0 == 0:
            (out / "trace.csv").write_text(TRACE_HEADER)
            (out / "traj.xyz").write_text("")
            if fire:
                (out / "fire_blocks.csv").write_text(FIRE_HEADER)
        kT = KB_EV * float(T) if T is not None else None
        if fire:
            M, N, fmax = fire
            next_fire = M
        series, hits, converged, n_fire = [], 0, False, 0
        g0, a0, h0 = self.generated, self.accepted, self.hs_rejected
        step = step0
        for k in range(1, int(n_steps) + 1):
            self.step(d, sigma, kT)
            step = step0 + k
            if fire and self.accepted >= next_fire:
                n_fire += 1
                next_fire += M
                x2_b, pos_b, t0 = self.x2, self.pos.copy(), time.time()
                e_b, e_a, nfs = self.fire(N, fmax)
                dm = (self.pos - pos_b) @ self.inv_cell
                dm -= np.round(dm)
                rmsd = float(np.sqrt(np.mean(np.sum((dm @ self.cell) ** 2, axis=1))))
                with open(out / "fire_blocks.csv", "a") as f:
                    f.write(f"{n_fire},{step},{x2_b:.8g},{self.x2:.8g},"
                            f"{e_b / self.n:.6f},{e_a / self.n:.6f},"
                            f"{nfs},{rmsd:.5f},{time.time() - t0:.1f}\n")
            if k % save_every == 0:
                e = f"{self.E / self.n:.6f}" if kT is not None else ""
                ng = max(self.generated - g0, 1)
                with open(out / "trace.csv", "a") as f:
                    f.write(f"{step},{self.x2:.8g},{e},"
                            f"{(self.accepted - a0) / ng:.4f},"
                            f"{(self.hs_rejected - h0) / ng:.4f},{n_fire}\n")
                g0, a0, h0 = self.generated, self.accepted, self.hs_rejected
                series.append(self.x2)
                if conv and k >= conv[0] and len(series) >= 2 * conv[1]:
                    burn, w, tol, need = conv
                    recent, prior = np.mean(series[-w:]), np.mean(series[-2 * w:-w])
                    hits = hits + 1 if abs(recent - prior) / max(abs(prior), 1e-12) < tol else 0
                    if hits >= need:
                        converged = True
                        break
            if frame_every and k % frame_every == 0:
                write_xyz(out / "traj.xyz", self.pos, self.species, self.cell, step, "a")
                print(f"step {step:,}  X2 {self.x2:.4f}  "
                      f"acc {self.accepted / self.generated:.3f}", flush=True)
        write_xyz(out / "final_structure.xyz", self.pos, self.species, self.cell, step)
        return dict(steps=step, converged=converged, X2=float(self.x2),
                    accepted=self.accepted, generated=self.generated,
                    hs_rejected=self.hs_rejected)
