"""Hybrid RMC with a MACE potential: inline (energy in the acceptance) or alternating
(pure-RMC moves interleaved with short FIRE relaxations)."""
from .engine import Engine, start_positions, mace_calculator, energy


def _engine(atoms, gr, hard_sphere, scattering_lengths, window, calc, rattle, seed,
            cn, cn_cutoff, cn_weight):
    return Engine(start_positions(atoms, rattle, seed), atoms.get_chemical_symbols(),
                  atoms.get_cell(), gr, hard_sphere, scattering_lengths, window,
                  cn=cn, cn_cutoff=cn_cutoff, cn_weight=cn_weight, calc=calc, seed=seed)


def inline(atoms, gr, hard_sphere, scattering_lengths, window, out, model,
           head=None, dtype="float32", enable_cueq=False, device="cuda",
           d=0.3, T=175.0, sigma=0.030, n_steps=1_200_000,
           frame_every=50_000, save_every=20_000, rattle=0.05, seed=11,
           cn=None, cn_cutoff=None, cn_weight=0.9):
    """Every move is accepted on exp(-dX2 / 2 sigma^2 - dE / kB T)."""
    calc = mace_calculator(model, head, dtype, enable_cueq, device)
    eng = _engine(atoms, gr, hard_sphere, scattering_lengths, window, calc, rattle, seed,
                  cn, cn_cutoff, cn_weight)
    res = eng.run(n_steps, {e: d for e in set(eng.species)}, sigma, out, T=T,
                  save_every=save_every, frame_every=frame_every)
    res["E_per_atom"] = eng.E / eng.n
    return res


def alternating(atoms, gr, hard_sphere, scattering_lengths, window, out, model,
                head=None, dtype="float32", enable_cueq=False, device="cuda",
                d=0.15, M=500, N=5, fmax=0.01, sigma=0.030, n_steps=20_000_000,
                frame_every=200_000, save_every=20_000, rattle=0.05, seed=11,
                cn=None, cn_cutoff=None, cn_weight=0.9):
    """Pure-RMC acceptance; after every M accepted moves, N FIRE steps (force cap fmax)."""
    calc = mace_calculator(model, head, dtype, enable_cueq, device)
    eng = _engine(atoms, gr, hard_sphere, scattering_lengths, window, None, rattle, seed,
                  cn, cn_cutoff, cn_weight)
    eng.calc = calc
    res = eng.run(n_steps, {e: d for e in set(eng.species)}, sigma, out,
                  fire=(M, N, fmax), save_every=save_every, frame_every=frame_every)
    res["E_per_atom"] = energy(calc, eng.pos, eng.species, eng.cell) / eng.n
    return res
