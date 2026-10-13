"""Pure reverse Monte Carlo: G(r) fit, hard spheres, optional coordination constraint."""
from .engine import Engine, start_positions


def run(atoms, gr, hard_sphere, scattering_lengths, window, out,
        d=1.0, sigma=0.030, cn=None, cn_cutoff=None, cn_weight=0.9,
        n_steps=40_000_000, n_harvest=5_000_000, conv_tol=0.002,
        frame_every=400_000, harvest_frame_every=250_000, save_every=20_000,
        rattle=0.05, seed=11):
    """Fit until the X2 plateau test fires (or n_steps), then harvest n_harvest moves.

    cn: None (off), "input" (target = CN distribution of the input) or
    {(centre, neighbour): [(cn, fraction), ...]}; cn_cutoff: {(centre, neighbour): r}.
    The plateau test compares the means of the last two windows of 20 trace points
    and fires after 3 consecutive relative changes below conv_tol.
    """
    eng = Engine(start_positions(atoms, rattle, seed), atoms.get_chemical_symbols(),
                 atoms.get_cell(), gr, hard_sphere, scattering_lengths, window,
                 cn=cn, cn_cutoff=cn_cutoff, cn_weight=cn_weight, seed=seed)
    dd = {e: d for e in set(atoms.get_chemical_symbols())}
    fit = eng.run(n_steps, dd, sigma, out, save_every=save_every, frame_every=frame_every,
                  conv=(2 * 20 * save_every, 20, conv_tol, 3))
    if n_harvest:
        res = eng.run(n_harvest, dd, sigma, out, step0=fit["steps"],
                      save_every=save_every, frame_every=harvest_frame_every)
        res.update(fit_steps=fit["steps"], converged=fit["converged"])
        return res
    return fit
