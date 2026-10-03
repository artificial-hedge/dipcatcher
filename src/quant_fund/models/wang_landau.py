"""Wang-Landau density-of-states estimation for the 1-D Ising chain.

Random spin flips on a 16-spin chain; log g(E) updated by ln f until
flatness (H within 20% of mean), then f -> sqrt(f). Recovers g(E)
ratios vs exact enumeration over 2^16 states.
"""

import itertools

import numpy as np


def _energy(s: np.ndarray) -> float:
    return float(-np.sum(s * np.roll(s, 1)))


def _exact_dos(n: int) -> dict[int, float]:
    dos: dict[int, float] = {}
    for bits in itertools.product([-1, 1], repeat=n):
        e = _energy(np.array(bits))
        dos[int(e)] = dos.get(int(e), 0.0) + 1.0
    return dos


def bench_wang_landau(seed: int = 5603) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 16
    exact = _exact_dos(n)
    energies = np.array(sorted(exact))
    e_idx = {e: i for i, e in enumerate(energies)}
    log_g = np.zeros(len(energies))
    hist = np.zeros(len(energies))
    s = rng.choice([-1, 1], size=n)
    ln_f = 1.0
    iters = 0
    while ln_f > 1e-4 and iters < 400000:
        i = rng.integers(n)
        e_old = _energy(s)
        s[i] *= -1
        e_new = _energy(s)
        diff = log_g[e_idx[int(e_old)]] - log_g[e_idx[int(e_new)]]
        if diff >= 0 or rng.random() < np.exp(diff):
            pass  # accept
        else:
            s[i] *= -1
            e_new = e_old
        log_g[e_idx[int(e_new)]] += ln_f
        hist[e_idx[int(e_new)]] += 1
        iters += 1
        if iters % 2000 == 0 and hist.min() > 0.8 * hist.mean():
            ln_f /= 2.0
            hist[:] = 0
    est = np.exp(log_g - log_g.max())
    tru = np.array([exact[e] for e in energies], dtype=float)
    tru /= tru.max()
    corr = float(np.corrcoef(np.log(np.maximum(est, 1e-300)), np.log(tru))[0, 1])
    return {
        "synthetic_wl_levels": float(len(energies)),
        "synthetic_wl_logcorr": corr,
        "synthetic_wl_iters": float(iters),
    }
