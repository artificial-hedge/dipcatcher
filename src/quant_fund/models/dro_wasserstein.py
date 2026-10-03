"""Wasserstein distributionally-robust newsvendor.

DRO: min_x sup_{P: W(P,P_hat)<=rho} E_P[loss(x,D)] over the empirical
sample distribution; solved by enumerating the worst-case over small
support perturbations of the SAA samples. Bench compares the robust
solution's out-of-sample loss vs the plain SAA solution under a
shifted demand distribution.
"""

import numpy as np


def _loss(x: float, d: np.ndarray) -> float:
    over = np.maximum(x - d, 0) * 1.0 + np.maximum(d - x, 0) * 2.0
    return float(over.mean())


def _saa(d: np.ndarray, grid: np.ndarray) -> float:
    return float(grid[int(np.argmin([_loss(x, d) for x in grid]))])


def _dro(d: np.ndarray, grid: np.ndarray, rho: float) -> float:
    best = np.inf
    best_x = 0.0
    for x in grid:
        # adversarial shift of each sample by up to rho on average
        worst = 0.0
        for sgn in (-1.0, 1.0):
            worst = max(worst, _loss(x, d + sgn * rho))
        if worst < best:
            best, best_x = worst, x
    return best_x


def bench_dro_wasserstein(seed: int = 5509) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d_tr = rng.gamma(3.0, 2.0, 200)
    d_te = rng.gamma(3.0, 2.4, 50000)  # shifted test distribution
    grid = np.linspace(0.0, 16.0, 321)
    x_saa = _saa(d_tr, grid)
    x_dro = _dro(d_tr, grid, 0.8)
    return {
        "synthetic_dro_x_saa": x_saa,
        "synthetic_dro_x_dro": x_dro,
        "synthetic_dro_loss_saa": _loss(x_saa, d_te),
        "synthetic_dro_loss_dro": _loss(x_dro, d_te),
        "synthetic_dro_gain": _loss(x_saa, d_te) - _loss(x_dro, d_te),
    }
