"""Eigenvalue optimization: min_x lambda_max(A0 + sum_i x_i A_i) (SYNTHETIC).

Subgradient method on the convex spectral function: step direction
uses the top-eigenvector diagonals g_i = v' A_i v. Validated on a small
5x5 fixture against a brute-force grid search over x.
"""

import numpy as np

_RNG = np.random.default_rng(42)
_A0 = _RNG.normal(size=(5, 5))
_A0 = _A0 + _A0.T
_AS = []
for _ in range(4):
    m = _RNG.normal(size=(5, 5))
    _AS.append(m + m.T)


def _lam_max(x: np.ndarray) -> tuple[float, np.ndarray]:
    m = _A0 + sum(xi * a for xi, a in zip(x, _AS, strict=True))
    w, v = np.linalg.eigh(m)
    i = int(np.argmax(w))
    return float(w[i]), v[:, i]


def bench_eigenvalue_opt(seed: int = 5303) -> dict[str, float]:
    x = np.zeros(4)
    lb, ub = -3.0, 3.0
    best_x = x.copy()
    best = np.inf
    for it in range(400):
        f, v = _lam_max(x)
        g = np.array([v @ a @ v for a in _AS])
        step = 0.5 / np.sqrt(it + 1)
        x = np.clip(x - step * g, lb, ub)
        if f < best:
            best, best_x = f, x.copy()
    # brute grid (coarse) as reference
    grid = np.linspace(lb, ub, 9)
    brute = np.inf
    from itertools import product as _p

    for xs in _p(*[grid] * 4):
        f, _ = _lam_max(np.array(xs))
        brute = min(brute, f)
    return {
        "synthetic_eo_lam": float(best),
        "synthetic_eo_grid_lb": float(brute),
        "synthetic_eo_margin": float(brute - best),
        "synthetic_eo_x_norm": float(np.linalg.norm(best_x)),
    }
