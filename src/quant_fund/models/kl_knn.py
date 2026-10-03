"""kNN KL-divergence estimator (wave 285).

Perez-Cruz / Wang style estimator D(P||Q) ~ (d/n) sum log(nu_i/rho_i) +
log(m/(n-1)) with nu,rho kNN distances under each sample. Validated on
shifted Gaussians vs the analytic closed form.
"""

import numpy as np

_SEED = 20261231 + 796


def kl_nn(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    n, m = len(x), len(y)
    tot = 0.0
    for i in range(n):
        dx = np.sort(np.abs(np.delete(x, i) - x[i]))[k - 1]
        dy = np.sort(np.abs(y - x[i]))[k - 1]
        tot += np.log(dy / dx)
    return float(np.log(m / (n - 1)) + tot / n)


def bench_kl_knn(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # N(0,1) vs N(1,1): KL = 0.5
    x = rng.randn(3000)
    y = rng.randn(3000) + 1.0
    est = kl_nn(x, y)
    return {"synthetic_kl_nn": float(abs(est - 0.5) < 0.15)}
