"""Synthetic fixtures shared by the information-geometry canon (SYNTHETIC).

Univariate-Gaussian model space (Fisher-Rao), an ill-conditioned Gaussian
fit (natural gradient), a simplex-constrained least squares (mirror
descent), a Poisson count matrix (Bregman NMF), multinomial endpoints
(alpha-geodesic), and an Ornstein-Uhlenbeck Fokker-Planck target (JKO).
"""

import numpy as np


def fr_gauss_pair() -> tuple[tuple[float, float], tuple[float, float]]:
    return (0.0, 1.0), (1.5, 0.7)


def gauss_fit(seed: int, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(2.0, 0.35, n)


def simplex_ls(seed: int, d: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) * rng.lognormal(0, 1.5, (d, d))
    x_star = rng.dirichlet(np.ones(d) * 0.4)
    b = A @ x_star + rng.normal(0.0, 0.01, d)
    return A, b


def poisson_nmf(seed: int, m: int = 30, n: int = 20, r: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.gamma(1.0, 1.0, (m, r))
    h = rng.gamma(1.0, 1.0, (r, n))
    return rng.poisson(w @ h).astype(np.float64) + 1e-6


def multinomial_pair() -> tuple[np.ndarray, np.ndarray]:
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.05, 0.15, 0.3, 0.5])
    return p, q


def ou_stationary_sigma() -> float:
    return 1.0
