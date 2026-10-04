"""Gibbs-method initial orbit determination from three position vectors."""

from __future__ import annotations

import numpy as np


def gibbs(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float) -> np.ndarray:
    """Return the velocity at the middle observation (Gibbs method)."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r3 = np.asarray(r3, float)
    r1m, r2m, r3m = (float(np.linalg.norm(r)) for r in (r1, r2, r3))
    c12 = np.cross(r1, r2)
    c23 = np.cross(r2, r3)
    c31 = np.cross(r3, r1)
    N = r1m * c23 + r2m * c31 + r3m * c12
    D = c12 + c23 + c31
    S = r1 * (r2m - r3m) + r2 * (r3m - r1m) + r3 * (r1m - r2m)
    B = np.cross(D, r2)
    L = np.sqrt(mu / (np.linalg.norm(N) * np.linalg.norm(D)))
    return np.asarray(L * (B / r2m + S))


def bench_gauss_iod(seed: int = 20261231 + 859) -> dict[str, float]:
    """Recover known velocities at three true anomalies of a known orbit."""
    from quant_fund.models.orbital_elements import elements_to_state, state_to_elements

    rng = np.random.default_rng(seed)
    mu = 398600.4418
    checks = 0.0
    total = 0
    for _ in range(20):
        a = float(rng.uniform(8000.0, 20000.0))
        e = float(rng.uniform(0.05, 0.5))
        inc = float(rng.uniform(0.05, 2.0))
        nu1 = float(rng.uniform(0.0, 2.0 * np.pi))
        d1, d2 = (float(rng.uniform(0.2, 0.8)) for _ in range(2))
        nus = [nu1, nu1 + d1, nu1 + d1 + d2]
        rs, vs = [], []
        for nu in nus:
            r, v = elements_to_state(a, e, inc, 0.7, 0.3, nu, mu)
            rs.append(r)
            vs.append(v)
        v2_hat = gibbs(rs[0], rs[1], rs[2], mu)
        total += 2
        checks += float(np.linalg.norm(v2_hat - vs[1]) / np.linalg.norm(vs[1]) < 1e-6)
        # recovered elements match truth
        a2, e2, i2, _, _, _ = state_to_elements(rs[1], v2_hat, mu)
        checks += float(abs(a2 - a) / a < 1e-6 and abs(e2 - e) < 1e-6 and abs(i2 - inc) < 1e-6)
    return {"synthetic_gauss_iod": checks / total}
