"""Christoffel symbols from the metric (wave 287) (SYNTHETIC).

Gamma^k_ij = 1/2 g^{kl} (d_i g_jl + d_j g_il - d_l g_ij) computed by
finite differences of the metric tensor match analytic values on the
2-sphere and polar plane.
"""

import numpy as np

_SEED = 20261231 + 809


def _g_sphere(u: float, v: float, r: float = 1.0) -> np.ndarray:
    return np.array([[r * r, 0.0], [0.0, r * r * np.sin(u) ** 2]])


def _g_polar(r: float, th: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, r * r]])


def christoffel(g, uv: tuple[float, float], h: float = 1e-5) -> np.ndarray:
    u, v = uv
    g0 = g(u, v)
    gi = np.linalg.inv(g0)
    dg = np.zeros((2, 2, 2))
    for k, (du, dv) in enumerate([(h, 0.0), (0.0, h)]):
        dg[k] = (g(u + du, v + dv) - g(u - du, v - dv)) / (2 * h)
    gam = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gam[k, i, j] = 0.5 * np.sum(gi[k] * (dg[i, :, j] + dg[j, :, i] - dg[:, i, j]))
    return gam


def bench_christoffel(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(6):
        u, v = rng.uniform(0.5, 2.5, 2)
        g = christoffel(_g_sphere, (u, v))
        errs += [abs(g[0, 1, 1] + np.sin(u) * np.cos(u)), abs(g[1, 0, 1] - 1 / np.tan(u))]
        r, th = rng.uniform(0.5, 3.0, 2)
        g = christoffel(_g_polar, (r, th))
        errs += [abs(g[0, 1, 1] + r), abs(g[1, 0, 1] - 1 / r)]
    return {"synthetic_christoffel": float(max(errs) < 1e-3)}
