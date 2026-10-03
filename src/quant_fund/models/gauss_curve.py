"""Gaussian curvature via shape operator (wave 287).

K = det(II)/det(I) at sampled points of a sphere of radius r equals
1/r^2; torus K varies sign as predicted by the formula
K = cos(u) / (r (R + r cos u)).
"""

import numpy as np

_SEED = 20261231 + 807


def _num2(f, uv: tuple[float, float], h: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = uv
    duu = (f(u + h, v) - 2 * f(u, v) + f(u - h, v)) / h**2
    dvv = (f(u, v + h) - 2 * f(u, v) + f(u, v - h)) / h**2
    duv = (f(u + h, v + h) - f(u + h, v - h) - f(u - h, v + h) + f(u - h, v - h)) / (4 * h**2)
    return duu, duv, dvv


def _k_num(f, uv: tuple[float, float]) -> float:
    u, v = uv
    h = 1e-5
    du = (f(u + h, v) - f(u - h, v)) / (2 * h)
    dv = (f(u, v + h) - f(u, v - h)) / (2 * h)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n)
    duu, duv, dvv = _num2(f, uv)
    L, M, N = duu @ n, duv @ n, dvv @ n
    E, F, G = du @ du, du @ dv, dv @ dv
    return float((L * N - M**2) / (E * G - F**2))


def _sphere(u: float, v: float, r: float = 2.5) -> np.ndarray:
    return r * np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)])


def _torus(u: float, v: float) -> np.ndarray:
    R, r = 3.0, 1.0
    return np.array(
        [(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)]
    )


def bench_gauss_curve(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(6):
        u, v = rng.uniform(0.4, 2.4, 2)
        errs.append(abs(_k_num(_sphere, (u, v)) - 1 / 6.25))
        want = np.cos(u) / (1.0 * (3.0 + np.cos(u)))
        errs.append(abs(_k_num(_torus, (u, v)) - want))
    return {"synthetic_gauss_k": float(max(errs) < 1e-2)}
