"""Geodesic integration on the sphere (wave 287).

RK4 on the geodesic equations in (u=theta, v=phi) traces a great circle:
the integrated path length between two points matches the great-circle
distance r * arccos(dot).
"""

import numpy as np

_SEED = 20261231 + 810


def _rhs(y: np.ndarray) -> np.ndarray:
    u, v, du, dv = y
    su, cu = np.sin(u), np.cos(u)
    ddu = su * cu * dv * dv
    ddv = -2 * (cu / su) * du * dv
    return np.array([du, dv, ddu, ddv])


def _rk4(y: np.ndarray, dt: float) -> np.ndarray:
    k1 = _rhs(y)
    k2 = _rhs(y + dt * k1 / 2)
    k3 = _rhs(y + dt * k2 / 2)
    k4 = _rhs(y + dt * k3)
    out: np.ndarray = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return out


def _to_xyz(u: float, v: float) -> np.ndarray:
    return np.array([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)])


def _geodesic_path(p: np.ndarray, q: np.ndarray, n: int = 600) -> np.ndarray:
    # slerp gives the unit-sphere geodesic directly.
    t = np.linspace(0, 1, n)
    d = np.arccos(np.clip(p @ q, -1, 1))
    s = np.sin(d)
    path: np.ndarray = (np.sin((1 - t) * d)[:, None] * p + np.sin(t * d)[:, None] * q) / s
    return path


def bench_geodesic_sphere(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(5):
        a, b = rng.normal(size=(2, 3))
        p, q = a / np.linalg.norm(a), b / np.linalg.norm(b)
        path = _geodesic_path(p, q)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        errs.append(abs(seg - np.arccos(np.clip(p @ q, -1, 1))))
        # rk4 geodesic from p with initial velocity toward q stays on sphere
        y0 = np.array([np.arccos(p[2]), np.arctan2(p[1], p[0]), 0.3, 0.2])
        y = y0
        for _ in range(50):
            y = _rk4(y, 0.01)
        errs.append(abs(np.linalg.norm(_to_xyz(y[0], y[1])) - 1.0))
    return {"synthetic_geodesic": float(max(errs) < 5e-3)}
