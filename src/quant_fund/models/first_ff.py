"""First fundamental form of parameterized surfaces (wave 287).

E,F,G from partial derivatives of sphere and torus parameterizations
match the closed-form metric coefficients.
"""

import numpy as np

_SEED = 20261231 + 806


def _num_diff(f, uv: tuple[float, float], h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    u, v = uv
    du = (f(u + h, v) - f(u - h, v)) / (2 * h)
    dv = (f(u, v + h) - f(u, v - h)) / (2 * h)
    return du, dv


def _sphere(u: float, v: float, r: float = 2.0) -> np.ndarray:
    return r * np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)])


def _torus(u: float, v: float) -> np.ndarray:
    R, r = 3.0, 1.0
    return np.array(
        [(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)]
    )


def bench_first_ff(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = 0.0
    for _ in range(6):
        u, v = rng.uniform(0.4, 2.4, 2)
        du, dv = _num_diff(_sphere, (u, v))
        e, f, g = du @ du, du @ dv, dv @ dv
        errs += abs(e - 4.0) + abs(f) + abs(g - 4.0 * np.sin(u) ** 2)
        du, dv = _num_diff(_torus, (u, v))
        e, f, g = du @ du, du @ dv, dv @ dv
        errs += abs(e - 1.0) + abs(f) + abs(g - (3 + np.cos(u)) ** 2)
    return {"synthetic_first_ff": float(errs / 12 < 1e-3)}
