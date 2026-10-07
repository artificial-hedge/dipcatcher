"""Method of lines — central-difference Laplacian + RK4 time
integration of the heat equation; classical baseline for the canon.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, eval_error, grid, u0


def bench_moc_lines(seed: int = 2525, nx: int = 101, nt: int = 2000) -> dict[str, float]:
    x = grid(nx)
    dx = x[1] - x[0]
    u = u0(x)
    dt = T_ / nt

    def rhs(v: np.ndarray) -> np.ndarray:
        dv = np.zeros_like(v)
        dv[1:-1] = 0.5 * SIG**2 * (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        return dv

    for _ in range(nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u[0] = u[-1] = 0.0

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        return np.asarray(np.interp(xq, x, u))

    return {"synthetic_moc_rel_l2": eval_error(pred), "synthetic_torch_available": 0.0}
