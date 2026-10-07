"""Mirror descent (exponentiated gradient) on simplex-constrained LS (SYNTHETIC).

min_x ||Ax - b||^2 s.t. x in the probability simplex. Exponentiated-gradient
updates x <- softmax(log x - eta grad) vs projected-gradient with simplex
projection; bench compares final residual and iteration count.
"""

import numpy as np

from quant_fund.models._ig_synth import simplex_ls


def _proj_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.asarray(np.maximum(v - theta, 0.0), dtype=np.float64)


def _solve(
    A: np.ndarray, b: np.ndarray, mirror: bool, lr: float, iters: int = 6000
) -> tuple[np.ndarray, float]:
    d = A.shape[1]
    x = np.full(d, 1.0 / d)
    for it in range(iters):
        g = A.T @ (A @ x - b)
        if mirror:
            x = x * np.exp(-lr * g)
            x /= x.sum()
        else:
            x = _proj_simplex(x - lr * g)
        if np.linalg.norm(g) < 1e-7:
            return x, float(it)
    return x, float(iters)


def bench_mirror_descent(seed: int = 4305) -> dict[str, float]:
    A, b = simplex_ls(seed)
    x_m, it_m = _solve(A, b, mirror=True, lr=0.01)
    x_p, it_p = _solve(A, b, mirror=False, lr=0.01)
    r_m = float(np.linalg.norm(A @ x_m - b))
    r_p = float(np.linalg.norm(A @ x_p - b))
    return {
        "synthetic_md_resid": r_m,
        "synthetic_pgd_resid": r_p,
        "synthetic_md_resid_gain": r_p - r_m,
        "synthetic_md_simplex_ok": float(abs(x_m.sum() - 1.0)),
        "synthetic_md_min": float(x_m.min()),
    }
