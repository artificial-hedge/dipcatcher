"""WHAM free-energy reconstruction from umbrella windows (1-D textbook).

Binned counts h_i(bin) per biased simulation i; iterate
    p(bin) = sum_i h_i(bin) / sum_i n_i exp(f_i - u_i(bin))
    f_i    = -log sum_bin p(bin) exp(-u_i(bin))
then F(bin) = -log p(bin). Bench compares the barrier free energy at
x=0 against the quadrature answer.
"""

import numpy as np


def _v(x: np.ndarray) -> np.ndarray:
    return 0.25 * x**4 - 2.0 * x**2 + 1.0


def _run(center: float, k: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = center
    out = np.zeros(n)
    for t in range(n):
        prop = x + rng.normal(0, 0.4)
        de = (
            _v(np.array(prop))
            - _v(np.array(x))
            + 0.5 * k * ((prop - center) ** 2 - (x - center) ** 2)
        )
        if rng.random() < np.exp(-float(de)):
            x = prop
        out[t] = x
    return out


def bench_wham(seed: int = 5609) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    centers = np.linspace(-2.4, 2.4, 9)
    k = 5.0
    n = 5000
    grid = np.linspace(-3.4, 3.4, 137)
    samples = [_run(c, k, n, rng) for c in centers]
    h = np.stack(
        [
            np.histogram(
                s,
                bins=np.append(grid - (grid[1] - grid[0]) / 2, grid[-1] + (grid[1] - grid[0]) / 2),
            )[0]
            for s in samples
        ],
        0,
    ).astype(float)  # nw x nb
    u = 0.5 * k * (grid[None, :] - centers[:, None]) ** 2  # nw x nb
    f = np.zeros(len(centers))
    p = np.ones(len(grid))
    for _ in range(2000):
        denom = (n * np.exp(f[:, None] - u)).sum(0)
        p = h.sum(0) / np.maximum(denom, 1e-300)
        f_new = -np.log(np.maximum((p[None, :] * np.exp(-u)).sum(1), 1e-300))
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < 1e-10:
            f = f_new
            break
        f = f_new
    fe = -np.log(np.maximum(p, 1e-300))
    fe -= fe[np.argmin(np.abs(grid - 2.0))]
    bar_hat = float(fe[np.argmin(np.abs(grid))])
    z = np.exp(-_v(grid))
    p_true = z / np.trapezoid(z, grid)
    fe_true = -np.log(p_true)
    fe_true -= fe_true[np.argmin(np.abs(grid - 2.0))]
    truth = float(fe_true[np.argmin(np.abs(grid))])
    return {
        "synthetic_wham_barrier": bar_hat,
        "synthetic_wham_truth": truth,
        "synthetic_wham_err": abs(bar_hat - truth),
    }
