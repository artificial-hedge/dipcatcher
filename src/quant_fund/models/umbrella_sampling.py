"""Umbrella sampling across a free-energy barrier (SYNTHETIC).

Windows with harmonic biases (k/2)(x-c_i)^2 on a double-well target;
biased MC per window, then naive unbiased reweighting per window to
recover the barrier height vs the analytic value.
"""

import numpy as np


def _v(x: float) -> float:
    return float(0.25 * x**4 - 2.0 * x**2 + 1.0)  # barrier at 0, wells +-2


def _run(center: float, k: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = center
    out = np.zeros(n)
    beta = 1.0
    for t in range(n):
        prop = x + rng.normal(0, 0.5)
        de = _v(prop) - _v(x) + 0.5 * k * ((prop - center) ** 2 - (x - center) ** 2)
        if rng.random() < np.exp(-beta * de):
            x = prop
        out[t] = x
    return out


def bench_umbrella_sampling(seed: int = 5605) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    centers = np.linspace(-2.4, 2.4, 7)
    k = 8.0
    xs = [_run(c, k, 4000, rng) for c in centers]
    # unbiased estimate of P(x in barrier region |x|<1) via reweighting
    num = 0.0
    den = 0.0
    for c, x in zip(centers, xs, strict=True):
        w = np.exp(0.5 * k * (x - c) ** 2)  # unbias
        num += float(np.sum(w * (np.abs(x) < 0.4)))
        den += float(np.sum(w))
    p_bar = num / den
    # truth by quadrature
    grid = np.linspace(-4, 4, 20001)
    z = np.exp(-np.array([_v(g) for g in grid]))
    truth = float(
        np.trapezoid(z[np.abs(grid) < 0.4], grid[np.abs(grid) < 0.4]) / np.trapezoid(z, grid)
    )
    # naive single-run estimate starting in one well (gets stuck)
    naive = float(np.mean(np.abs(_run(-2.0, 0.0, 20000, rng)) < 0.4))
    return {
        "synthetic_us_p_barrier": p_bar,
        "synthetic_us_truth": truth,
        "synthetic_us_naive": naive,
        "synthetic_us_err": abs(p_bar - truth),
        "synthetic_us_naive_err": abs(naive - truth),
    }
