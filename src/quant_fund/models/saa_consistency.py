"""Sample-average approximation consistency for a newsvendor.

True problem: max_x E[min(D,x)] - c x with D ~ Gamma. SAA solves on N
samples; the optimality gap is estimated with M independent replicates
(lower bound via SAA objectives, upper bound via out-of-sample eval of
the SAA solution). Bench reports gap shrinkage N=50 -> N=400.
"""

import numpy as np


def _saa_obj(x: float, d: np.ndarray, c: float) -> float:
    return float(np.mean(np.minimum(d, x)) - c * x)


def _solve_saa(d: np.ndarray, c: float, grid: np.ndarray) -> tuple[float, float]:
    vals = np.array([_saa_obj(x, d, c) for x in grid])
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])


def bench_saa_consistency(seed: int = 5505) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    c = 0.6
    grid = np.linspace(0.0, 12.0, 301)
    # true optimum: E[min(D,x)]' = P(D>x) = c -> x* = quantile
    d_big = rng.gamma(3.0, 2.0, 200000)
    xs_star = float(np.quantile(d_big, 1.0 - c))
    truth = _saa_obj(xs_star, d_big, c)
    out: dict[str, float] = {"synthetic_saa_true_obj": truth, "synthetic_saa_x_star": xs_star}
    for n in (50, 400):
        gaps = []
        for _ in range(40):
            d = rng.gamma(3.0, 2.0, n)
            x_hat, saa_val = _solve_saa(d, c, grid)
            # out-of-sample eval of x_hat
            oos = _saa_obj(x_hat, rng.gamma(3.0, 2.0, 20000), c)
            gaps.append(truth - oos)
        out[f"synthetic_saa_gap_{n}"] = float(np.mean(gaps))
    return out
