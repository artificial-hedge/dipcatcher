"""Cobweb model: supply lag → price oscillation (stable iff slope ratio < 1)."""

import numpy as np

_SEED = 20261231 + 757


def cobweb_run(d_slope: float, s_slope: float, steps: int, p0: float) -> np.ndarray:
    """p_{t+1} = -(s_slope/d_slope) p_t (naive expectations)."""
    p = np.zeros(steps)
    p[0] = p0
    for t in range(1, steps):
        p[t] = -(s_slope / d_slope) * p[t - 1]
    return p


def bench_cobweb_model(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    stable = cobweb_run(1.0, 0.5, 60, 1.0)  # |ratio|<1 → converge
    diverge = cobweb_run(1.0, 2.0, 60, 1.0)  # |ratio|>1 → explode
    ok = float(abs(stable[-1]) < 0.01 and abs(diverge[-1]) > 1e10)
    return {"synthetic_cobweb_stability": ok}
