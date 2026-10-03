"""Forward-filter backward-simulation particle smoother on a stochastic
volatility toy: x_t = a x_{t-1} + s v, y = exp(x/2) e. Bootstrap filter
with ancestor tracking, then backward-simulated smoothing paths; bench
compares smoother vs filter RMSE against truth.
"""

import numpy as np


def _sim(seed: int, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + 0.3 * rng.normal()
        y[t] = np.exp(x[t] / 2) * rng.normal()
    return x, y


def _ffbs(x_t: np.ndarray, y: np.ndarray, np_: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    parts = rng.normal(0, 1.0, np_)
    anc = np.zeros((n, np_), dtype=int)
    store = np.zeros((n, np_))
    store[0] = parts
    for t in range(1, n):
        prop = 0.95 * parts + 0.3 * rng.normal(0, 1, np_)
        lw = -0.5 * (y[t] ** 2) * np.exp(-prop) - prop / 2
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        idx = rng.choice(np_, np_, p=w)
        anc[t] = idx
        parts = prop[idx]
        store[t] = parts
    # backward simulation: trace one path back through ancestors
    b = np.zeros(n)
    j = rng.integers(np_)
    for t in range(n - 1, -1, -1):
        b[t] = store[t, j]
        j = anc[t, j]
    filt = store.mean(1)
    # multi-path smoother estimate: average several backward paths
    smooth = np.zeros(n)
    for _ in range(20):
        j = rng.integers(np_)
        for t in range(n - 1, -1, -1):
            smooth[t] += store[t, j]
            j = anc[t, j]
    smooth /= 20
    return filt, smooth


def bench_particle_smoother(seed: int = 5711) -> dict[str, float]:
    x, y = _sim(seed)
    filt, smooth = _ffbs(x, y, 400, seed)
    f_rmse = float(np.sqrt(np.mean((filt - x) ** 2)))
    s_rmse = float(np.sqrt(np.mean((smooth - x) ** 2)))
    return {
        "synthetic_ps_filt_rmse": f_rmse,
        "synthetic_ps_smooth_rmse": s_rmse,
        "synthetic_ps_gain": float(s_rmse < f_rmse),
    }
