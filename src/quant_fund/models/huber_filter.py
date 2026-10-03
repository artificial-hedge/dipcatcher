"""Huber-robust Kalman filter: clipped innovation update.

Standard KF recursion but the innovation is Huberized at delta before
the gain is applied — bounded influence vs Gaussian-tailed outliers.
"""

import numpy as np


def _sim(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + rng.normal(0, 0.15)
        y[t] = x[t] + rng.standard_t(3.0) * 0.5  # heavy-tailed obs noise
    return x, y


def _kf(y: np.ndarray, huber: float | None) -> np.ndarray:
    x, p = 0.0, 1.0
    out = np.zeros(len(y))
    for t in range(len(y)):
        x = 0.97 * x
        p = 0.97**2 * p + 0.0225
        k = p / (p + 0.25)
        inn = y[t] - x
        if huber is not None:
            inn = np.clip(inn, -huber, huber)
        x += k * inn
        p *= 1 - k
        out[t] = x
    return out


def bench_huber_filter(seed: int = 5709) -> dict[str, float]:
    x, y = _sim(seed)
    g = _kf(y, None)
    h = _kf(y, 0.8)
    g_rmse = float(np.sqrt(np.mean((g - x) ** 2)))
    h_rmse = float(np.sqrt(np.mean((h - x) ** 2)))
    g_max = float(np.max(np.abs(g - x)))
    h_max = float(np.max(np.abs(h - x)))
    return {
        "synthetic_hf_rmse": h_rmse,
        "synthetic_hf_kf_rmse": g_rmse,
        "synthetic_hf_maxerr": h_max,
        "synthetic_hf_kf_maxerr": g_max,
        "synthetic_hf_gain": float(h_rmse < g_rmse),
    }
