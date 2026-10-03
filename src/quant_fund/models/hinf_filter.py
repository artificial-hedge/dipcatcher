"""H-infinity filter vs Kalman on a model-misspecified tracking problem.

True process has heavier drift noise than the filter assumes; the
H∞ filter bounds the worst-case gain from disturbances to error via a
gamma parameter and a modified Riccati. Bench: worst-case error ratio
under adversarial noise vs Kalman.
"""

import numpy as np


def _sim(seed: int, n: int = 400, adversarial: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    a, q_true, r_true = 0.98, 0.2, 0.5
    for t in range(1, n):
        if adversarial and 150 < t < 250:
            w = 2.0 * np.sqrt(q_true) * np.sign(np.sin(0.4 * t))
        else:
            w = rng.normal(0, np.sqrt(q_true))
        x[t] = a * x[t - 1] + w
        y[t] = x[t] + rng.normal(0, np.sqrt(r_true))
    return x, y


def _kalman(y: np.ndarray) -> np.ndarray:
    a, q, r = 0.98, 0.05, 0.5  # misspecified q (too small)
    x, p = 0.0, 1.0
    out = np.zeros(len(y))
    for t in range(len(y)):
        x = a * x
        p = a * a * p + q
        k = p / (p + r)
        x = x + k * (y[t] - x)
        p = (1 - k) * p
        out[t] = x
    return out


def _hinf(y: np.ndarray, gamma: float = 0.35) -> np.ndarray:
    a, q, r = 0.98, 0.05, 0.5
    x, p = 0.0, 1.0
    out = np.zeros(len(y))
    for t in range(len(y)):
        x = a * x
        # H∞ Riccati: P <- a² (P^{-1} - gamma^{-2} I + H'R^{-1}H)^{-1} + Q
        pi = 1.0 / p - 1.0 / gamma**2 + 1.0 / r
        pi = max(pi, 1e-6)
        p = a * a / pi + q
        k = p / (p + r)
        x = x + k * (y[t] - x)
        p = (1 - k) * p
        out[t] = x
    return out


def bench_hinf_filter(seed: int = 5701) -> dict[str, float]:
    x, y = _sim(seed, adversarial=True)
    k_est = _kalman(y)
    h_est = _hinf(y)
    seg = slice(150, 250)
    k_err = float(np.max(np.abs(k_est[seg] - x[seg])))
    h_err = float(np.max(np.abs(h_est[seg] - x[seg])))
    k_rmse = float(np.sqrt(np.mean((k_est - x) ** 2)))
    h_rmse = float(np.sqrt(np.mean((h_est - x) ** 2)))
    return {
        "synthetic_hinf_worst": h_err,
        "synthetic_hinf_k_worst": k_err,
        "synthetic_hinf_rmse": h_rmse,
        "synthetic_hinf_k_rmse": k_rmse,
        "synthetic_hinf_gain": float(h_err < k_err),
    }
