"""Moving-horizon estimation vs EKF on a linear system with a sparse
outlier measurement noise process — MHE's windowed quadratic program
rejects the outlier through its L1-ish robust cost clamp.
"""

import numpy as np
from scipy.optimize import minimize


def _sim(seed: int, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 0.1)
        y[t] = x[t] + (rng.normal(0, 0.3) if t % 30 else rng.normal(0, 3.0))
    return x, y


def _mhe(y: np.ndarray, horizon: int = 8) -> np.ndarray:
    n = len(y)
    out = np.zeros(n)
    q_i, r_i = 1.0 / 0.01, 1.0 / 0.09
    for t in range(n):
        lo = max(0, t - horizon + 1)

        yw = y[lo : t + 1].copy()
        anchor = out[lo - 1] if lo > 0 else 0.0

        def cost(xs: np.ndarray, yw: np.ndarray = yw, anchor: float = anchor) -> float:
            proc = xs[1:] - 0.95 * xs[:-1]
            meas = np.clip(yw - xs, -0.9, 0.9)  # robust clamp
            prior = xs[0] - anchor
            return float(q_i * np.sum(proc**2) + r_i * np.sum(meas**2) + 2.0 * prior**2)

        res = minimize(cost, yw, method="L-BFGS-B", options={"maxiter": 60})
        out[t] = res.x[-1] if res.x is not None else y[t]
    return out


def _ekf(y: np.ndarray) -> np.ndarray:
    x, p = 0.0, 1.0
    out = np.zeros(len(y))
    for t in range(len(y)):
        x = 0.95 * x
        p = 0.95**2 * p + 0.01
        k = p / (p + 0.09)
        x += k * (y[t] - x)
        p *= 1 - k
        out[t] = x
    return out


def bench_mhe(seed: int = 5705) -> dict[str, float]:
    x, y = _sim(seed)
    m = _mhe(y)
    e = _ekf(y)
    m_rmse = float(np.sqrt(np.mean((m - x) ** 2)))
    e_rmse = float(np.sqrt(np.mean((e - x) ** 2)))
    return {
        "synthetic_mhe_rmse": m_rmse,
        "synthetic_mhe_ekf_rmse": e_rmse,
        "synthetic_mhe_gain": float(m_rmse < e_rmse),
    }
