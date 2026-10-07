"""Ornstein–Uhlenbeck bridge: conditional path pinned at both endpoints (SYNTHETIC)
(mean-reverting interpolation). Endpoint adherence + intermediate
variance vs unconditional OU paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ou_bridge(x0: float, xT: float, k: float, sig: float, T: float, steps: int, rng) -> FloatArray:
    """Exact OU bridge via conditioned-Gaussian construction."""
    ts = np.linspace(0, T, steps + 1)
    # unconditional OU covariance: cov(x_s, x_t) = sig²/(2k)(e^{-k|t-s|} - e^{-k(t+s)})
    t1 = ts[:, None]
    t2 = ts[None, :]
    C = sig**2 / (2 * k) * (np.exp(-k * np.abs(t1 - t2)) - np.exp(-k * (t1 + t2)))
    m = x0 * np.exp(-k * ts)
    # condition on endpoint x_T: adjust mean/cov
    idx = np.arange(1, steps)
    c_tt = C[-1, -1]
    c_iT = C[idx, -1]
    c_ii = C[np.ix_(idx, idx)]
    m_cond = m[idx] + c_iT / c_tt * (xT - m[-1])
    K = c_ii - np.outer(c_iT, c_iT) / c_tt
    K += np.eye(len(K)) * 1e-10
    samp = rng.multivariate_normal(m_cond, K)
    path = np.concatenate([[x0], samp, [xT]])
    return np.asarray(path)


def bench_ou_bridge(seed: int = 2939, trials: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    end_err, mid_vars = [], []
    x0, xT, k, sig, T = 1.0, -0.5, 1.5, 0.5, 1.0
    for _ in range(trials):
        p = _ou_bridge(x0, xT, k, sig, T, 40, rng)
        end_err.append(abs(p[-1] - xT))
        mid_vars.append(p[20])
    # bridge variance at midpoint should be < unconditional variance
    sig_u = sig**2 / (2 * k) * (1 - np.exp(-k * 1.0))
    var_mid = float(np.var(mid_vars))
    return {
        "synthetic_oub_endpoint_err": float(np.mean(end_err)),
        "synthetic_oub_mid_var": var_mid,
        "synthetic_oub_uncond_var": float(sig_u),
        "synthetic_torch_available": 0.0,
    }
