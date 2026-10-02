"""Shared SYNTHETIC fixture for wave-179 survival-DL canon:
Weibull event times with covariate-dependent scale + independent
censoring; concordance index (C-index) vs exponential Cox PH.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sv_data(
    seed: int,
    n: int = 400,
    d: int = 6,
    beta: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], FloatArray]:
    """Returns (X, time_observed, event_indicator, beta)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if beta is None:
        beta = rng.standard_normal(d)
        beta /= np.linalg.norm(beta)
    risk = X @ beta  # higher → shorter survival
    scale = np.exp(-risk)
    shape = 1.5
    T_true = scale * (-np.log(rng.random(n))) ** (1 / shape)
    C = np.exp(rng.standard_normal(n) * 0.5 + 1.0)
    t = np.minimum(T_true, C)
    e = (T_true <= C).astype(np.int64)
    return X, t, e, np.asarray(beta)


def cindex(t: FloatArray, risk_pred: FloatArray, e: NDArray[np.int64]) -> float:
    """Harrell C-index on uncensored pairs."""
    conc = disc = 0.0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if e[i] == 1 and t[i] < t[j]:
                if risk_pred[i] > risk_pred[j]:
                    conc += 1
                elif risk_pred[i] < risk_pred[j]:
                    disc += 1
                else:
                    conc += 0.5
    return conc / max(conc + disc, 1)


def cox_ph(X: FloatArray, t: FloatArray, e: NDArray[np.int64], iters: int = 300) -> FloatArray:
    """Cox PH partial likelihood gradient ascent → coefficient vector."""
    d = X.shape[1]
    b = np.zeros(d)
    order = np.argsort(t)
    for _ in range(iters):
        xb = X @ b
        exb = np.exp(xb)
        grad = np.zeros(d)
        for i in order:
            if e[i] == 0:
                continue
            risk_set = t >= t[i]
            wsum = exb[risk_set].sum()
            grad += X[i] - (X[risk_set] * exb[risk_set, None]).sum(0) / max(wsum, 1e-9)
        b += 0.01 * grad
    return b
