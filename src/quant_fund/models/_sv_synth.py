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
    if n < 1 or d < 1:
        raise ValueError(f"need n,d >= 1, got {n},{d}")
    if beta is not None and np.asarray(beta).shape != (d,):
        raise ValueError(f"beta must be (d={d},), got {np.asarray(beta).shape}")
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
    if not (len(t) == len(risk_pred) == len(e)) or len(t) < 1:
        raise ValueError(
            f"t/risk_pred/e must be non-empty equal lengths, got {len(t)},{len(risk_pred)},{len(e)}"
        )
    if not np.isfinite(t).all() or not np.isfinite(risk_pred).all():
        raise ValueError("non-finite t or risk_pred")
    conc = 0.0
    pairs = 0.0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if e[i] == 1 and t[i] < t[j]:
                pairs += 1
                if risk_pred[i] > risk_pred[j]:
                    conc += 1
                elif risk_pred[i] < risk_pred[j]:
                    pass
                else:
                    conc += 0.5
    if pairs == 0:
        raise ValueError("no comparable uncensored pairs — C-index undefined")
    return conc / pairs


def cox_ph(X: FloatArray, t: FloatArray, e: NDArray[np.int64], iters: int = 300) -> FloatArray:
    """Cox PH partial likelihood gradient ascent → coefficient vector."""
    if iters < 1:
        raise ValueError(f"need iters>=1, got {iters}")
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"X must be a non-empty (n,d) array, got {X.shape}")
    if not (len(t) == len(e) == X.shape[0]):
        raise ValueError("t/e must match X rows")
    if not set(np.unique(e)) <= {0, 1}:
        raise ValueError("event indicator must be binary {0,1}")
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
