"""Linear predictive coding: autocorrelation + Levinson-Durbin + residual.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 904


def autocorr(x: np.ndarray, p: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    r = np.array([np.dot(x[: x.size - k], x[k:]) / x.size for k in range(p + 1)])
    return np.asarray(r, dtype=np.float64)


def levinson(r: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """Levinson-Durbin recursion; returns (a, residual_power)."""
    r = np.asarray(r, dtype=np.float64)
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = float(r[0])
    for i in range(1, p + 1):
        lam = -sum(a[j] * r[i - j] for j in range(1, i)) - r[i]
        k = lam / max(e, 1e-12)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e = float(max((1.0 - k * k) * e, 1e-15))
    return np.asarray(a, dtype=np.float64), e


def lpc_residual(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    p = a.size - 1
    r = x.copy()
    for i in range(p, x.size):
        r[i] = x[i] + float(np.dot(a[1:], x[i - p : i][::-1]))
    return np.asarray(r, dtype=np.float64)


def synthesize(exc: np.ndarray, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    p = a.size - 1
    y = np.zeros(exc.size)
    for i in range(exc.size):
        acc = 0.0
        for j in range(1, min(i, p) + 1):
            acc += a[j] * y[i - j]
        y[i] = exc[i] - acc
    return np.asarray(y, dtype=np.float64)


def bench_lpc_analysis(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # AR(4) process
    a_true = np.array([1.0, -0.9, 0.5, -0.25, 0.1])
    e = rng.normal(0, 1.0, 8000)
    x = synthesize(e, a_true)
    r = autocorr(x, 4)
    a_hat, _res = levinson(r, 4)
    score += 1.0 if np.max(np.abs(a_hat - a_true)) < 0.15 else 0.0
    # residual ≈ white: variance ~ excitation variance
    res = lpc_residual(x, a_hat)
    ratio = float(np.var(res[200:]) / np.var(e[200:]))
    score += 1.0 if 0.9 < ratio < 1.3 else 0.0
    # synthesis roundtrip exact-ish
    xr = synthesize(res, a_hat)
    score += 1.0 if np.max(np.abs(xr[50:] - x[50:])) < 1e-6 else 0.0
    # reflection coeffs stable (|k|<1 → stable filter): error decays each order
    r2 = autocorr(x, 12)
    errs = []
    for p in (1, 4, 12):
        _, ep = levinson(r2, p)
        errs.append(ep)
    score += 1.0 if errs[0] > errs[1] > errs[2] > 0 else 0.0
    return {"synthetic_lpc_analysis": score / 4.0}
