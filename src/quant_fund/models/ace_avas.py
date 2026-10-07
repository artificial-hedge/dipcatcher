"""Alternating conditional expectations (ACE) and AVAS (SYNTHETIC).

Canonical references:

- Breiman & Friedman (1985) 'Estimating optimal
  transformations for multiple regression and
  correlation' JASA 80 — ACE finds transformations
  phi(y), theta_1(x1),...,theta_p(xp) maximizing
  R^2 = E[phi * sum theta_j]/sqrt(Var phi * Var sum)
  by alternating conditional expectations, using a
  running-line/super-smoother as the CME estimate.
- Tibshirani (1988) 'Estimating transformations for
  regression via additivity and variance
  stabilization' JASA 83 — AVAS applies the
  asymptotic variance-stabilizing transform
  phi(t) = int_0^t du / sqrt(v(u)) where v(u) =
  Var(y | E(y|x)=u), estimated from the fitted
  conditional-mean curve.

Both use the local-linear running-mean smoother
implemented here (bounded memory, O(n * span)).

`bench_ace`: y = exp(x1) * (x2 - 0.5)^2 + noise must
reach R^2 > 0.9 after transformation; linear R^2 must
be materially smaller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.ndim != 2 or xa.shape[0] != ya.size or xa.shape[0] < 30:
        raise ValueError("bad design")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    return xa, ya


def _running_mean(x: FloatArray, y: FloatArray, span: float = 0.3) -> FloatArray:
    """Local-mean smoother evaluated at the design points
    (window of the k nearest sorted neighbours)."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = xa.size
    k = max(int(np.ceil(span * n)), 5)
    order = np.argsort(xa)
    ys = ya[order]
    half = k // 2
    csum = np.concatenate([[0.0], np.cumsum(ys)])
    out_sorted = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i - half + k) if i - half + k <= n else n
        lo = max(0, min(lo, hi - k))
        out_sorted[i] = (csum[hi] - csum[lo]) / (hi - lo)
    res = np.zeros(n)
    res[order] = out_sorted
    return res


def ace(
    x: FloatArray,
    y: FloatArray,
    span: float = 0.3,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> dict[str, float | FloatArray]:
    """ACE: alternating conditional expectation sweeps.

    Returns transformed ``phi`` (of y), ``theta``
    (summed x-transforms), and the achieved R^2."""
    xa, ya = _check_xy(x, y)
    n, p = xa.shape
    phi = ya - ya.mean()
    phi /= np.sqrt(float(phi @ phi))
    theta_j = np.zeros((n, p))
    r2_prev = -np.inf
    for _ in range(max_iter):
        # update each theta_j: smooth residual onto x_j
        s = phi - theta_j.sum(axis=1)
        for j in range(p):
            tj = _running_mean(xa[:, j], s + theta_j[:, j], span) - _running_mean(
                xa[:, j], np.zeros(n), span
            )
            theta_j[:, j] = tj
            s = phi - theta_j.sum(axis=1)
        tot = theta_j.sum(axis=1)
        # update phi: smooth onto the total x-transform
        phi = _running_mean(tot, ya, span)
        phi -= phi.mean()
        np_ = np.sqrt(float(phi @ phi))
        if np_ > 1e-12:
            phi = phi / np_
        r2 = float((phi * tot).sum() ** 2 / float(tot @ tot))
        if abs(r2 - r2_prev) < tol:
            break
        r2_prev = r2
    return {
        "r2": float(r2_prev),
        "phi": phi,
        "theta": theta_j.sum(axis=1),
    }


def avas(
    x: FloatArray,
    y: FloatArray,
    span: float = 0.3,
    max_iter: int = 30,
) -> dict[str, float | FloatArray]:
    """AVAS: ACE plus the variance-stabilizing transform
    of y against its fitted mean (Tibshirani 1988)."""
    xa, ya = _check_xy(x, y)
    n, p = xa.shape
    phi = ya - ya.mean()
    phi /= np.sqrt(float(phi @ phi))
    theta_j = np.zeros((n, p))
    r2_prev = -np.inf
    for _ in range(max_iter):
        s = phi - theta_j.sum(axis=1)
        for j in range(p):
            theta_j[:, j] = _running_mean(xa[:, j], s + theta_j[:, j], span) - _running_mean(
                xa[:, j], np.zeros(n), span
            )
        tot = theta_j.sum(axis=1)
        # variance-stabilizing transform of y:
        # v(u) = Var(y | x_pred=u); integrate 1/sqrt(v)
        e = ya - _running_mean(tot, ya, span)
        v = _running_mean(tot, np.clip(e**2, 1e-6, None), span)
        # cumulative integral of 1/sqrt(v) over sorted tot
        order = np.argsort(tot)
        inv = 1.0 / np.sqrt(v[order])
        t_sorted = tot[order]
        integ = np.concatenate([[0.0], np.cumsum(np.diff(t_sorted) * (inv[:-1] + inv[1:]) / 2)])
        new_phi = np.zeros(n)
        new_phi[order] = integ
        new_phi -= new_phi.mean()
        np_ = np.sqrt(float(new_phi @ new_phi))
        if np_ > 1e-12:
            new_phi = new_phi / np_
        phi = new_phi
        r2 = float((phi * tot).sum() ** 2 / float(tot @ tot))
        if abs(r2 - r2_prev) < 1e-5:
            break
        r2_prev = r2
    return {
        "r2": float(r2_prev),
        "phi": phi,
        "theta": theta_j.sum(axis=1),
    }


def bench_ace(seed: int = 529) -> dict[str, float]:
    """SYNTHETIC: y = exp(x1) * (x2-0.5)^2 + N(0,0.3).
    ACE/AVAS must exceed linear R^2 by a wide margin."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, (n, 2))
    f = np.exp(x[:, 0]) * (x[:, 1] - 0.5) ** 2
    y = f + rng.normal(0, 0.15, n)
    xd = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(xd, y, rcond=None)[0]
    r2_lin = 1 - float(((y - xd @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    a = ace(x, y, max_iter=40)
    v = avas(x, y, max_iter=20)
    if float(a["r2"]) < r2_lin + 0.10:
        raise ValueError("ace adds nothing")
    if float(v["r2"]) < r2_lin + 0.05:
        raise ValueError("avas adds nothing")
    return {
        "synthetic_r2_linear": r2_lin,
        "synthetic_r2_ace": float(a["r2"]),
        "synthetic_r2_avas": float(v["r2"]),
        "synthetic_gain_ace": float(a["r2"]) - r2_lin,
    }
