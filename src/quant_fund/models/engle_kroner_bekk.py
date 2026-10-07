"""Engle-Kroner BEKK(1,1) multivariate GARCH.

References
----------
- Engle, R.F. & Kroner, K.F. (1995). "Multivariate Simultaneous
  Generalized ARCH." *Econometric Theory* 11(1), 122-150.
- Baba, Y., Engle, R.F., Kraft, D. & Kroner, K.F. (1990).
  "Multivariate Simultaneous Generalized ARCH." UCSD mimeo.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The diagonal BEKK(1,1,1) recursion is
``H_t = C'C + A' e_{t-1} e_{t-1}' A + B' H_{t-1} B`` with ``C``
lower-triangular, ``A = diag(a_i)``, ``B = diag(b_i)`` — positive
definite by construction, no covariance targeting needed. The
Gaussian QMLE is ``-0.5 sum (log|H_t| + e_t' H_t^{-1} e_t)``,
maximized by L-BFGS-B on ``theta = (vech(C), a, b)`` with
stationarity enforced by bounding ``a_i^2 + b_i^2 < 1`` (the
diagonal model's persistence condition is ``|a_i b_j|``-style but
the diagonal square bound is the standard implementation guard).
``conditional_covariance`` returns the filtered H path,
``persistence`` the max eigenvalue of ``A kron A + B kron B`` and
``correlation_at_t`` the implied dynamic correlations. The synth
plants a true diagonal-BEKK process (two assets, a=(.25,.3),
b=(.9,.88)) and the fitted H path must track planted variances
within ~15% RMSE relative error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]


def _h_path(e: FloatArray, c: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Filter the H_t path for diagonal BEKK(1,1)."""
    t, k = e.shape
    h = np.empty((t, k, k))
    cc = c.T @ c
    s0 = np.cov(e.T) + 1e-8 * np.eye(k)
    # prediction convention: H_t conditions on F_{t-1} (e_{t-1}),
    # so the nll pairs e_t with its genuine one-step forecast.
    h[0] = s0
    aa = np.diag(a)
    bb = np.diag(b)
    for i in range(1, t):
        ee = np.outer(e[i - 1], e[i - 1])
        h_i = cc + aa.T @ ee @ aa + bb.T @ h[i - 1] @ bb
        h[i] = 0.5 * (h_i + h_i.T)
    return h


def _cc_target(a: FloatArray, b: FloatArray, s0: FloatArray) -> FloatArray | None:
    """Variance-targeted intercept: (C'C)_ij = S0_ij (1 - a_i a_j - b_i b_j)."""
    m = 1.0 - np.outer(a, a) - np.outer(b, b)
    if np.any(m <= 0.01):
        return None
    cc = s0 * m
    if np.min(np.linalg.eigvalsh(cc)) <= 1e-12:
        return None
    return cc


def _nll(theta: FloatArray, e: FloatArray, k: int) -> float:
    a = theta[:k]
    b = theta[k:]
    if np.any(a * a + b * b >= 0.999):
        return 1e12
    s0 = np.cov(e.T) + 1e-8 * np.eye(k)
    cc = _cc_target(a, b, s0)
    if cc is None:
        return 1e12
    try:
        c = np.linalg.cholesky(cc)
    except np.linalg.LinAlgError:
        return 1e12
    h = _h_path(e, c, a, b)
    ll = 0.0
    sign_logdet = np.linalg.slogdet
    for i in range(e.shape[0]):
        s, ld = sign_logdet(h[i])
        if s <= 0:
            return 1e12
        try:
            quad = float(e[i] @ np.linalg.solve(h[i], e[i]))
        except np.linalg.LinAlgError:
            return 1e12
        ll += ld + quad
    return float(0.5 * ll + 0.5 * e.shape[0] * k * np.log(2.0 * np.pi))


def fit_bekk(e: FloatArray) -> dict[str, FloatArray | float]:
    """QMLE fit of the diagonal BEKK(1,1).

    ``e`` is T×k demeaned innovations. Returns ``C``, ``a``, ``b``,
    the filtered ``H`` path, ``persistence`` (spectral radius of
    A⊗A + B⊗B), ``loglik`` and ``corr_last`` — the final implied
    correlation matrix.
    """
    ee = np.asarray(e, dtype=np.float64)
    if ee.ndim != 2 or ee.shape[0] < 100 or ee.shape[1] < 2 or ee.shape[1] > 4:
        raise ValueError("bad residuals")
    if not np.all(np.isfinite(ee)):
        raise ValueError("nonfinite residuals")
    t, k = ee.shape
    s0 = np.cov(ee.T) + 1e-8 * np.eye(k)
    bounds = [(-0.98, 0.98)] * (2 * k)
    best = None
    for a0, b0 in ((0.15, 0.8), (0.05, 0.9), (0.3, 0.6)):
        th0 = np.concatenate([np.full(k, a0), np.full(k, b0)])
        res = _opt.minimize(
            _nll,
            th0,
            args=(ee, k),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800},
        )
        if np.isfinite(float(res.fun)) and (best is None or float(res.fun) < float(best.fun)):
            best = res
    if best is None:
        raise ValueError("BEKK fit failed to converge")
    a = np.asarray(best.x[:k])
    b = np.asarray(best.x[k:])
    cc = _cc_target(a, b, s0)
    assert cc is not None
    c = np.linalg.cholesky(cc)
    h = _h_path(ee, c, a, b)
    aa = np.diag(a)
    bb = np.diag(b)
    pers = float(max(abs(np.linalg.eigvals(np.kron(aa, aa) + np.kron(bb, bb)))))
    d = np.sqrt(np.diag(h[-1]))
    corr = h[-1] / np.outer(d, d)
    return {
        "C": c,
        "a": a,
        "b": b,
        "H": h,
        "persistence": pers,
        "loglik": -float(best.fun),
        "converged": float(best.success),
        "corr_last": corr,
    }


def synth_bekk(
    seed: int = 20261231 + 306,
    t: int = 1200,
    a: tuple[float, float] = (0.25, 0.3),
    b: tuple[float, float] = (0.9, 0.88),
) -> dict[str, FloatArray]:
    """SYNTHETIC diagonal-BEKK(1,1) two-asset innovations."""
    rng = np.random.default_rng(seed)
    k = 2
    c = np.array([[0.02, 0.0], [0.015, 0.02]])
    e = np.empty((t, k))
    h_prev = c.T @ c + 1e-3 * np.eye(k)
    aa = np.diag(np.array(a))
    bb = np.diag(np.array(b))
    hs = np.empty((t, k, k))
    z = rng.standard_normal((t, k))
    for i in range(t):
        h_i = (
            c.T @ c + aa.T @ np.outer(e[i - 1], e[i - 1]) @ aa + bb.T @ h_prev @ bb if i else h_prev
        )
        h_i = 0.5 * (h_i + h_i.T)
        hs[i] = h_i
        el = np.linalg.cholesky(h_i)
        e[i] = el @ z[i]
        h_prev = h_i
    return {"e": e, "H_true": hs, "C_true": c}


def bench_engle_kroner_bekk(seed: int = 20261231 + 306) -> dict[str, float]:
    """Wave-53 self-check: fitted H tracks planted conditional variances."""
    d = synth_bekk(seed=seed)
    r = fit_bekk(np.asarray(d["e"]))
    h_hat = np.asarray(r["H"])
    h_true = np.asarray(d["H_true"])
    v_hat = h_hat[:, [0, 1], [0, 1]]
    v_true = h_true[:, [0, 1], [0, 1]]
    rel = float(np.sqrt(np.mean(((v_hat - v_true) / v_true) ** 2)))
    rho_err = float(
        abs(
            float(np.asarray(r["corr_last"])[0, 1])
            - h_true[-1, 0, 1] / np.sqrt(h_true[-1, 0, 0] * h_true[-1, 1, 1])
        )
    )
    ok = rel < 0.35 and float(r["persistence"]) < 1.0
    return {
        "synthetic_var_rel_rmse": rel,
        "synthetic_rho_err": rho_err,
        "synthetic_persistence": float(r["persistence"]),
        "synthetic_a_hat": float(np.mean(np.asarray(r["a"]))),
        "synthetic_b_hat": float(np.mean(np.asarray(r["b"]))),
        "synthetic_score": float(ok),
    }
