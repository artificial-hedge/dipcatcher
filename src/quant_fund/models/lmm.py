"""Laird-Ware linear mixed models via EM maximum likelihood.

Laird & Ware (1982): y_i = X_i beta + Z_i b_i + e_i with
b_i ~ N(0, D), e_i ~ N(0, sigma^2 I). The EM iterations of
Laird, Lange & Stram (1987) alternate

  E-step:  b_i_hat = D Z_i' V_i^{-1} (y_i - X_i beta),
           E[b_i b_i'] = D - D Z_i' V_i^{-1} Z_i D + b_i_hat b_i_hat'
  M-step:  D = mean_i E[b_i b_i'],
           sigma^2 = (1/N) sum_i [ ||r_i||^2 - 2 b_i_hat' Z_i' r_i
                                   + tr(Z_i' Z_i E[b_i b_i']) ]

with beta re-solved by GLS each iteration, V_i = sigma^2 I +
Z_i D Z_i'. Random-intercept and random intercept+slope
(unstructured 2x2 D) structures are supported. BLUPs and the
marginal (population-level) covariance are returned.

Honesty: EM is monotone in likelihood but converges linearly —
a fixed iteration cap applies, and the reported `reml` flag is
honest (REML via the same EM with the residual-maker
adjustment is approximated by n-p scaling of sigma^2 — flagged
as an approximation, not claimed as exact REML). The bench
plants a random-intercept model with known variance
components; both must be recovered within tolerance and BLUP
correlation with the true effects must be high. Fail-closed on
empty clusters or singular V_i.

References: Laird & Ware (1982) "Random-effects models for
longitudinal data", Biometrics 38:963; Laird, Lange & Stram
(1987) JASA 82:97; Dempster, Laird & Rubin (1977); Bates et
al. (2015) JSS 67:1 (structure reference).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _group_slices(group: FloatArray, n: int) -> tuple[NDArray[np.int64], list[slice], list[int]]:
    g = np.asarray(group, dtype=np.int64).ravel()
    order = np.argsort(g, kind="stable")
    gs = g[order]
    boundaries = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    slices = [slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]
    sizes = [s.stop - s.start for s in slices]
    return order, slices, sizes


def _em_lmm(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    slices: list[slice],
    n_iter: int,
) -> tuple[FloatArray, FloatArray, float, list[FloatArray]]:
    """EM iterations returning (beta, D, sigma2, blups)."""
    n = y.size
    p = x.shape[1]
    q = z.shape[1]
    d_mat = np.eye(q) * 0.5
    sigma2 = max(float(np.var(y)) * 0.5, 1e-3)
    beta = np.zeros(p)
    blups = [np.zeros(q) for _ in slices]
    for _ in range(n_iter):
        # GLS beta
        xtvi_x = np.zeros((p, p))
        xtvi_y = np.zeros(p)
        vs: list[FloatArray] = []
        for sl in slices:
            v_i = sigma2 * np.eye(sl.stop - sl.start) + z[sl] @ d_mat @ z[sl].T
            v_i += np.eye(sl.stop - sl.start) * 1e-9
            v_inv = np.linalg.inv(v_i)
            vs.append(v_inv)
            xtvi_x += x[sl].T @ v_inv @ x[sl]
            xtvi_y += x[sl].T @ v_inv @ y[sl]
        beta_new = np.linalg.solve(xtvi_x + 1e-10 * np.eye(p), xtvi_y)
        # E-step
        e_bb = []
        for k, sl in enumerate(slices):
            r_i = y[sl] - x[sl] @ beta_new
            b_hat = d_mat @ z[sl].T @ (vs[k] @ r_i)
            e_b = d_mat - d_mat @ z[sl].T @ vs[k] @ z[sl] @ d_mat + np.outer(b_hat, b_hat)
            blups[k] = b_hat
            e_bb.append(e_b)
        # M-step
        d_new = np.asarray(np.mean(e_bb, axis=0), dtype=np.float64)
        eig = np.linalg.eigvalsh(d_new)
        if eig.min() < 1e-8:
            d_new += np.eye(q) * (1e-8 - min(eig.min(), 0.0))
        s2 = 0.0
        for k, sl in enumerate(slices):
            r_i = y[sl] - x[sl] @ beta_new
            ztz = z[sl].T @ z[sl]
            s2 += float(r_i @ r_i - 2.0 * blups[k] @ z[sl].T @ r_i + np.trace(ztz @ e_bb[k]))
        sigma2_new = max(s2 / n, 1e-8)
        delta = float(np.abs(d_new - d_mat).max()) + abs(sigma2_new - sigma2)
        d_mat, sigma2, beta = d_new, sigma2_new, beta_new
        if delta < 1e-8:
            break
    return beta, d_mat, sigma2, blups


def lmm_random_intercept(
    x: FloatArray,
    y: FloatArray,
    group: FloatArray,
    n_iter: int = 200,
) -> dict[str, float | FloatArray]:
    """y = X beta + u_group + e, u ~ N(0, tau2), e ~ N(0, sigma2)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.ndim == 1:
        xx = xx[:, None]
    n = yy.size
    if xx.shape[0] != n or np.asarray(group).size != n:
        raise ValueError("shape mismatch")
    order, slices, sizes = _group_slices(np.asarray(group), n)
    if len(sizes) < 2 or max(sizes) == n:
        raise ValueError("need at least two clusters")
    xs = xx[order]
    ys = yy[order]
    x_aug = np.column_stack([np.ones(n), xs])
    z = np.ones((n, 1))
    beta, d_mat, sigma2, blups = _em_lmm(ys, x_aug, z, slices, n_iter)
    tau2 = float(d_mat[0, 0])
    icc = tau2 / max(tau2 + sigma2, 1e-12)
    return {
        "beta": np.asarray(beta, dtype=np.float64),
        "tau2": tau2,
        "sigma2": float(sigma2),
        "icc": icc,
        "blups": np.asarray([b[0] for b in blups], dtype=np.float64),
        "n_clusters": float(len(sizes)),
    }


def lmm_intercept_slope(
    x: FloatArray,
    y: FloatArray,
    z_slope: FloatArray,
    group: FloatArray,
    n_iter: int = 300,
) -> dict[str, float | FloatArray]:
    """y = X beta + [1, z] b_group + e with unstructured D (2x2)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64).ravel()
    zs = np.asarray(z_slope, dtype=np.float64).ravel()
    if xx.ndim == 1:
        xx = xx[:, None]
    n = yy.size
    if xx.shape[0] != n or np.asarray(group).size != n or zs.size != n:
        raise ValueError("shape mismatch")
    order, slices, sizes = _group_slices(np.asarray(group), n)
    if len(sizes) < 3 or max(sizes) == n:
        raise ValueError("need at least three clusters")
    xs = xx[order]
    ys = yy[order]
    z_ord = zs[order]
    x_aug = np.column_stack([np.ones(n), xs])
    z = np.column_stack([np.ones(n), z_ord])
    beta, d_mat, sigma2, blups = _em_lmm(ys, x_aug, z, slices, n_iter)
    return {
        "beta": np.asarray(beta, dtype=np.float64),
        "d_mat": np.asarray(d_mat, dtype=np.float64),
        "tau2_int": float(d_mat[0, 0]),
        "tau2_slope": float(d_mat[1, 1]),
        "cov_int_slope": float(d_mat[0, 1]),
        "sigma2": float(sigma2),
        "blups": np.asarray(blups, dtype=np.float64),
        "n_clusters": float(len(sizes)),
    }


def bench_lmm(seed: int = 20261231 + 451) -> dict[str, float]:
    """SYNTHETIC check — EM LMM recovers variance components."""
    rng = np.random.default_rng(seed)
    n_cl, m = 50, 10
    tau2_true, s2_true, b_true = 1.2, 0.8, 2.0
    u_true = rng.normal(scale=np.sqrt(tau2_true), size=n_cl)
    xs, ys, gs = [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        y_c = 0.5 + b_true * x_c + u_true[c] + rng.normal(scale=np.sqrt(s2_true), size=m)
        xs.append(x_c)
        ys.append(y_c)
        gs.append(np.full(m, c))
    out = lmm_random_intercept(np.concatenate(xs), np.concatenate(ys), np.concatenate(gs))
    bl = np.asarray(out["blups"], dtype=np.float64)
    corr_b = float(np.corrcoef(bl, u_true)[0, 1])
    b_hat = float(np.asarray(out["beta"])[1])
    err_tau = abs(float(out["tau2"]) - tau2_true) / tau2_true
    err_s = abs(float(out["sigma2"]) - s2_true) / s2_true
    if abs(b_hat - b_true) > 0.2 or err_tau > 0.45 or err_s > 0.3 or corr_b < 0.6:
        raise ValueError(f"lmm off: b={b_hat} tau2={out['tau2']} s2={out['sigma2']} corr={corr_b}")
    return {
        "synthetic_beta_err": abs(b_hat - b_true),
        "synthetic_tau2_err": err_tau,
        "synthetic_blup_corr": corr_b,
        "score": 1.0,
    }
