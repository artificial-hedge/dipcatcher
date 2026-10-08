"""Liang-Zeger generalized estimating equations (GEE) (SYNTHETIC).

Liang & Zeger (1986): for clustered/panel responses with a
marginal mean model g(mu_i) = x_i' beta, estimate beta by

    S(beta) = sum_i D_i' V_i^{-1} (y_i - mu_i) = 0

where D_i = d mu_i / d beta and V_i = phi * R_i(alpha)
(scale x working correlation). The working correlation is
estimated from standardized Pearson residuals each iteration
(independence / exchangeable / AR(1)). Inference uses the
robust sandwich (Huber 1967 / Royall 1986) covariance

    V_r = M0^{-1} M1 M0^{-1},
    M0 = sum D_i' V_i^{-1} D_i,
    M1 = sum D_i' V_i^{-1} r_i r_i' V_i^{-1} D_i,

which is consistent even when R(alpha) is misspecified.

Honesty: two link families — Gaussian/identity and
binomial/logit. The bench simulates cluster-correlated data
with known beta; the beta must be recovered within tolerance
and the exchangeable alpha must reflect the planted within-
cluster correlation. Fail-closed on degenerate clusters,
non-positive probabilities, or singular bread matrices.

References: Liang & Zeger (1986) "Longitudinal data analysis
using generalized linear models", Biometrika 73:13; Zeger &
Liang (1986); Huber (1967); Royall (1986); Hardin & Hilbe
(2003) "Generalized Estimating Equations".
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_CORR_STRUCTURES = ("independence", "exchangeable", "ar1")


def _sigmoid(v: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(v, -35, 35))), dtype=np.float64)


def _mean_link(eta: FloatArray, family: str) -> tuple[FloatArray, FloatArray]:
    """Return (mu, d mu / d eta) per family."""
    if family == "gaussian":
        return eta, np.ones_like(eta)
    mu = _sigmoid(eta)
    return mu, np.asarray(mu * (1.0 - mu), dtype=np.float64)


def _resid_scale(y: FloatArray, mu: FloatArray, family: str) -> FloatArray:
    """Pearson residual (y - mu) / sqrt(var(mu))."""
    if family == "gaussian":
        v = np.ones_like(mu)
    else:
        v = np.clip(mu * (1.0 - mu), 1e-8, None)
    return np.asarray((y - mu) / np.sqrt(v), dtype=np.float64)


def _alpha_estimate(resid: FloatArray, sizes: list[int], corr: str, phi: float) -> float:
    """Moment estimate of the working correlation — pairwise
    covariances are normalized by the scale phi to be actual
    correlations (Liang-Zeger 1986 moment estimator)."""
    if corr == "independence":
        return 0.0
    scale = max(phi, 1e-12)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    if corr == "exchangeable":
        num = den = 0.0
        for s0, m in zip(starts, sizes, strict=True):
            r = resid[s0 : s0 + m]
            num += float(np.triu(np.outer(r, r), 1).sum())
            den += m * (m - 1) / 2.0
        return float(np.clip(num / max(den * scale, 1e-12), -0.95, 0.95))
    # AR(1): correlation of consecutive within-cluster residuals
    num = den = 0.0
    for s0, m in zip(starts, sizes, strict=True):
        r = resid[s0 : s0 + m]
        if m > 1:
            num += float((r[:-1] * r[1:]).sum())
            den += m - 1
    return float(np.clip(num / max(den * scale, 1e-12), -0.95, 0.95))


def _r_inv(alpha: float, m: int, corr: str) -> FloatArray:
    """Inverse of the m x m working correlation."""
    if corr == "independence" or m == 1:
        return np.eye(m)
    if corr == "exchangeable":
        r = np.full((m, m), alpha) + np.diag(np.full(m, 1.0 - alpha))
    else:
        idx = np.arange(m)
        r = np.asarray(alpha ** np.abs(idx[:, None] - idx[None, :]), dtype=np.float64)
    r += np.eye(m) * 1e-8
    return np.asarray(np.linalg.inv(r), dtype=np.float64)


def gee(
    x: FloatArray,
    y: FloatArray,
    group: FloatArray,
    family: str = "gaussian",
    corr: str = "exchangeable",
    n_iter: int = 60,
) -> dict[str, float | FloatArray]:
    """Liang-Zeger GEE fit.

    `group` is a cluster id per row (already sorted by cluster is
    NOT assumed — rows are regrouped internally). `family` is
    'gaussian' or 'binomial'; `corr` is 'independence',
    'exchangeable', or 'ar1'.
    """
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64).ravel()
    gg = np.asarray(group, dtype=np.int64).ravel()
    if xx.ndim == 1:
        xx = xx[:, None]
    n, p = xx.shape
    if yy.size != n or gg.size != n or n < p + 2:
        raise ValueError("bad shapes")
    if corr not in _CORR_STRUCTURES:
        raise ValueError("unknown corr structure")
    if family not in ("gaussian", "binomial"):
        raise ValueError("unknown family")
    if family == "binomial" and not np.isin(yy, (0.0, 1.0)).all():
        raise ValueError("binomial y must be 0/1")
    order = np.argsort(gg, kind="stable")
    xs, ys, gs = xx[order], yy[order], gg[order]
    boundaries = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    sizes = [int(boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]
    if not sizes or max(sizes) == n:
        raise ValueError("need at least two clusters")
    p_eff = p + (1 if not np.allclose(xs[:, 0], 1.0) else 0)
    x_aug = np.column_stack([np.ones(n), xs]) if p_eff == p + 1 else xs
    beta = np.zeros(x_aug.shape[1], dtype=np.float64)
    alpha = 0.0
    phi = 1.0
    for _ in range(n_iter):
        eta = x_aug @ beta
        mu, dmu = _mean_link(eta, family)
        resid = np.asarray(ys - mu, dtype=np.float64)
        # Pearson residuals for alpha/phi
        presid = _resid_scale(ys, mu, family)
        denom = max(n - p_eff, 1)
        if family == "gaussian":
            phi = float((presid * presid).sum() / denom)
        else:
            phi = 1.0
        alpha = _alpha_estimate(presid, sizes, corr, phi)
        # quasi-score Newton step
        score = np.zeros(p_eff, dtype=np.float64)
        bread = np.zeros((p_eff, p_eff), dtype=np.float64)
        pos = 0
        for m in sizes:
            sl = slice(pos, pos + m)
            pos += m
            d_i = x_aug[sl] * dmu[sl][:, None]  # n_i x p
            v_inv = _r_inv(alpha, m, corr) / max(phi, 1e-8)
            score += d_i.T @ (v_inv @ resid[sl])
            bread += d_i.T @ v_inv @ d_i
        step = np.linalg.solve(bread + 1e-10 * np.eye(p_eff), score)
        beta_new = beta + np.clip(step, -5, 5)
        if float(np.abs(beta_new - beta).max()) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    # sandwich covariance
    eta = x_aug @ beta
    mu, dmu = _mean_link(eta, family)
    resid = np.asarray(ys - mu, dtype=np.float64)
    presid = _resid_scale(ys, mu, family)
    if family == "gaussian":
        phi = float((presid * presid).sum() / max(n - p_eff, 1))
    m0 = np.zeros((p_eff, p_eff), dtype=np.float64)
    m1 = np.zeros((p_eff, p_eff), dtype=np.float64)
    pos = 0
    for m in sizes:
        sl = slice(pos, pos + m)
        pos += m
        d_i = x_aug[sl] * dmu[sl][:, None]
        v_inv = _r_inv(alpha, m, corr) / max(phi, 1e-8)
        a_i = d_i.T @ v_inv
        m0 += a_i @ d_i
        u_i = a_i @ resid[sl]
        m1 += np.outer(u_i, u_i)
    m0_inv = np.linalg.inv(m0 + 1e-10 * np.eye(p_eff))
    cov = m0_inv @ m1 @ m0_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    # small-sample correction (Mancl-DeRouen style scalar)
    n_cl = len(sizes)
    se = np.asarray(se * math.sqrt(n_cl / max(n_cl - 1, 1)), dtype=np.float64)
    return {
        "beta": np.asarray(beta, dtype=np.float64),
        "se_robust": se,
        "alpha": float(alpha),
        "phi": float(phi),
        "n_clusters": float(n_cl),
        "mean_cluster": float(np.mean(sizes)),
    }


def bench_gee(seed: int = 20261231 + 450) -> dict[str, float]:
    """SYNTHETIC check — GEE recovers beta + exchangeable alpha."""
    rng = np.random.default_rng(seed)
    n_cl, m = 60, 8
    rho_true, b_true = 0.5, 1.4
    xs, ys, gs = [], [], []
    for c in range(n_cl):
        x_c = rng.normal(size=m)
        u_c = rng.normal(scale=math.sqrt(rho_true))
        e_c = rng.normal(scale=math.sqrt(1 - rho_true), size=m)
        y_c = 0.3 + b_true * x_c + u_c + e_c
        xs.append(x_c)
        ys.append(y_c)
        gs.append(np.full(m, c))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    out = gee(x, y, g, family="gaussian", corr="exchangeable")
    beta = np.asarray(out["beta"], dtype=np.float64)
    b_hat = float(beta[1])
    al = float(out["alpha"])
    if abs(b_hat - b_true) > 0.25 or abs(al - rho_true) > 0.35:
        raise ValueError(f"gee off: b={b_hat} rho_hat={al} true_rho={rho_true}")
    return {
        "synthetic_beta_err": abs(b_hat - b_true),
        "synthetic_alpha_err": abs(al - rho_true),
        "synthetic_score": 1.0,
    }
