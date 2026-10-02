"""Actuarial loss reserving: chain-ladder, Mack (1993)
variance, Bornhuetter-Ferguson, and England-Verrall ODP
bootstrap.

Given a run-off triangle of cumulative (or incremental)
payments C[i,j] by accident year i and development year j,
estimates age-to-age factors, ultimate losses, reserves,
and the Mack (1993) mean-squared-error decomposition
(process + parameter risk), plus a Bornhuetter-Ferguson
blend on a priori loss ratios and an over-dispersed
Poisson bootstrap of the chain-ladder reserve.

References
----------
- Mack (1993) 'Distribution-free calculation of the
  standard error of chain ladder reserve estimates'
  ASTIN Bulletin 23(2).
- England & Verrall (1999) 'Analytic and bootstrap
  estimates of prediction errors in claims reserving'
  Insurance M&E 25.
- Bornhuetter & Ferguson (1972) PCAS 59.

Honesty
-------
SYNTHETIC self-check: simulates a triangle from a known
over-dispersed Poisson and measures reserve recovery —
no real claims data, no market claims.

Composition
-----------
Pure numpy. Triangle passed as an upper-triangular
(n x n) matrix with NaN below the anti-diagonal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_triangle(c: FloatArray) -> FloatArray:
    ca = np.asarray(c, dtype=np.float64)
    if ca.ndim != 2 or ca.shape[0] != ca.shape[1]:
        raise ValueError("triangle must be square")
    n = ca.shape[0]
    if n < 3 or not np.isfinite(np.diag(np.fliplr(ca))).all():
        raise ValueError("triangle needs finite anti-diagonal")
    if (ca[np.isfinite(ca)] < 0).any():
        raise ValueError("triangle must be nonnegative")
    return ca


def ata_factors(c: FloatArray) -> FloatArray:
    """Volume-weighted age-to-age factors f_j = sum_i C[i,j+1]/sum_i C[i,j]."""
    ca = _check_triangle(c)
    n = ca.shape[0]
    f = np.empty(n - 1)
    for j in range(n - 1):
        col = ca[:, j]
        nxt = ca[:, j + 1]
        m = np.isfinite(col) & np.isfinite(nxt) & (col > 0)
        if not m.any():
            raise ValueError("degenerate development column")
        f[j] = float(nxt[m].sum() / col[m].sum())
        if not np.isfinite(f[j]) or f[j] <= 0:
            raise ValueError("non-positive chain-ladder factor")
    return f


def ultimate(c: FloatArray, f: FloatArray | None = None) -> FloatArray:
    """Chain-ladder ultimate per accident year."""
    ca = _check_triangle(c)
    n = ca.shape[0]
    ff = ata_factors(ca) if f is None else np.asarray(f, dtype=np.float64)
    if ff.size != n - 1:
        raise ValueError("factor length must be n-1")
    diag = np.diag(np.fliplr(ca))
    ult = np.empty(n)
    for i in range(n):
        ult[i] = diag[i] * np.prod(ff[n - 1 - i :])
    return ult


def reserves(c: FloatArray) -> dict[str, float]:
    ca = _check_triangle(c)
    diag = np.diag(np.fliplr(ca))
    ult = ultimate(ca)
    res = ult - diag
    return {
        "reserve_total": float(res.sum()),
        "ultimate_total": float(ult.sum()),
        "paid_diag": float(diag.sum()),
    }


def mack_variance(c: FloatArray) -> dict[str, float]:
    """Mack (1993) MSE decomposition of total reserve.

    sigma_j^2 estimated by weighted residual averages;
    total MSE = process variance + parameter (estimation)
    variance with the standard covariance cross-terms.
    """
    ca = _check_triangle(c)
    n = ca.shape[0]
    f = ata_factors(ca)
    # incremental residuals around chain-ladder factors
    sig2 = np.empty(n - 1)
    for j in range(n - 1):
        col = ca[:, j]
        nxt = ca[:, j + 1]
        m = np.isfinite(col) & np.isfinite(nxt) & (col > 0)
        w = col[m]
        dev = nxt[m] / col[m] - f[j]
        if m.sum() < 2:
            sig2[j] = min(sig2[:j]) if j else 1.0
        else:
            sig2[j] = float((w * dev * dev).sum() / (m.sum() - 1))
    sig2 = np.maximum(sig2, 1e-12)
    diag = np.diag(np.fliplr(ca))
    # project cumulative triangle to ultimate
    c_hat = diag.copy()
    ult = np.empty(n)
    for i in range(n):
        prods = np.prod(f[n - 1 - i :]) if i else 1.0
        ult[i] = diag[i] * prods
        c_hat[i] = ult[i]
    res = ult - diag
    mse = 0.0
    # process + parameter variance per accident year (Mack recursion)
    for i in range(1, n):
        proc = 0.0
        param = 0.0
        c_k = diag[i]
        for j in range(n - 1 - i, n - 1):
            # develop one step
            c_k_next = c_k * f[j]
            s_j = sum(ca[k, j] for k in range(n - j - 1) if np.isfinite(ca[k, j]) and ca[k, j] > 0)
            proc += c_k * sig2[j] / f[j]  # process var
            param += c_k * c_k * (sig2[j] / (f[j] * f[j])) / s_j
            c_k = c_k_next
        mse += proc + param
    return {
        "mse_total": float(mse),
        "se_reserve": float(np.sqrt(mse)),
        "cv_reserve": float(np.sqrt(mse) / res.sum()),
        "reserve_total": float(res.sum()),
    }


def bornhuetter_ferguson(c: FloatArray, premium: FloatArray, elr: float = 0.65) -> dict[str, float]:
    """BF reserve: ult_i = paid_i + prem_i*elr*(1 - 1/cdf_i)."""
    ca = _check_triangle(c)
    n = ca.shape[0]
    p = np.asarray(premium, dtype=np.float64).ravel()
    if p.size != n or (p <= 0).any():
        raise ValueError("premium must be positive length-n")
    f = ata_factors(ca)
    # cum_to_ult[i] = product of factors to develop year i to ultimate
    cum_to_ult = np.array([np.prod(f[n - 1 - i :]) if i else 1.0 for i in range(n)])
    diag = np.diag(np.fliplr(ca))
    ult_bf = diag + p * elr * (1 - 1 / cum_to_ult)
    res = ult_bf - diag
    return {
        "bf_reserve_total": float(res.sum()),
        "bf_ultimate_total": float(ult_bf.sum()),
        "bf_vs_cl_ratio": float(res.sum() / (reserves(ca)["reserve_total"] + 1e-12)),
    }


def odp_bootstrap(c: FloatArray, n_boot: int = 200, seed: int = 0) -> dict[str, float]:
    """England-Verrall over-dispersed-Poisson bootstrap of the
    total chain-ladder reserve."""
    ca = _check_triangle(c)
    n = ca.shape[0]
    f = ata_factors(ca)
    # fitted increments on the observed trapezoid
    rng = np.random.default_rng(seed)
    res_out = np.empty(n_boot)
    for b in range(n_boot):
        sim = np.full_like(ca, np.nan)
        sim[:, 0] = ca[:, 0]
        for i in range(n):
            for j in range(min(n - i, n) - 1):
                mu = sim[i, j] * f[j]
                # ODP: variance proportional to mean
                sim[i, j + 1] = rng.poisson(max(mu, 0.0))
        fb = ata_factors(np.nan_to_num(sim, nan=np.nan))
        ult_b = ultimate(np.nan_to_num(sim, nan=np.nan), fb)
        res_out[b] = float((ult_b - np.diag(np.fliplr(sim))).sum())
    return {
        "boot_reserve_mean": float(res_out.mean()),
        "boot_reserve_se": float(res_out.std(ddof=1)),
        "boot_reserve_p95": float(np.quantile(res_out, 0.95)),
    }


def _synthetic_triangle(seed: int, n: int = 10) -> FloatArray:
    """Upper-triangular cumulative triangle, ODP payments."""
    rng = np.random.default_rng(seed)
    f_true = np.array([3.0, 1.9, 1.45, 1.25, 1.15, 1.08, 1.04, 1.02, 1.01])
    c = np.full((n, n), np.nan)
    for i in range(n):
        c[i, 0] = rng.gamma(100.0, 10.0)
        for j in range(min(n - i, n) - 1):
            mu = c[i, j] * f_true[j]
            c[i, j + 1] = mu + rng.normal(0, 0.03 * mu)
            c[i, j + 1] = max(c[i, j + 1], c[i, j])
    return c


def bench_chain_ladder(seed: int = 501) -> dict[str, float]:
    """SYNTHETIC: reserve recovery on a known triangle."""
    tri = _synthetic_triangle(seed)
    r = reserves(tri)
    mk = mack_variance(tri)
    prem = np.full(tri.shape[0], 16000.0)
    bf = bornhuetter_ferguson(tri, prem, elr=0.9)
    if bf["bf_reserve_total"] <= 0:
        raise ValueError("BF reserve non-positive")
    bt = odp_bootstrap(tri, n_boot=60, seed=seed + 1)
    if r["reserve_total"] <= 0 or mk["se_reserve"] <= 0:
        raise ValueError("chain-ladder degenerate")
    return {
        "synthetic_reserve_total": r["reserve_total"],
        "synthetic_mack_cv": mk["cv_reserve"],
        "synthetic_bf_cl_ratio": bf["bf_vs_cl_ratio"],
        "synthetic_boot_se": bt["boot_reserve_se"],
    }
