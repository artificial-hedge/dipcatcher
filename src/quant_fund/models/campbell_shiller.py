"""Campbell-Shiller log-linear VAR decomposition — Campbell (1991).

Under the log-linearization r_{t+1} ≈ kappa + rho * pd_{t+1} - pd_t
+ d_{t+1} (pd = log price-dividend ratio, rho = discount
1/(1+e^{pd_bar})), the unexpected return splits as

    e_{t+1} = eta_{d,t+1} - eta_{r,t+1}

where eta_d is news about current and discounted future dividend
growth and eta_r is news about discounted future returns:

    eta_r = e1' rho A (I - rho A)^{-1} u_{t+1}     (VAR(1) z_{t+1} =
                                                    A z_t + u)
    eta_d = e1' u_{t+1} + eta_r                    (residual identity)

and Var(e) = Var(eta_d) + Var(eta_r) - 2 Cov(eta_d, eta_r) holds
exactly — shares sum to 1 by construction.

References
----------
- Campbell & Shiller (1988) Rev. Fin. Stud., log-linearization.
- Campbell (1991) J. Finance, "A variance decomposition for stock
  returns".

Honesty
-------
VAR(1) on SYNTHETIC log-returns only; variance shares are computed
from the fitted A matrix — no real-market claim.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_campbell_shiller``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def var_decompose(z: FloatArray, rho: float = 0.96) -> dict[str, FloatArray | float]:
    """Fit VAR(1) on z=[r, d_growth, pd]; decompose return variance.

    Returns innovation news eta_d (dividend news, residual), eta_r
    (discount-rate news) and the variance shares of each.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 3 or z.shape[0] < 20:
        raise ValueError("z must be (T,3) with T>=20")
    if not (0.5 < rho < 1.0):
        raise ValueError("rho must be in (0.5, 1)")
    zc = z - z.mean(axis=0)
    a = np.linalg.lstsq(zc[:-1], zc[1:], rcond=None)[0].T  # z_{t+1} = A z_t
    eig = float(np.max(np.abs(np.linalg.eigvals(a))))
    if eig >= 1.0:
        raise ValueError("VAR not stable")
    eps = (zc[1:] - zc[:-1] @ a.T).T  # innovations u_{t+1}, shape (3, T-1)
    e1 = np.eye(3)[0]
    eta_r = (e1 @ (rho * a) @ np.linalg.inv(np.eye(3) - rho * a)) @ eps
    e_h = e1 @ eps
    eta_d = e_h + eta_r  # Campbell residual identity
    var_e = float(np.var(e_h, ddof=1))
    return {
        "eta_d": eta_d,
        "eta_r": eta_r,
        "var_e": var_e,
        "spec_rad": eig,
        "share_div": float(np.var(eta_d, ddof=1) / var_e),
        "share_disc": float(np.var(eta_r, ddof=1) / var_e),
        "share_cov": float(-2.0 * np.cov(eta_d, eta_r)[0, 1] / var_e),
    }


def bench_campbell_shiller(seed: int = 20261231 + 377) -> dict[str, float]:
    """SYNTHETIC check — decomposition shares sum to one; plausible split."""
    rng = np.random.default_rng(seed)
    n = 20000
    # DGP: expected dividend growth mean-reverts, driving pd_t; return
    # innovations split mostly toward dividend news but with a nonzero
    # discount-rate share.
    g = np.zeros(n)
    mu_g = np.zeros(n)
    for t in range(1, n):
        mu_g[t] = 0.9 * mu_g[t - 1] + 0.02 * rng.standard_normal()
        g[t] = 0.02 + mu_g[t] + 0.10 * rng.standard_normal()
    rho = 0.96
    pd_t = mu_g * rho / (1.0 - 0.9 * rho) + 0.05 * rng.standard_normal(n)
    r = np.zeros(n)
    r[1:] = -pd_t[:-1] + rho * pd_t[1:] + g[1:]
    z = np.column_stack([r[1:], g[1:], pd_t[1:]])
    out = var_decompose(z, rho=rho)
    s = float(out["share_div"]) + float(out["share_disc"]) + float(out["share_cov"])
    if abs(s - 1.0) > 1e-6:
        raise ValueError("variance shares fail the accounting identity")
    if not (0.3 < float(out["share_div"]) < 0.98):
        raise ValueError("dividend share implausible under this DGP")
    if not (0.0 < float(out["share_disc"]) < 0.7):
        raise ValueError("discount-rate share implausible")
    corr = float(np.corrcoef(out["eta_d"], out["eta_r"])[0, 1])
    return {
        "synthetic_cs_share_div": float(out["share_div"]),
        "synthetic_cs_share_disc": float(out["share_disc"]),
        "synthetic_cs_share_cov": float(out["share_cov"]),
        "synthetic_cs_corr_news": corr,
        "synthetic_score": 1.0,
    }
