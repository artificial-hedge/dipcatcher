"""Beveridge-Nelson permanent/transitory decomposition of I(1) series.

References
----------
- Beveridge, S. & Nelson, C.R. (1981). "A New Approach to
  Decomposition of Economic Time Series into Permanent and
  Transitory Components with Particular Attention to Measurement
  of the 'Business Cycle'." *Journal of Monetary Economics* 7(2),
  151-174.
- Kamber, G., Morley, J. & Wong, B. (2018). "Intuitive and Reliable
  Estimates of the Output Gap from a Beveridge-Nelson Filter."
  *Review of Economics and Statistics* 100(3), 550-566.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
For an ARIMA(p,1,0) series the BN permanent component is the
long-horizon forecast of the level plus drift:
``P_t = y_t + lim_{h->inf} (E_t y_{t+h} - h*mu - y_t)``. Writing
the AR(p) fit on differences ``dy`` in companion form
``Z_t = F Z_{t-1} + e_t`` with ``dy_t = g' Z_t``, the limit is the
Neumann series ``sum_{k>=1} F^k = F(I-F)^{-1}``, giving the exact
formula ``cycle_t = -g' F (I-F)^{-1} Z_t`` and
``perm_t = y_t - cycle_t`` (levels version; a nonzero drift is
absorbed by detrending dy before fitting and adding the drift
back into the permanent random walk). The filter is implemented
in closed form — no iterative forecasting — with a stationarity
guard on the AR roots (all inside the unit circle after the
unit root is differenced out). The synth plants a random walk
with drift plus an AR(1) transitory cycle, differences it, and
the recovered permanent component must track the planted
martingale (BN is exact for this DGP up to estimation error).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _fit_arp(y: FloatArray, p: int) -> tuple[FloatArray, float]:
    """OLS AR(p) with intercept; returns (coef, intercept)."""
    t = y.size
    if t <= p + 10:
        raise ValueError("series too short")
    rows = np.column_stack([np.ones(t - p)] + [y[p - k - 1 : t - k - 1] for k in range(p)])
    coef, *_ = np.linalg.lstsq(rows, y[p:], rcond=None)
    return np.asarray(coef[1:], dtype=np.float64), float(coef[0])


def _companion(phi: FloatArray) -> FloatArray:
    p = phi.size
    f = np.zeros((p, p))
    f[0, :] = phi
    f[1:, :-1] = np.eye(p - 1)
    return f


def beveridge_nelson(y: FloatArray, p: int = 2) -> dict[str, FloatArray | float]:
    """BN decomposition of an I(1) level series.

    Returns ``permanent`` (random-walk-with-drift component) and
    ``cycle`` (stationary transitory component, y - permanent),
    plus ``mu`` (drift), ``phi`` (AR coefficients on differences)
    and ``max_root`` (largest AR root modulus — must be < 1).
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 1 or yy.size < 60 or not np.all(np.isfinite(yy)):
        raise ValueError("bad series")
    if p < 1 or p > 6:
        raise ValueError("bad p")
    dy = np.diff(yy)
    phi, c = _fit_arp(dy, p)
    f = _companion(phi)
    roots = np.roots(np.concatenate([[1.0], -phi]))
    max_root = float(np.max(np.abs(roots)))
    if max_root >= 1.0:
        raise ValueError("nonstationary AR on differences")
    # state: Z_t = (dy_t - mu, ..., dy_{t-p+1} - mu)' ; g' = e1'
    t = dy.size
    mu = c / (1.0 - phi.sum())
    z = np.column_stack([dy[p - 1 - k : t - k] for k in range(p)]) - mu
    eye = np.eye(p)
    g = np.zeros(p)
    g[0] = 1.0
    # cycle_t = -g' F (I-F)^{-1} Z_t ; gain' = g' F (I-F)^{-1}
    gain = np.linalg.solve((eye - f).T, f.T @ g)
    cyc = -(z @ gain)
    n_out = cyc.size
    permanent = yy[p : p + n_out] - cyc
    return {
        "permanent": permanent,
        "cycle": cyc,
        "y": yy[p : p + n_out],
        "mu": float(c / (1.0 - phi.sum())),
        "phi": phi,
        "max_root": max_root,
    }


def synth_bn(
    seed: int = 20261231 + 304,
    t: int = 1500,
    mu: float = 0.004,
    rho_c: float = 0.4,
    sig_eps: float = 0.05,
    sig_cyc: float = 0.15,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC RW+drift (permanent) + AR(1) transitory cycle."""
    rng = np.random.default_rng(seed)
    eps = mu + sig_eps * rng.standard_normal(t)
    perm = np.cumsum(eps)
    cyc = np.empty(t)
    cyc[0] = 0.0
    e2 = sig_cyc * rng.standard_normal(t)
    for i in range(1, t):
        cyc[i] = rho_c * cyc[i - 1] + e2[i]
    return {
        "y": perm + cyc,
        "perm_true": perm,
        "cyc_true": cyc,
        "mu": mu,
        "rho_c": rho_c,
    }


def bench_beveridge_nelson(seed: int = 20261231 + 304) -> dict[str, float]:
    """Wave-52 self-check: permanent tracks the planted martingale."""
    d = synth_bn(seed=seed)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    perm_hat = np.asarray(r["permanent"])
    perm_true = np.asarray(d["perm_true"])[3:]
    cyc_true = np.asarray(d["cyc_true"])[3:]
    cyc_hat = np.asarray(r["cycle"])
    perm_err = float(np.sqrt(np.mean((perm_hat - perm_true) ** 2)))
    perm_scale = float(np.std(perm_true))
    cyc_corr = float(np.corrcoef(cyc_hat, cyc_true)[0, 1])
    # sign of serial correlation in recovered cycle should be +
    ok = perm_err / perm_scale < 0.5 and cyc_corr > 0.6
    return {
        "perm_err": perm_err,
        "perm_scale": perm_scale,
        "rel_err": perm_err / perm_scale,
        "cyc_corr": cyc_corr,
        "mu_hat": float(r["mu"]),
        "mu_true": float(d["mu"]),
        "max_root": float(r["max_root"]),
        "score": float(ok),
    }
