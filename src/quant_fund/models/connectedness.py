"""Diebold-Yilmaz connectedness index from VAR forecast-error
variance decomposition (generalized FEVD).

Fit a VAR(p) on a multivariate series, decompose each variable's
H-step-ahead forecast error variance into shares attributable to
each shock (Pesaran-Shin generalized FEVD — ordering-invariant),
row-normalize, and read off total/directional connectedness.
Captures volatility/return spillover structure that pairwise
correlations miss.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure spillover recovery on
generated VAR systems — never market evidence.

References:
- Diebold, F. X., Yilmaz, K. (2012). Better to give than to
  receive: predictive directional measurement of volatility
  spillovers. *Int. J. Forecasting* 28, 57-66.
- Diebold, F. X., Yilmaz, K. (2014). On the network topology of
  variance decompositions. *J. Econometrics* 182, 119-134 —
  connectedness table and total connectedness index.
- Pesaran, H. H., Shin, Y. (1998). Generalized impulse response
  analysis in linear multivariate models. *Economics Letters*
  58 — ordering-invariant FEVD.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time
  Series Analysis* — companion-form FEVD recursion.

Composition: pure numpy — VAR OLS, companion-matrix impulse
response recursion, generalized FEVD; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _var_fit(y: FloatArray, p: int) -> tuple[FloatArray, FloatArray]:
    """VAR(p) OLS: y_t = c + Σ_l A_l y_{t-l} + e_t.

    Returns (A stacked (k × k·p), Σ_e)."""
    t, k = y.shape
    dep = y[p:]
    reg = np.ones((t - p, 1))
    for lag in range(1, p + 1):
        reg = np.column_stack([reg, y[p - lag : t - lag]])
    b = np.linalg.lstsq(reg, dep, rcond=None)[0]
    resid = dep - reg @ b
    sigma = (resid.T @ resid) / (t - p)
    return b[1:].T, sigma  # drop intercept row


def connectedness(
    y: FloatArray,
    p: int = 1,
    h: int = 10,
) -> dict[str, float]:
    """Diebold-Yilmaz table for a (T × k) system.

    Returns total connectedness index (mean off-diagonal
    normalized generalized FEVD share), the largest pairwise
    directional share, and VAR diagnostics."""
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2:
        raise ValueError("y must be (T × k)")
    t, k = yy.shape
    if t < 100 or k < 2 or k > 8:
        raise ValueError("T>=100, k in 2..8")
    if not (1 <= p <= 3):
        raise ValueError("p in 1..3")
    if not (5 <= h <= 40):
        raise ValueError("h in 5..40")
    if not np.all(np.isfinite(yy)):
        raise ValueError("finite inputs required")
    if np.min(np.std(yy, axis=0)) < 1e-9:
        raise ValueError("each series must vary")

    a_stack, sigma = _var_fit(yy, p)
    # companion form: F = [A_1..A_p; I 0] (k·p × k·p)
    a = [a_stack[:, lag * k : (lag + 1) * k] for lag in range(p)]
    f = np.zeros((k * p, k * p))
    f[:k, :] = np.hstack(a)
    f[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    # stability check
    eig = np.max(np.abs(np.linalg.eigvals(f)))
    if eig >= 1.0:
        raise ValueError("estimated VAR is non-stationary")

    j_sel = np.zeros((k, k * p))
    j_sel[:, :k] = np.eye(k)
    d = np.sqrt(np.diag(sigma))

    # generalized FEVD: θ_ij(h) = σ_jj⁻¹ Σ_h' (e_i' Φ_h' Σ e_j)²
    #                             / Σ_h' (e_i' Φ_h' Σ Φ_h e_i)
    theta = np.zeros((k, k))
    psi_list = [np.eye(k * p)]
    for _ in range(1, h):
        psi_list.append(psi_list[-1] @ f)
    for i in range(k):
        num = np.zeros(k)
        den = 0.0
        ei = np.zeros(k)
        ei[i] = 1.0
        for s in range(h):
            psi = j_sel @ psi_list[s] @ j_sel.T
            row = ei @ psi @ sigma  # (k,) — e_i' Ψ_s Σ
            num += row**2 / (d**2)  # σ_jj⁻¹ (e_i' Ψ_s Σ e_j)²
            den += float(ei @ psi @ sigma @ psi.T @ ei)
        theta[i] = num / max(den, 1e-300)
    # row-normalize so shares sum to 1 (as in DY2014 table)
    theta_norm = theta / theta.sum(axis=1, keepdims=True)

    off = theta_norm.copy()
    np.fill_diagonal(off, 0.0)
    # TCI = mean off-diagonal share (×100 for the index scale)
    tci = float(np.sum(off) / k) * 100.0
    # directional: θ_ij = share of i's variance from j's shock →
    # row i = from-others (received), column j = to-others (given)
    to_others = off.sum(axis=0)
    from_others = off.sum(axis=1)
    net = to_others - from_others
    i_max = int(np.unravel_index(np.argmax(off), off.shape)[0])
    j_max = int(np.unravel_index(np.argmax(off), off.shape)[1])

    return {
        "t": float(t),
        "k": float(k),
        "tci": tci,
        "max_pairwise": float(off[i_max, j_max]),
        "net_0": float(net[0]),
        "net_1": float(net[1]),
        "rho_01_resid": float(sigma[0, 1] / (d[0] * d[1])),
        "max_eig": float(eig),
    }


def synth_var_spill(
    t: int = 600,
    spill: float = 0.5,
    rho_resid: float = 0.0,
    seed: int = 0,
) -> FloatArray:
    """2-var VAR(1): y1 AR(0.5); y2 loads on y1's shock with
    coefficient ``spill`` (one-way spillover y1 → y2)."""
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0.0, 1.0, t)
    e2 = rho_resid * e1 + np.sqrt(max(1e-9, 1 - rho_resid**2)) * rng.normal(0.0, 1.0, t)
    y = np.zeros((t, 2))
    for i in range(1, t):
        y[i, 0] = 0.5 * y[i - 1, 0] + e1[i]
        y[i, 1] = 0.3 * y[i - 1, 1] + spill * (e1[i]) + e2[i]
    return y


def bench_connectedness(seed: int = 20261231 + 230) -> dict[str, float]:
    """Connectedness self-check: one-way spillover detected in the
    FEVD table; an independent VAR shows near-zero TCI.
    All ``synthetic_*``."""
    y = synth_var_spill(spill=0.6, seed=seed)
    out = connectedness(y, p=1, h=10)
    y0 = synth_var_spill(spill=0.0, seed=seed + 1)
    out0 = connectedness(y0, p=1, h=10)
    out_b = connectedness(y, p=1, h=10)

    tci = float(out["tci"])
    tci0 = float(out0["tci"])
    return {
        "synthetic_tci": tci,
        "synthetic_tci_indep": tci0,
        "synthetic_tci_gap": tci - tci0,
        "synthetic_max_pairwise": float(out["max_pairwise"]),
        "synthetic_net_transmitter": float(out["net_0"]),
        "synthetic_max_eig": float(out["max_eig"]),
        "synthetic_detects": float(tci > tci0 + 5.0 and float(out["net_0"]) > 0.0),
        "synthetic_determinism": float(tci == float(out_b["tci"])),
    }
