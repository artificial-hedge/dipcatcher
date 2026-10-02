"""Risk-parity / equal-risk-contribution portfolios.

Roncalli (2013): the risk contribution of asset i to
sigma(w) = sqrt(w' S w) is

    RC_i = w_i (S w)_i / sigma(w).

The ERC portfolio solves w_i (S w)_i = sigma^2(w)/n for all i,
which is the (unique, up to normalization) fixed point of the
convex program min_w w' S w - c sum_i ln w_i (Spinu 2013).
We solve it by cyclical coordinate descent on the logarithmic
barrier (Griveau-Billion, Richard & Roncalli 2013) — each
asset's weight has the closed-form update

    w_i = [-(w' S)_i^{0} + sqrt((w' S)_i^{0 2} + 4 S_ii c)] / (2 S_ii)

and the unconstrained program is then rescaled to sum(w) = 1.
Concentration diagnostics report the Herfindahl of RC shares
and the effective number of risk bets.

Honesty: covariance input must be symmetric PD — fail closed
on non-PD or near-singular S (verified via eigh); the solver
uses the log-barrier form which requires positive weights, so
the ERC solution here is the long-only one (documented; no
leverage/shorting claim). The bench plants a diagonal-ish
covariance with heteroskedastic vols and checks RC shares
flatten toward 1/n. Fail-closed on singular S.

References: Roncalli (2013) "Introduction to Risk Parity and
Budgeting"; Maillard, Roncalli & Teiletche (2010) J. Portfolio
Mgmt 36:60; Spinu (2013) J. Comp. Finance; Griveau-Billion,
Richard & Roncalli (2013) SSRN; Chaves, Hsu, Li & Shakernia
(2012) J. Inv. Strategies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_cov(cov: FloatArray) -> FloatArray:
    s = np.asarray(cov, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 2:
        raise ValueError("cov must be square k>=2")
    if not np.allclose(s, s.T, atol=1e-8):
        raise ValueError("cov must be symmetric")
    eig = np.linalg.eigvalsh(s)
    if float(eig.min()) <= 1e-12:
        raise ValueError("cov must be positive definite")
    return s


def _risk_contrib(w: FloatArray, s: FloatArray) -> FloatArray:
    sig = math.sqrt(max(float(w @ s @ w), 1e-20))
    return np.asarray(w * (s @ w) / sig, dtype=np.float64)


def erc_weights(
    cov: FloatArray,
    budget: FloatArray | None = None,
    n_iter: int = 500,
) -> dict[str, float | FloatArray]:
    """Equal (or budgeted) risk-contribution weights via
    coordinate descent on the log-barrier objective."""
    s = _check_cov(cov)
    n = s.shape[0]
    b = np.ones(n) / n if budget is None else np.asarray(budget, dtype=np.float64)
    if b.size != n or (b <= 0).any() or not np.isfinite(b).all():
        raise ValueError("budget must be positive")
    b = b / b.sum()
    w = np.ones(n)
    c_const = 1.0
    for _ in range(n_iter):
        w_old = w.copy()
        for i in range(n):
            sw = s @ w
            a = s[i, i]
            b0 = sw[i] - a * w[i]
            # closed-form CCD step for log-barrier b_i * c
            w[i] = (-b0 + math.sqrt(b0 * b0 + 4.0 * a * b[i] * c_const)) / (2.0 * a)
        if float(np.abs(w - w_old).max()) < 1e-12:
            break
    w = w / w.sum()
    rc = _risk_contrib(w, s)
    share = rc / rc.sum()
    herf = float((share * share).sum())
    return {
        "w": np.asarray(w, dtype=np.float64),
        "rc_share": np.asarray(share, dtype=np.float64),
        "herfindahl_rc": herf,
        "enrb": 1.0 / herf,
        "max_rc_dev": float(np.abs(share - b).max()),
        "sigma": float(math.sqrt(w @ s @ w)),
    }


def concentrated_parity(cov: FloatArray, n_iter: int = 300) -> dict[str, float | FloatArray]:
    """Inverse-vol (naive parity) baseline for comparison."""
    s = _check_cov(cov)
    vols = np.sqrt(np.diag(s))
    w = (1.0 / vols) / (1.0 / vols).sum()
    rc = _risk_contrib(w, s)
    share = rc / rc.sum()
    return {
        "w": np.asarray(w, dtype=np.float64),
        "rc_share": np.asarray(share, dtype=np.float64),
        "enrb": float(1.0 / ((share * share).sum())),
    }


def bench_risk_parity(seed: int = 20261231 + 459) -> dict[str, float]:
    """SYNTHETIC check — ERC flattens RC shares on hetero vols."""
    rng = np.random.default_rng(seed)
    n = 5
    a = rng.normal(size=(n, n))
    corr = a @ a.T
    d = np.diag(1.0 / np.sqrt(np.diag(corr)))
    r_mat = d @ corr @ d
    vols = np.array([0.05, 0.10, 0.15, 0.20, 0.30])
    cov = np.diag(vols) @ r_mat @ np.diag(vols)
    out = erc_weights(cov)
    share = np.asarray(out["rc_share"], dtype=np.float64)
    dev = float(out["max_rc_dev"])
    if dev > 0.01:
        raise ValueError(f"erc off: rc shares {np.round(share, 3).tolist()}")
    # inverse-vol baseline should deviate more
    naive = concentrated_parity(cov)
    naive_dev = float(np.abs(np.asarray(naive["rc_share"]) - 1.0 / n).max())
    if naive_dev <= dev:
        raise ValueError("naive parity unexpectedly better than ERC")
    return {
        "synthetic_max_dev": dev,
        "synthetic_naive_dev": naive_dev,
        "score": 1.0,
    }
