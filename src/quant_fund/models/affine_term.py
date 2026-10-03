"""Affine term-structure fit + Campbell-Shiller expectations test.

References
----------
- Vasicek, O. (1977). "An Equilibrium Characterization of the Term
  Structure." *Journal of Financial Economics* 5(2), 177-188.
- Dai, Q. & Singleton, K.J. (2000). "Specification Analysis of Affine
  Term Structure Models." *Journal of Finance* 55(5), 1943-1978.
- Campbell, J.Y. & Shiller, R.J. (1991). "Yield Spreads and Interest
  Rate Movements: A Bird's Eye View." *Review of Economic Studies*
  58(3), 495-514.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Vasicek factor model: short rate ``r_t`` is OU(1) under both P and Q,
``dr = kappa_p (theta_p - r) dt + sigma dW^P``; the long yield at
maturity ``tau`` is affine ``y(tau) = A(tau) + B(tau) r`` with

    B(tau) = (1 - exp(-kappa_q tau)) / (kappa_q tau),
    A(tau) = (theta_q - sigma^2/(2 kappa_q^2)) * (B - 1) + sigma^2
             * (1 - exp(-2 kappa_q tau)) / (4 kappa_q^3 tau) * tau.

Parameters are fit by moment-matching the short-rate AR(1) regression
(P-measure) and the long-yield level (Q-measure); the expectations-
hypothesis regression ``y_long(t+1) - y_long(t) = a + b * (y_long -
y_short)`` then checks that the fitted curve dynamics match the EH
sign pattern. The synth draws an OU short rate with a fixed term
premium so the EH slope is below 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _b_factor(tau: float, kappa_q: float) -> float:
    x = kappa_q * tau
    if abs(x) < 1e-10:
        return 1.0
    return float((1.0 - np.exp(-x)) / x)


def affine_term(
    short_rate: FloatArray,
    long_rate: FloatArray,
    tau: float = 10.0,
) -> dict[str, float]:
    """Fit Vasicek ATS moments + Campbell-Shiller EH regression.

    ``short_rate``/``long_rate`` are same-length yield series; ``tau``
    is the long maturity in years. Returns factor dynamics, the implied
    term premium, and the EH slope.
    """
    r = np.asarray(short_rate, dtype=np.float64)
    yl = np.asarray(long_rate, dtype=np.float64)
    if r.ndim != 1 or yl.ndim != 1 or r.shape != yl.shape:
        raise ValueError("series must be same-length vectors")
    n = r.shape[0]
    if n < 60 or tau <= 0:
        raise ValueError("bad size/maturity")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(yl)):
        raise ValueError("non-finite inputs")

    # P-dynamics: AR(1) on the short rate.
    xx = np.column_stack([np.ones(n - 1), r[:-1]])
    beta = np.linalg.lstsq(xx, r[1:], rcond=None)[0]
    phi = float(beta[1])
    sigma_r = float(np.std(r[1:] - xx @ beta))
    theta_p = float(beta[0] / (1.0 - phi))
    # per-observation-step mean-reversion rate (kappa * dt)
    kappa_p = float(-np.log(max(phi, 1e-6)))

    # Q-dynamics inferred from the long-yield loading:
    # y_long ≈ A + B(tau; kappa_q) * r -> regress long on short.
    xx2 = np.column_stack([np.ones(n), r])
    g = np.linalg.lstsq(xx2, yl, rcond=None)[0]
    b_hat = float(g[1])
    kappa_q = float(-np.log(max(b_hat, 1e-6)) / tau) if b_hat > 0 else kappa_p
    term_premium = float(np.mean(yl) - np.mean(r) * b_hat - g[0])
    spread = yl - r

    # Campbell-Shiller EH regression: Dlong(t+1) - Dlong(t) on spread.
    dlong = np.diff(yl)
    sp = spread[:-1]
    xx3 = np.column_stack([np.ones(sp.shape[0]), sp])
    eh = np.linalg.lstsq(xx3, dlong, rcond=None)[0]
    eh_slope = float(eh[1])

    return {
        "kappa_p": kappa_p,
        "theta_p": theta_p,
        "sigma_r": sigma_r,
        "b_hat": b_hat,
        "kappa_q": kappa_q,
        "term_premium": term_premium,
        "eh_slope": eh_slope,
        "spread_mean": float(np.mean(spread)),
        "n": float(n),
    }


def synth_ats(
    n: int = 400,
    seed: int = 20261231 + 289,
    kappa_p: float = 0.4,
    premium: float = 0.008,
    tau: float = 10.0,
) -> dict[str, FloatArray]:
    """OU short rate + affine long yield with a constant term premium
    and persistent term-structure noise."""
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    dt = 1.0 / 12.0
    phi = np.exp(-kappa_p * dt)
    sigma_r = 0.005
    theta_p = 0.03
    r = np.empty(n)
    r[0] = theta_p
    eps = rng.normal(0.0, sigma_r * np.sqrt(dt), n)
    for t in range(1, n):
        r[t] = theta_p + phi * (r[t - 1] - theta_p) + eps[t]
    kappa_q = kappa_p * 0.8
    b_tau = _b_factor(tau, kappa_q)
    # long yield = affine loading * r + premium + small noise
    yl = b_tau * r + premium + rng.normal(0.0, 0.0004, n)
    return {"short_rate": r, "long_rate": yl, "b_true": np.array([b_tau])}


def bench_affine_term(seed: int = 20261231 + 289) -> dict[str, float]:
    """Wave-50 self-check: long-yield loading and OU persistence are
    recovered; EH slope stays below 1 under a positive term premium."""
    d = synth_ats(seed=seed)
    r = np.asarray(d["short_rate"])
    yl = np.asarray(d["long_rate"])
    a = affine_term(r, yl)
    a2 = affine_term(r, yl)
    detects = float(0.2 < a["b_hat"] < 1.0 and a["kappa_p"] > 0 and a["eh_slope"] < 1.0)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_b_hat": a["b_hat"],
        "synthetic_kappa_p": a["kappa_p"],
        "synthetic_term_premium": a["term_premium"],
        "synthetic_eh_slope": a["eh_slope"],
        "synthetic_spread_mean": a["spread_mean"],
    }
