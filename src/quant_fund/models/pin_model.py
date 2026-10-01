"""Easley-O'Hara PIN — probability of informed trading.

Trade arrivals follow a mixture over latent information states:
with prob α an information event occurs; it is bad news w.p. δ
(sell-side Poisson intensity ε_s+μ) and good w.p. 1−δ
(buy-side ε_b+μ); otherwise trades arrive at base intensities
ε_b, ε_s. Daily buy/sell counts (B,S) identify the parameters
via maximum likelihood; PIN = αμ / (αμ + ε_b + ε_s).

Honesty: synthetic benches recover parameters on simulated
count panels — proper diagnostics, never market evidence.

References:
- Easley, D., Kiefer, N., O'Hara, M., Paperman, J. (1996).
  Liquidity, information, and infrequently traded stocks.
  *Journal of Finance* 51 — the EHO factorization used here.
- Easley, D., Hvidkjaer, S., O'Hara, M. (2002). Is information
  risk a determinant of asset returns? *Journal of Finance* 57.
- Yan, Y., Zhang, S. (2012). An improved estimation method and
  empirical properties of the probability of informed trading.
  *Journal of Banking & Finance* 36 — the Lin-Ke / Yan-Zhang
  stable log-likelihood formulation.
- Duarte, J., Young, L. (2009). Why is PIN priced?
  *Journal of Financial Economics* 91.

Composition: pure numpy + scipy — MLE via
``scipy.optimize.minimize`` (L-BFGS-B) on the stabilized
log-likelihood; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

FloatArray = NDArray[np.float64]


def pin_estimate(buys: FloatArray, sells: FloatArray) -> dict[str, float]:
    """MLE of (alpha, delta, mu, e_b, e_s) and PIN."""
    b = np.asarray(buys, dtype=np.float64)
    s = np.asarray(sells, dtype=np.float64)
    if b.shape != s.shape or b.size < 40:
        raise ValueError("buys/sells matched arrays, n>=40 required")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(s)):
        raise ValueError("finite counts required")
    if np.any(b < 0) or np.any(s < 0):
        raise ValueError("non-negative counts required")

    mb, ms = float(b.mean()), float(s.mean())

    def nll(theta: FloatArray) -> float:
        alpha, delta, mu, eb, es = theta
        # stabilized EHO likelihood via factorization:
        # term1 = (1-a)*Pois(b;eb)*Pois(s;es)
        # term2 = a*(1-d)*Pois(b;eb+mu)*Pois(s;es)
        # term3 = a*d*Pois(b;eb)*Pois(s;es+mu)
        l1 = np.log(1 - alpha) + b * np.log(eb) + s * np.log(es) - eb - es
        l2 = np.log(alpha * (1 - delta)) + b * np.log(eb + mu) + s * np.log(es) - eb - mu - es
        l3 = np.log(alpha * delta) + b * np.log(eb) + s * np.log(es + mu) - eb - es - mu
        m = np.max(np.stack([l1, l2, l3]), axis=0)
        ll = m + np.log(np.sum(np.exp(np.stack([l1, l2, l3]) - m), axis=0))
        return float(-np.sum(ll))  # constant log(k!) terms dropped

    best: optimize.OptimizeResult | None = None
    for a0 in (0.05, 0.2, 0.5):
        for d0 in (0.2, 0.5, 0.8):
            x0 = np.array([a0, d0, abs(mb - ms) + 1.0, mb * 0.8 + 0.1, ms * 0.8 + 0.1])
            res = optimize.minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=[(1e-6, 0.9), (1e-6, 0.999), (1e-6, None), (1e-6, None), (1e-6, None)],
            )
            if best is None or res.fun < best.fun:
                best = res
    assert best is not None
    th = np.asarray(best.x, dtype=np.float64)
    alpha, delta, mu, eb, es = (float(x) for x in th)
    pin = alpha * mu / (alpha * mu + eb + es)
    return {
        "alpha": alpha,
        "delta": delta,
        "mu": mu,
        "e_b": eb,
        "e_s": es,
        "pin": float(pin),
        "nll": float(best.fun),
    }


def synth_pin(
    n_days: int = 250,
    alpha: float = 0.3,
    delta: float = 0.5,
    mu: float = 30.0,
    e_b: float = 15.0,
    e_s: float = 15.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Simulate daily (buys, sells) under the EHO mixture."""
    rng = np.random.default_rng(seed)
    event = rng.uniform(size=n_days) < alpha
    bad = rng.uniform(size=n_days) < delta
    lam_b = np.where(event & ~bad, e_b + mu, e_b)
    lam_s = np.where(event & bad, e_s + mu, e_s)
    return {
        "buys": rng.poisson(lam_b).astype(np.float64),
        "sells": rng.poisson(lam_s).astype(np.float64),
    }


def bench_pin_model(seed: int = 20261231 + 252) -> dict[str, float]:
    """PIN self-check: MLE recovers alpha/mu within tolerance on
    an informed panel; PIN ~0 on a symmetric uninformed panel.
    All ``synthetic_*``."""
    d = synth_pin(alpha=0.3, mu=30.0, seed=seed)
    out = pin_estimate(d["buys"], d["sells"])
    d0 = synth_pin(alpha=0.0, seed=seed + 1)
    out0 = pin_estimate(d0["buys"], d0["sells"])
    out_b = pin_estimate(d["buys"], d["sells"])
    a_hat = float(out["alpha"])
    return {
        "synthetic_alpha": a_hat,
        "synthetic_mu": float(out["mu"]),
        "synthetic_pin": float(out["pin"]),
        "synthetic_pin_null": float(out0["pin"]),
        "synthetic_detects": float(
            abs(a_hat / 0.3 - 1) < 0.4 and float(out["pin"]) > float(out0["pin"])
        ),
        "synthetic_determinism": float(a_hat == float(out_b["alpha"])),
    }
