"""Melick-Thomas implied PDF recovery with disaster component.

References
----------
- Melick, W.R. & Thomas, C.P. (1997). "Recovering an Asset's
  Implied PDF from Option Prices: An Application to Crude Oil
  during the Gulf Crisis." *Journal of Financial and Quantitative
  Analysis* 32(1), 91-115.
- Black, F. & Scholes, M. (1973). "The Pricing of Options and
  Corporate Liabilities." *JPE* 81(3), 637-654.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Melick-Thomas recover the risk-neutral PDF as a weighted mixture
of lognormal components plus an elicited jump/disaster mass, fit
by minimizing squared call-price errors subject to the martingale
constraint that the mixture mean equals the forward. The
implementation here fits a 3-component lognormal mixture
(component i: weight w_i, log-mean m_i, log-sd s_i) whose
risk-neutral mean is pinned at the forward — w is in the open
simplex and s_i bounded — via ``scipy.least_squares`` on pricing
residuals plus the left-tail disaster mass
``P(S_T < q_lo*F)`` read off the fitted CDF. The synth prices a
planted mixture (with a fat low component) exactly, and the fit
must recover the left-tail mass within tolerance; the planted
weights/means themselves are weakly identified (mixture geometry
is degenerate), so the check pins the disaster mass and the
implied variance — the quantities Melick-Thomas actually read.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt
from scipy import special as _special

FloatArray = NDArray[np.float64]

_NC = 3


def _norm_cdf(z: FloatArray | float) -> FloatArray:
    return np.asarray(0.5 * (1.0 + _special.erf(np.asarray(z) / np.sqrt(2.0))))


def _bs_call_mixture(
    strikes: FloatArray,
    fwd: float,
    tau: float,
    weights: FloatArray,
    means: FloatArray,
    sds: FloatArray,
) -> FloatArray:
    """Undiscounted call prices under a lognormal mixture.

    Component i has log-mean ``means[i]`` (on ln S_T, level, with
    mean shifted so each component's own mean equals ``fwd`` for
    identifiability we fold the martingale constraint into
    ``means`` upstream) and log-sd ``sds[i]``; the call is
    ``sum_i w_i [e^{m_i+s_i^2/2} N(d1) - K N(d2)]`` where
    ``d1 = (m_i - ln K + s_i^2)/s_i``, ``d2 = d1 - s_i``.
    """
    k = np.asarray(strikes, dtype=np.float64)
    ln_k = np.log(k)
    out = np.zeros_like(k)
    comp_mean = np.exp(means + 0.5 * sds * sds)
    d1 = (means[:, None] - ln_k[None, :] + sds[:, None] ** 2) / sds[:, None]
    d2 = d1 - sds[:, None]
    calls = comp_mean[:, None] * _norm_cdf(d1) - k[None, :] * _norm_cdf(d2)
    out = np.sum(weights[:, None] * calls, axis=0)
    return out


def implied_pdf(
    strikes: FloatArray,
    call_prices: FloatArray,
    spot: float,
    rate: float,
    tau: float,
) -> dict[str, float | FloatArray]:
    """Fit the Melick-Thomas 3-lognormal mixture to a call smile.

    Raises ValueError on degenerate inputs. Returns the fitted
    mixture parameters, the left-tail disaster mass below
    ``0.7 * forward``, the implied variance of ``ln S_T``, and the
    RMS pricing error.
    """
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(call_prices, dtype=np.float64)
    if k.ndim != 1 or k.size != c.size or k.size < 8:
        raise ValueError("need >=8 strikes")
    if (
        not np.all(np.isfinite(k))
        or not np.all(np.isfinite(c))
        or np.any(k <= 0.0)
        or np.any(c < 0.0)
        or spot <= 0.0
        or tau <= 0.0
    ):
        raise ValueError("bad inputs")
    fwd = spot * float(np.exp(rate * tau))

    # parameter vector: [w1, w2, m1, m2, m3, ln s1, ln s2, ln s3]
    # w3 = 1 - w1 - w2 enforced via softmax-free bound trick:
    # use stick-breaking weights w_i = u_i / sum(u), u>0 -> reparam
    def unpack(p: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        u = np.exp(np.clip(p[:_NC], -8.0, 8.0))
        w = u / u.sum()
        m = p[_NC : 2 * _NC]
        s = np.exp(np.clip(p[2 * _NC :], -4.0, 1.0))
        return w, m, s

    def resid(p: FloatArray) -> FloatArray:
        w, m, s = unpack(p)
        # martingale constraint folded in: shift component means so
        # the mixture mean equals fwd
        comp_mean = np.exp(m + 0.5 * s * s)
        mix_mean = float(w @ comp_mean)
        m_adj = m - (np.log(mix_mean) - np.log(fwd))
        pred = _bs_call_mixture(k, fwd, tau, w, m_adj, s)
        return (pred - c) / max(fwd * 0.01, 1e-8)

    p0 = np.array([0.0, 0.0, 0.0] + [np.log(fwd)] * _NC + [-1.0, -1.0, -0.2])
    best: _opt.OptimizeResult | None = None
    rng = np.random.default_rng(0)
    for start in range(4):
        p_init = p0.copy()
        if start:
            p_init[_NC : 2 * _NC] += rng.standard_normal(_NC) * 0.1
            p_init[:_NC] += rng.standard_normal(_NC) * 0.5
            p_init[2 * _NC :] += rng.standard_normal(_NC) * 0.3
        res = _opt.least_squares(resid, p_init, method="lm", max_nfev=4000)
        if best is None or float(res.cost) < float(best.cost):
            best = res
    if not (best is not None):
        raise ValueError("best is not None")
    w, m, s = unpack(np.asarray(best.x))
    comp_mean = np.exp(m + 0.5 * s * s)
    mix_mean = float(w @ comp_mean)
    m_adj = m - (np.log(mix_mean) - np.log(fwd))
    # implied variance of ln S_T: E[m^2] - (E m)^2
    e1 = float(w @ (m_adj + 0.5 * s * s))  # E[ln S] proxy via comp
    # exact: for component, E[ln S] = m_i, Var = s_i^2
    e_ln = float(w @ m_adj)
    e_ln2 = float(w @ (m_adj * m_adj + s * s))
    var_ln = e_ln2 - e_ln * e_ln
    # left-tail disaster mass below 0.7*fwd
    q = np.log(0.7 * fwd)
    tail_mass = float(np.sum(w * _norm_cdf((q - m_adj) / s)))
    pred = _bs_call_mixture(k, fwd, tau, w, m_adj, s)
    rmse = float(np.sqrt(np.mean((pred - c) ** 2)) / fwd)
    _ = e1
    return {
        "weights": w,
        "means": m_adj,
        "sds": s,
        "fwd": fwd,
        "var_ln": var_ln,
        "tail_mass": tail_mass,
        "rmse_rel": rmse,
        "cost": float(best.cost),
    }


def synth_smile(
    seed: int = 20261231 + 301,
    n: int = 60,
    disaster_w: float = 0.15,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC smile priced by a planted fat-left-tail mixture."""
    rng = np.random.default_rng(seed)
    _ = rng  # deterministic shapes
    spot, rate, tau = 100.0, 0.02, 0.5
    fwd = spot * float(np.exp(rate * tau))
    weights = np.array([disaster_w, 0.55, 1.0 - disaster_w - 0.55])
    means = np.log(fwd) + np.array([-0.35, 0.02, 0.12])
    sds = np.array([0.55, 0.22, 0.18])
    # pin the martingale mean exactly
    comp_mean = np.exp(means + 0.5 * sds * sds)
    means = means - (np.log(float(weights @ comp_mean)) - np.log(fwd))
    strikes = np.linspace(0.4 * fwd, 2.2 * fwd, n)
    calls = _bs_call_mixture(strikes, fwd, tau, weights, means, sds)
    q = np.log(0.7 * fwd)
    tail_true = float(np.sum(weights * _norm_cdf((q - means) / sds)))
    var_true = float(weights @ (means * means + sds * sds) - (weights @ means) ** 2)
    return {
        "strikes": strikes,
        "calls": calls,
        "spot": spot,
        "rate": rate,
        "tau": tau,
        "tail_true": tail_true,
        "var_true": var_true,
    }


def bench_melick_thomas(seed: int = 20261231 + 301) -> dict[str, float]:
    """Wave-52 self-check: recovers disaster mass + implied variance."""
    d = synth_smile(seed=seed)
    r = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    tail_err = abs(float(r["tail_mass"]) - float(d["tail_true"]))
    var_err = abs(float(r["var_ln"]) - float(d["var_true"])) / float(d["var_true"])
    ok = tail_err < 0.06 and var_err < 0.25 and float(r["rmse_rel"]) < 0.01
    return {
        "synthetic_tail_hat": float(r["tail_mass"]),
        "synthetic_tail_true": float(d["tail_true"]),
        "synthetic_tail_err": tail_err,
        "synthetic_var_err_rel": var_err,
        "synthetic_rmse_rel": float(r["rmse_rel"]),
        "synthetic_score": float(ok),
    }
