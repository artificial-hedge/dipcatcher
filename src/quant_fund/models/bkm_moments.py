"""Bakshi-Kapadia-Madan model-free risk-neutral moments from option prices.

References
----------
- Bakshi, G., Kapadia, N. & Madan, D. (2003). "Stock Return
  Characteristics, Skew Laws, and the Differential Pricing of
  Individual Equity Options." *Review of Financial Studies* 16(1),
  101-143.
- Bali, T.G., Hu, J. & Murray, S. (2019). "Option Implied Volatility,
  Skewness, and Kurtosis and the Cross-Section of Expected Stock
  Returns." Georgetown McDonough School of Business Research Paper
  (trapezoidal discretization, Appendix B).
- Carr, P. & Madan, D. (2001). "Optimal Positioning in Derivative
  Securities." *Quantitative Finance* 1(1), 19-37.
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). "A Guide to
  Volatility and Variance Swaps." *Journal of Derivatives* 6(4), 9-32.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The quadratic, cubic, and quartic contracts pay the second, third,
and fourth powers of the log return ``R = ln(S_T/S_t)``. Carr-Madan
spanning about the spot ``S_t`` prices them from a cross-section of
out-of-the-money options with signed kernels

    k2(K) = 2(1 - ln(K/S))/K^2
    k3(K) = (6 ln(K/S) - 3 ln(K/S)^2)/K^2
    k4(K) = (12 ln(K/S)^2 - 4 ln(K/S)^3)/K^2,

so the cubic contract is automatically long OTM calls and short OTM
puts (the kernel is negative for K < S). The legs integrate
``k_n(K) * option_price(K)`` trapezoidally, puts over K < S, calls
over K >= S. With V, W, X the resulting contract prices,

    mu    = e^{r t} - 1 - (e^{r t}/2) V - (e^{r t}/6) W - (e^{r t}/24) X
    varQ  = e^{r t} V - mu^2
    skewQ = (e^{r t} W - 3 mu e^{r t} V + 2 mu^3) / varQ^{3/2}
    kurtQ = (e^{r t} X - 4 mu e^{r t} W + 6 mu^2 e^{r t} V - 3 mu^4)
            / varQ^2,

where ``mu`` is the BKM approximation to ``e^{r t} E^Q[R]`` from the
fourth-order expansion of ``e^R``. The synth draws strikes around a
two-component lognormal-mixture risk-neutral density, prices the
options analytically per component, and checks the recovered moments
against the closed-form moments of the mixture.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import special as _special

FloatArray = NDArray[np.float64]


def _kernel(strikes: FloatArray, spot: float, order: int) -> FloatArray:
    lk = np.log(strikes / spot)
    k2 = strikes * strikes
    if order == 2:
        return 2.0 * (1.0 - lk) / k2
    if order == 3:
        return (6.0 * lk - 3.0 * lk * lk) / k2
    if order == 4:
        return (12.0 * lk * lk - 4.0 * lk * lk * lk) / k2
    raise ValueError("order must be 2, 3, or 4")


def _trapz(strikes: FloatArray, values: FloatArray) -> float:
    if strikes.size < 2:
        return 0.0
    dK = np.diff(strikes)
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * dK))


def _contracts(
    strikes: FloatArray,
    prices: FloatArray,
    is_call: FloatArray,
    spot: float,
) -> tuple[float, float, float]:
    puts = ~is_call.astype(bool)
    v = w = x = 0.0
    for mask in (puts, is_call.astype(bool)):
        k = strikes[mask]
        if k.size == 0:
            continue
        order = np.argsort(k)
        k = k[order]
        p = prices[mask][order]
        v += _trapz(k, _kernel(k, spot, 2) * p)
        w += _trapz(k, _kernel(k, spot, 3) * p)
        x += _trapz(k, _kernel(k, spot, 4) * p)
    return v, w, x


def bkm_moments(
    strikes: FloatArray,
    prices: FloatArray,
    is_call: FloatArray,
    spot: float,
    rate: float,
    tau: float,
) -> dict[str, float]:
    """Model-free risk-neutral variance, skewness, and kurtosis.

    ``strikes`` / ``prices`` are the quoted OTM strike and
    (undiscounted-market) option price vectors, ``is_call`` flags the
    call legs (puts fill the rest), ``spot`` is the underlying level,
    ``rate`` the continuously compounded risk-free rate, and ``tau``
    the time to expiry in years. Returns the annualized risk-neutral
    variance and the standardized skewness/kurtosis of the
    BKM-implied distribution, plus the raw contract prices.
    """
    kk = np.asarray(strikes, dtype=np.float64)
    pp = np.asarray(prices, dtype=np.float64)
    cc = np.asarray(is_call, dtype=np.float64)
    if kk.shape != pp.shape or kk.shape != cc.shape or kk.ndim != 1:
        raise ValueError("strikes/prices/is_call must share a 1-D shape")
    if kk.size < 4:
        raise ValueError("need at least four options")
    if not np.all(np.isfinite(kk)) or not np.all(np.isfinite(pp)):
        raise ValueError("non-finite strikes or prices")
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be positive and finite")
    if not np.isfinite(rate) or not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("bad rate/tau")
    if np.any(kk <= 0.0) or np.any(pp < 0.0):
        raise ValueError("strikes must be positive, prices nonnegative")
    n_calls = int(np.sum(cc != 0.0))
    n_puts = kk.size - n_calls
    if n_calls == 0 or n_puts == 0:
        raise ValueError("need both call and put legs")
    if np.min(kk) >= spot or np.max(kk) <= spot:
        raise ValueError("strikes must bracket the spot")

    v, w, x = _contracts(kk, pp, cc, spot)
    ert = float(np.exp(rate * tau))
    mu = ert - 1.0 - 0.5 * ert * v - ert * w / 6.0 - ert * x / 24.0
    ev = ert * v - mu * mu
    if ev <= 0.0:
        raise ValueError("implied variance nonpositive (degenerate surface)")
    ew = ert * w
    ex = ert * x
    var_q = ev / tau
    skew_q = (ew - 3.0 * mu * ert * v + 2.0 * mu**3) / ev**1.5
    kurt_q = (ex - 4.0 * mu * ew + 6.0 * mu * mu * ert * v - 3.0 * mu**4) / (ev * ev)
    return {
        "var_q": var_q,
        "vol_q": float(np.sqrt(var_q)),
        "skew_q": skew_q,
        "kurt_q": kurt_q,
        "excess_kurt_q": kurt_q - 3.0,
        "contract_v": v,
        "contract_w": w,
        "contract_x": x,
        "mu": mu,
        "n_calls": float(n_calls),
        "n_puts": float(n_puts),
    }


def mixture_option_prices(
    strikes: FloatArray,
    spot: float,
    rate: float,
    tau: float,
    weights: FloatArray,
    means: FloatArray,
    sds: FloatArray,
) -> dict[str, FloatArray | float]:
    """Analytic OTM option prices under a lognormal-mixture RN density.

    ``weights``, ``means``, ``sds`` describe components
    ``ln S_T ~ w_i N(m_i, s_i^2)`` of the risk-neutral density. The
    component means are shifted so the mixture prices the forward at
    ``spot * e^{rate tau}``. Returns call and put prices (puts below
    the spot, calls at-or-above) on the requested strike grid plus the
    exact raw log-return moments of the mixture for bench comparison.
    """
    kk = np.asarray(strikes, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64)
    mm = np.asarray(means, dtype=np.float64)
    ss = np.asarray(sds, dtype=np.float64)
    if kk.ndim != 1 or ww.shape != mm.shape or mm.shape != ss.shape:
        raise ValueError("bad mixture shapes")
    if not (np.all(np.isfinite(kk)) and np.all(np.isfinite(ww))):
        raise ValueError("non-finite inputs")
    if np.any(kk <= 0.0) or np.any(ww <= 0.0) or np.any(ss <= 0.0):
        raise ValueError("strikes/weights/sds must be positive")
    if not np.all(np.isfinite(mm)) or not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("bad means/spot")
    if not np.isfinite(rate) or not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("bad rate/tau")
    ww = ww / np.sum(ww)
    comp_fwd = np.exp(mm + 0.5 * ss * ss)
    target = spot * np.exp(rate * tau)
    shift = np.log(target / np.sum(ww * comp_fwd))
    mm = mm + shift
    ln_k = np.log(kk)
    call = np.zeros_like(kk)
    put = np.zeros_like(kk)
    disc = np.exp(-rate * tau)
    for i in range(ww.size):
        d1 = (mm[i] - ln_k + ss[i] * ss[i]) / ss[i]
        d2 = d1 - ss[i]
        fwd_i = np.exp(mm[i] + 0.5 * ss[i] * ss[i])
        nd1 = 0.5 * (1.0 + _special.erf(d1 / np.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + _special.erf(d2 / np.sqrt(2.0)))
        call += ww[i] * disc * (fwd_i * nd1 - kk * nd2)
        put += ww[i] * disc * (kk * (1.0 - nd2) - fwd_i * (1.0 - nd1))
    alpha = mm - np.log(spot)
    s2 = ss * ss
    er = float(np.sum(ww * alpha))
    er2 = float(np.sum(ww * (alpha * alpha + s2)))
    er3 = float(np.sum(ww * (alpha**3 + 3.0 * alpha * s2)))
    er4 = float(np.sum(ww * (alpha**4 + 6.0 * alpha * alpha * s2 + 3.0 * s2 * s2)))
    return {
        "call_prices": call,
        "put_prices": put,
        "forward_implied": float(np.sum(ww * np.exp(mm + 0.5 * s2))),
        "mean_r": er,
        "raw_r2": er2,
        "raw_r3": er3,
        "raw_r4": er4,
    }


def synth_bkm(
    seed: int = 20261231 + 293,
    n: int = 240,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC surface: skewed lognormal mixture, analytic prices."""
    rng = np.random.default_rng(seed)
    spot = 100.0
    rate, tau = 0.03, 0.5
    weights = np.array([0.72, 0.28])
    means = np.array([np.log(spot) + 0.02, np.log(spot) - 0.10])
    sds = np.array([0.18, 0.45])
    strikes = np.linspace(0.2 * spot, 4.0 * spot, n)
    out = mixture_option_prices(strikes, spot, rate, tau, weights, means, sds)
    calls = np.asarray(out["call_prices"], dtype=np.float64)
    puts = np.asarray(out["put_prices"], dtype=np.float64)
    is_call = (strikes >= spot).astype(np.float64)
    prices = np.where(is_call.astype(bool), calls, puts)
    jitter = rng.normal(0.0, 1e-10, n)
    mu_r = float(out["mean_r"])
    var_t = float(out["raw_r2"]) - mu_r * mu_r
    skew_t = (float(out["raw_r3"]) - 3.0 * mu_r * float(out["raw_r2"]) + 2.0 * mu_r**3) / var_t**1.5
    kurt_t = (
        float(out["raw_r4"])
        - 4.0 * mu_r * float(out["raw_r3"])
        + 6.0 * mu_r * mu_r * float(out["raw_r2"])
        - 3.0 * mu_r**4
    ) / (var_t * var_t)
    return {
        "strikes": strikes,
        "prices": prices + jitter,
        "is_call": is_call,
        "spot": spot,
        "rate": rate,
        "tau": tau,
        "var_true": var_t,
        "skew_true": skew_t,
        "kurt_true": kurt_t,
    }


def bench_bkm_moments(seed: int = 20261231 + 293) -> dict[str, float]:
    """Wave-51 self-check: implied moments track the mixture truth."""
    d = synth_bkm(seed=seed)
    est = bkm_moments(
        np.asarray(d["strikes"]),
        np.asarray(d["prices"]),
        np.asarray(d["is_call"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    est2 = bkm_moments(
        np.asarray(d["strikes"]),
        np.asarray(d["prices"]),
        np.asarray(d["is_call"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    var_q = est["var_q"] * float(d["tau"])
    var_true = float(d["var_true"])
    skew_err = abs(est["skew_q"] - float(d["skew_true"]))
    kurt_err = abs(est["kurt_q"] - float(d["kurt_true"]))
    ok = (
        abs(var_q - var_true) / var_true < 0.10
        and skew_err < 0.12
        and kurt_err / float(d["kurt_true"]) < 0.20
        and est["skew_q"] < 0.0
        and est == est2
    )
    return {
        "var_q": var_q,
        "var_true": var_true,
        "skew_q": est["skew_q"],
        "skew_true": float(d["skew_true"]),
        "kurt_q": est["kurt_q"],
        "kurt_true": float(d["kurt_true"]),
        "score": float(ok),
    }
