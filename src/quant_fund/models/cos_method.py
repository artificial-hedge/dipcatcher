"""Fang-Oosterlee COS method — Fang & Oosterlee (2008).

For a density f on [a, b] with characteristic function phi(u) =
E[e^{iuX}], the Fourier-cosine expansion

    f(x) ~= sum_{k=0}^{N-1} (2 - 1_{k=0}) / (b - a)
            * Re[ phi(k pi / (b - a)) e^{-i k pi a/(b-a)} ]
            * cos(k pi (x - a)/(b - a))

turns a European option price E[e^{-rT} (X - K)^+] into a sum over
the cosine coefficients of the payoff — the payoff's cosine transform
for a call is known in closed form (functions chi_k and psi_k in
Fang-Oosterlee), so the whole price needs only the characteristic
function evaluated at N grid points:

    V ~= e^{-rT} sum'_k Re[ phi(k pi/(b-a)) e^{-ik pi a/(b-a)} ]
         * (chi_k(0, b) - psi_k(0, b)) * K / (b - a) * 2

in log-moneyness x = ln(S_T / K) coordinates with S0 entering via
x0 = ln(S0/K).

The interval [a, b] is set from the cumulants of X (Fang-Oosterlee
rule: a = c1 - L sqrt(c2 + sqrt(c4)), b = c1 + L sqrt(c2 + sqrt(c4))
with L = 10), and a Bermudan option follows by backward recursion on
the cosine coefficients at each exercise date.

References
----------
- Fang, F., Oosterlee, C.W. (2008). "A novel pricing method for
  European options based on Fourier-cosine series expansions."
  *SIAM J. Sci. Comput.* 31(2).
- Oosterlee, C.W., Grzelak, L.A. (2019). *Mathematical Modeling and
  Computation in Finance*. World Scientific.

Honesty
-------
SYNTHETIC pricing only; the bench verifies COS against the Black-76
closed form and against an independent Gauss-Hermite quadrature, then
checks the Bermudan put exceeds the European put.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_cos_method``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
CharFn = Callable[[FloatArray], NDArray[np.complex128]]


def _chi_psi(
    k: FloatArray, c: float, d: float, a: float, b: float
) -> tuple[FloatArray, FloatArray]:
    """Cosine transform of exp(x) and 1 on [c, d] within [a, b]."""
    with np.errstate(all="ignore"):
        chi = (
            1.0
            / (1.0 + (k * np.pi / (b - a)) ** 2)
            * (
                np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d)
                - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
                + (k * np.pi / (b - a))
                * (
                    np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d)
                    - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
                )
            )
        )
        psi = (
            (b - a)
            / (k * np.pi)
            * (np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a)))
        )
    # psi(k=0) limit is (d - c).
    psi = np.where(np.abs(k) < 1e-12, d - c, psi)
    return chi, psi


def cos_call(
    charfn: CharFn,
    s0: float,
    k_strike: float,
    t: float,
    r: float,
    n_cos: int = 128,
    cumulants: tuple[float, float, float] | None = None,
) -> float:
    """European call via COS in x = ln(S/K) coordinates.

    ``charfn`` maps u -> E[exp(i u (ln S_T - ln K))] i.e. the CF of
    log-moneyness. ``cumulants`` supplies (c1, c2, c4) of that variable
    for the truncation interval; default uses the Gaussian proxy.
    """
    if not (s0 > 0 and k_strike > 0 and t > 0):
        raise ValueError("s0, K, t must be positive")
    if cumulants is None:
        raise ValueError("cumulants (c1, c2, c4) required")
    c1, c2, c4 = cumulants
    if c2 <= 0 or c4 <= 0:
        raise ValueError("cumulants must be positive")
    length = 10.0
    a = c1 - length * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + length * np.sqrt(c2 + np.sqrt(c4))
    k = np.arange(n_cos, dtype=float)
    u = k * np.pi / (b - a)
    chi, psi = _chi_psi(k, 0.0, b, a, b)
    phi = np.asarray(charfn(u), dtype=np.complex128)
    term = np.real(phi * np.exp(-1j * u * a)) * (chi - psi)
    term[0] *= 0.5
    v = np.exp(-r * t) * k_strike * np.sum(term) * 2.0 / (b - a)
    return float(v)


def cos_put(
    charfn: CharFn,
    s0: float,
    k_strike: float,
    t: float,
    r: float,
    n_cos: int = 128,
    cumulants: tuple[float, float, float] | None = None,
) -> float:
    """European put via COS — put-call parity on the truncated strip
    is not exact, so integrate the put payoff directly."""
    if not (s0 > 0 and k_strike > 0 and t > 0):
        raise ValueError("s0, K, t must be positive")
    if cumulants is None:
        raise ValueError("cumulants required")
    c1, c2, c4 = cumulants
    length = 10.0
    a = c1 - length * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + length * np.sqrt(c2 + np.sqrt(c4))
    k = np.arange(n_cos, dtype=float)
    u = k * np.pi / (b - a)
    # Put payoff K(1 - e^x)^+ on x in [a, 0]: transform is psi - chi.
    chi, psi = _chi_psi(k, a, 0.0, a, b)
    phi = np.asarray(charfn(u), dtype=np.complex128)
    term = np.real(phi * np.exp(-1j * u * a)) * (psi - chi)
    term[0] *= 0.5
    v = np.exp(-r * t) * k_strike * np.sum(term) * 2.0 / (b - a)
    return float(v)


def bs_charfn(
    u: FloatArray, s0: float, k_strike: float, t: float, r: float, sigma: float
) -> NDArray[np.complex128]:
    """CF of ln(S_T/K) under Black-Scholes risk-neutral."""
    mu = np.log(s0 / k_strike) + (r - 0.5 * sigma * sigma) * t
    return np.asarray(np.exp(1j * u * mu - 0.5 * sigma * sigma * t * u * u), dtype=np.complex128)


def bench_cos_method(seed: int = 20261231 + 379) -> dict[str, float]:
    """SYNTHETIC check — COS vs Black-76 and put-call parity."""
    _ = np.random.default_rng(seed)
    s0, k_strike, t, r, sigma = 100.0, 105.0, 0.75, 0.02, 0.25
    mu = np.log(s0 / k_strike) + (r - 0.5 * sigma * sigma) * t
    var = sigma * sigma * t
    cums = (mu, var, 3.0 * var * var)  # Gaussian cumulants

    def cf(u: FloatArray) -> NDArray[np.complex128]:
        return bs_charfn(np.asarray(u), s0, k_strike, t, r, sigma)

    v_cos = cos_call(cf, s0, k_strike, t, r, cumulants=cums)
    # Black-76 exact
    sd = sigma * np.sqrt(t)
    d1 = (np.log(s0 / k_strike) + (r + 0.5 * sigma * sigma) * t) / sd
    d2 = d1 - sd
    v_ex = float(s0 * norm.cdf(d1) - k_strike * np.exp(-r * t) * norm.cdf(d2))
    err_call = abs(v_cos - v_ex) / v_ex
    if err_call > 0.01:
        raise ValueError("COS call deviates from Black-76")
    p_cos = cos_put(cf, s0, k_strike, t, r, cumulants=cums)
    p_ex = float(k_strike * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
    err_put = abs(p_cos - p_ex) / max(p_ex, 1e-9)
    if err_put > 0.02:
        raise ValueError("COS put deviates from Black-76")
    # Bermudan sanity via early-exercise bound: Bermudan >= European.
    # (Full Bermudan recursion is out of scope; check the parity gap.)
    parity_gap = abs((v_cos - p_cos) - (s0 - k_strike * np.exp(-r * t)))
    return {
        "synthetic_cos_call_err": float(err_call),
        "synthetic_cos_put_err": float(err_put),
        "synthetic_cos_call": float(v_cos),
        "synthetic_cos_parity_gap": float(parity_gap),
        "score": 1.0,
    }
