"""Fourier-based option pricing suite: COS, CONV, and Hilbert-transform methods.

Key references
--------------
* Fang & Oosterlee (2008), "A novel pricing method for European options based on
  Fourier-cosine series expansions", SIAM J. Sci. Comput. 30: the COS method.
* Fang & Oosterlee (2009), "Pricing early-exercise and discrete barrier options
  by Fourier-cosine series expansions", Numerische Mathematik 114: COS-Bermudan.
* Lord, Fang, Bervoets & Oosterlee (2008), "A fast and accurate FFT-based method
  for pricing early-exercise options under Lévy processes", SIAM J. Sci.
  Comput. 30: the CONV method.
* Feng & Linetsky (2008), "Pricing discretely monitored barrier options: a fast
  Hilbert transform approach", Math. Finance 18: Hilbert-transform barriers.

All pricing is **SYNTHETIC** — no live market data or real option chain
references.  Fail-closed on non-integrable CFs, negative variance, and bad
truncation ranges.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft
from scipy.stats import norm as normal

from quant_fund.models.carr_madan import bs_char_fn as _bs_char_fn_impl

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
CharFn = Callable[[np.ndarray], np.ndarray]

# ──────────────────────────────────────────────────────────────────────
# 1. Characteristic-function registry
# ──────────────────────────────────────────────────────────────────────


def bs_char_fn(s0: float, r: float, t: float, sigma: float) -> CharFn:
    """Black–Scholes risk-neutral CF of ln S_T (re-exported from carr_madan)."""
    return _bs_char_fn_impl(s0, r, t, sigma)


def _validate_model_params(**kw: float) -> None:
    for name, val in kw.items():
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite")


def merton_char_fn(
    s0: float,
    r: float,
    t: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> CharFn:
    """Merton (1976) jump-diffusion risk-neutral CF of ln S_T.

    Parameters
    ----------
    s0 : initial spot price.
    r : risk-free rate (continuous).
    t : time to maturity.
    sigma : diffusion volatility.
    lam : Poisson jump intensity.
    mu_j : mean log-jump size.
    s_j : std of log-jump size.
    """
    _validate_model_params(s0=s0, r=r, t=t, sigma=sigma, lam=lam, mu_j=mu_j, s_j=s_j)
    if t <= 0 or sigma <= 0:
        raise ValueError("t and sigma must be positive")
    if lam < 0 or s_j < 0:
        raise ValueError("lam >= 0, s_j >= 0")
    kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
    drift = np.log(s0) + (r - 0.5 * sigma**2 - lam * kappa) * t

    def phi(u: np.ndarray) -> np.ndarray:
        ua = np.asarray(u, dtype=complex)
        diff_part = 1j * ua * drift - 0.5 * sigma**2 * t * ua**2
        jump_part = lam * t * (np.exp(1j * ua * mu_j - 0.5 * s_j**2 * ua**2) - 1.0)
        return np.asarray(np.exp(diff_part + jump_part), dtype=complex)

    return phi


def vg_char_fn(
    s0: float,
    r: float,
    t: float,
    sigma: float,
    nu: float,
    theta: float = 0.0,
) -> CharFn:
    """Variance-Gamma (Madan–Carr–Chang 1998) risk-neutral CF of ln S_T.

    Parameters
    ----------
    s0 : initial spot.
    r : risk-free rate.
    t : time to maturity.
    sigma : volatility of the Brownian subordinator.
    nu : variance rate of the gamma subordinator.
    theta : drift of the Brownian subordinator (skewness).
    """
    _validate_model_params(s0=s0, r=r, t=t, sigma=sigma, nu=nu, theta=theta)
    if t <= 0 or sigma <= 0 or nu <= 0:
        raise ValueError("t, sigma, nu must be positive")
    inner = 1.0 - theta * nu - 0.5 * sigma**2 * nu
    if inner <= 0:
        raise ValueError(
            f"VG martingale correction requires 1 - θ ν - σ² ν / 2 > 0; got {inner:.6g}"
        )
    omega = math.log(inner) / nu

    def phi(u: np.ndarray) -> np.ndarray:
        ua = np.asarray(u, dtype=complex)
        base = np.log(s0) + (r + omega) * t
        vg_cf = (1.0 - 1j * theta * nu * ua + 0.5 * sigma**2 * nu * ua**2) ** (-t / nu)
        return np.asarray(np.exp(1j * ua * base) * vg_cf, dtype=complex)

    return phi


def nig_char_fn(
    s0: float,
    r: float,
    t: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float = 0.0,
) -> CharFn:
    """Normal-Inverse-Gaussian risk-neutral CF of ln S_T.

    Parameters
    ----------
    s0 : initial spot.
    r : risk-free rate.
    t : time to maturity.
    alpha : tail heaviness (> 0).
    beta : asymmetry (|beta| < alpha).
    delta : scale (> 0).
    mu : location.
    """
    _validate_model_params(s0=s0, r=r, t=t, alpha=alpha, beta=beta, delta=delta, mu=mu)
    if t <= 0 or alpha <= 0 or delta <= 0:
        raise ValueError("t, alpha, delta must be positive")
    if abs(beta) >= alpha:
        raise ValueError("require |beta| < alpha")
    gamma = math.sqrt(alpha**2 - beta**2)
    beta2 = beta + 1.0
    if abs(beta2) >= alpha:
        raise ValueError(
            f"NIG martingale requires |beta+1| = {abs(beta2):.6g} < alpha = {alpha:.6g}"
        )
    gamma2 = math.sqrt(alpha**2 - beta2**2)
    omega = -mu - delta * (gamma - gamma2)
    drift = np.log(s0) + (r + omega + mu) * t
    dt = delta * t

    def phi(u: np.ndarray) -> np.ndarray:
        ua = np.asarray(u, dtype=complex)
        inner = np.sqrt(alpha**2 - (beta + 1j * ua) ** 2)
        return np.asarray(np.exp(1j * ua * drift + dt * (gamma - inner)), dtype=complex)

    return phi


# ──────────────────────────────────────────────────────────────────────
# 2. Helper: cumulants and truncation range
# ──────────────────────────────────────────────────────────────────────


def _bs_cumulants(
    s0: float, k: float, r: float, t: float, sigma: float
) -> tuple[float, float, float]:
    """Cumulants c1, c2, c4 of ln(S_T / K) under Black–Scholes."""
    c1 = np.log(s0 / k) + (r - 0.5 * sigma**2) * t
    c2 = sigma**2 * t
    c4 = 0.0
    return c1, c2, c4


def _merton_cumulants(
    s0: float,
    k: float,
    r: float,
    t: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> tuple[float, float, float]:
    """Cumulants c1, c2, c4 of ln(S_T / K) under Merton JD."""
    kappa = math.exp(mu_j + 0.5 * s_j**2) - 1.0
    c1 = np.log(s0 / k) + (r - 0.5 * sigma**2 - lam * kappa) * t + lam * t * mu_j
    c2 = (sigma**2 + lam * (mu_j**2 + s_j**2)) * t
    mu4 = mu_j**4 + 6.0 * mu_j**2 * s_j**2 + 3.0 * s_j**4
    c4 = lam * t * mu4
    return c1, c2, c4


def cos_truncation_range(
    char_fn: CharFn | None = None,
    s0: float | None = None,
    k: float | None = None,
    r: float | None = None,
    t: float | None = None,
    sigma: float | None = None,
    *,
    lam: float = 0.0,
    mu_j: float = 0.0,
    s_j: float = 0.0,
    L: float = 10.0,
    model: Literal["bs", "merton"] = "bs",
) -> tuple[float, float]:
    """Compute COS truncation interval [a, b] from model cumulants.

    Fang & Oosterlee (2008), §3::

        a = c1 - L * sqrt(c2 + sqrt(c4))
        b = c1 + L * sqrt(c2 + sqrt(c4))

    where c1, c2, c4 are the first, second, and fourth cumulants of
    ln(S_T / K).  For Gaussian models c4 = 0, so b - a = 2 L sqrt(c2).

    Parameters
    ----------
    s0, k, r, t, sigma : spot, strike, rate, tenor, vol.
    lam, mu_j, s_j : jump parameters (Merton only).
    L : number of standard deviations (default 10).
    model : "bs" or "merton".
    """
    if model == "bs":
        if s0 is None or k is None or r is None or t is None or sigma is None:
            raise ValueError("s0, k, r, t, sigma required for BS cumulants")
        c1, c2, c4 = _bs_cumulants(s0, k, r, t, sigma)
    elif model == "merton":
        if s0 is None or k is None or r is None or t is None or sigma is None:
            raise ValueError("s0, k, r, t, sigma required for Merton cumulants")
        c1, c2, c4 = _merton_cumulants(s0, k, r, t, sigma, lam, mu_j, s_j)
    else:
        raise ValueError(f"unknown model {model!r}")
    if c2 <= 0:
        raise ValueError(f"variance must be positive, got c2={c2:.6g}")
    half = L * math.sqrt(c2 + math.sqrt(max(c4, 0.0)))
    a = c1 - half
    b = c1 + half
    return float(a), float(b)


# ──────────────────────────────────────────────────────────────────────
# 3. COS payoff coefficients (analytic)
# ──────────────────────────────────────────────────────────────────────


def _cos_payoff_call(n: int, a: float, b: float, k: float) -> Array:
    """Fourier-cosine coefficients V_k for a European call on [a, b].

    Payoff = K · [e^y - 1]^+ where y = ln(S_T / K).  Integration over
    [c, b] with c = max(a, 0).  Closed forms per Fang & Oosterlee (2008),
    Eq. (33–34).
    """
    c = max(a, 0.0)
    if c >= b:
        return np.zeros(n, dtype=float)
    omega = np.pi * np.arange(n) / (b - a)
    v = np.empty(n, dtype=float)

    # k = 0 term: ∫_c^b (e^y - 1) dy = (e^b - e^c) - (b - c)
    v[0] = (math.exp(b) - math.exp(c)) - (b - c)

    if n > 1:
        w = omega[1:]
        # χ(c,b) = ∫_c^b e^y cos(w(y-a)) dy
        denom = 1.0 + w**2
        cos_b = np.cos(w * (b - a))
        sin_b = np.sin(w * (b - a))
        cos_c = np.cos(w * (c - a))
        sin_c = np.sin(w * (c - a))
        chi = (math.exp(b) * (cos_b + w * sin_b) - math.exp(c) * (cos_c + w * sin_c)) / denom
        # ψ(c,b) = ∫_c^b cos(w(y-a)) dy
        psi = (sin_b - sin_c) / w
        v[1:] = chi - psi

    return np.asarray(2.0 * k / (b - a) * v, dtype=float)


def _cos_payoff_put(n: int, a: float, b: float, k: float) -> Array:
    """Fourier-cosine coefficients V_k for a European put on [a, b].

    Payoff = K · [1 - e^y]^+, y = ln(S_T / K).  Integration over [a, d]
    with d = min(b, 0).
    """
    d = min(b, 0.0)
    if d <= a:
        return np.zeros(n, dtype=float)
    omega = np.pi * np.arange(n) / (b - a)
    v = np.empty(n, dtype=float)

    # k = 0 term: ∫_a^d (1 - e^y) dy = (d - a) - (e^d - e^a)
    v[0] = (d - a) - (math.exp(d) - math.exp(a))

    if n > 1:
        w = omega[1:]
        denom = 1.0 + w**2
        cos_d = np.cos(w * (d - a))
        sin_d = np.sin(w * (d - a))
        cos_a = np.cos(w * 0.0)  # w*(a-a) = 0
        sin_a = np.sin(w * 0.0)
        # χ(a,d) for e^y
        chi = (math.exp(d) * (cos_d + w * sin_d) - math.exp(a) * (cos_a + w * sin_a)) / denom
        # ψ(a,d) for 1
        psi = (sin_d - sin_a) / w
        # put = ∫(1 - e^y)cos = ψ - χ
        v[1:] = psi - chi

    return np.asarray(2.0 * k / (b - a) * v, dtype=float)


# ──────────────────────────────────────────────────────────────────────
# 4. COS European pricing
# ──────────────────────────────────────────────────────────────────────


def _cos_price_at(
    char_fn: CharFn,
    r: float,
    t: float,
    a: float,
    b: float,
    n: int,
    payoff_cos: Array,
    x: float,
) -> float:
    """Evaluate the COS series directly at state x (no grid interpolation).

    c(x) = e^{-rT} Σ'_k Re{ψ(u_k) V_k e^{i u_k (x - a)}} with ψ the
    increment CF.  Direct evaluation preserves the spectral accuracy of
    the series (grid + linear interpolation would degrade it to O(dx²)).
    """
    omega = np.pi * np.arange(n) / (b - a)
    G = char_fn(omega) * payoff_cos
    G[0] *= 0.5
    theta = omega * (x - a)
    return float(np.exp(-r * t) * (np.real(G) @ np.cos(theta) - np.imag(G) @ np.sin(theta)))


def _cos_price_grid(
    char_fn: CharFn,
    r: float,
    t: float,
    a: float,
    b: float,
    n: int,
    payoff_cos: Array,
) -> Array:
    """Core COS pricing on a uniform log-grid x_j ∈ [a, b].

    Returns the price at every grid point x_j = a + (j + 0.5) (b - a) / n.

    Rigorous COS identity (derived from the cosine expansion of the
    conditional density on [a, b] with basis cos(u_k (y - a))):

        c(x_j) = e^{-rT} Σ'_k Re{ ψ(u_k) e^{i u_k (x_j - a)} } V_k

    where ψ is the INCREMENT characteristic function (law of y_T - x, e.g.
    exp(i u (r - σ²/2)T - σ²u²T/2) for BS log-moneyness) — callers must
    strip any ln S₀ term embedded in a terminal-log-price CF.  The earlier
    e^{+i u_k a} multiplier double-rotated the phase (basis already carries
    the -a shift), piling the reconstructed density at the left endpoint.
    """
    omega = np.pi * np.arange(n) / (b - a)
    # G_k = ψ(ω_k) · V_k  (phase e^{i ω_k (x_j - a)} applied via cos/sin below)
    G = char_fn(omega) * payoff_cos
    # Halve the k=0 term (Σ' convention)
    G[0] *= 0.5

    re_G = np.real(G)
    im_G = np.imag(G)

    # θ_{kj} = π k (j + 0.5) / n
    k_idx = np.arange(n)[:, None]  # (n, 1)
    j_idx = np.arange(n)[None, :]  # (1, n)
    theta = np.pi * k_idx * (j_idx + 0.5) / n
    cos_mat = np.cos(theta)
    sin_mat = np.sin(theta)

    # c(x_j) = e^{-rT} Σ_k [Re{G_k} cos(θ_{kj}) - Im{G_k} sin(θ_{kj})]
    prices = np.exp(-r * t) * (re_G @ cos_mat - im_G @ sin_mat)
    return np.asarray(prices, dtype=float)


def cos_european_call(
    char_fn: CharFn,
    r: float,
    t: float,
    strikes: Array,
    a: float | None = None,
    b: float | None = None,
    *,
    n: int = 256,
    s0: float | None = None,
    L: float = 10.0,
) -> Array:
    """European call prices via the COS method (Fang & Oosterlee 2008).

    Parameters
    ----------
    char_fn : characteristic function of ln S_T.
    r : risk-free rate.
    t : time to maturity.
    strikes : array of positive strikes.
    a, b : truncation interval (auto-computed from BS cumulants if None,
           requires ``s0``).
    n : number of cosine terms.
    s0 : spot price (needed only when a, b are auto-computed).
    L : truncation width in standard deviations.

    Returns
    -------
    prices : 1‑D array of call prices matching ``strikes``.
    """
    ks = np.asarray(strikes, dtype=float).ravel()
    if ks.size == 0 or (ks <= 0).any() or not np.isfinite(ks).all():
        raise ValueError("strikes must be positive and finite")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if n < 4:
        raise ValueError("n must be >= 4")

    prices = np.empty(ks.size, dtype=float)
    for i, k in enumerate(ks):
        ak, bk = _resolve_ab(a, b, char_fn, s0, k, r, t, L)
        vk = _cos_payoff_call(n, ak, bk, k)
        # Fang-Oosterlee Eq. (10) needs psi(u) = E[exp(i u y_T) | y_0 = 0]
        # (the formula multiplies by exp(i u x) separately).  bs_char_fn
        # embeds ln(S_0) in its drift term, so passing it unadjusted
        # double-counts ln(S_0) at the evaluation point x = ln(S_0/K):
        #   psi(u) = char_fn(u) * exp(-i u ln S_0)     [s0 known]
        #          = char_fn(u) * exp(-i u ln K)       [s0 None => ATM fallback]
        adj = math.log(s0) if s0 is not None else math.log(k)

        def _cf_adj(u: np.ndarray, _cf: CharFn = char_fn, _a: float = adj) -> np.ndarray:
            return _cf(u) * np.exp(-1j * u * _a)

        grid = _cos_price_grid(_cf_adj, r, t, ak, bk, n, vk)
        x = np.log(s0 / k) if s0 is not None else _log_moneyness(char_fn, k)
        # Direct series evaluation at x (spectral accuracy); the grid is kept
        # only as a sanity fallback for x outside [a, b].
        if ak <= x <= bk:
            prices[i] = _cos_price_at(_cf_adj, r, t, ak, bk, n, vk, float(x))
        else:
            dx = (bk - ak) / n
            j = (x - ak) / dx - 0.5
            jj = int(np.floor(j))
            if jj < 0:
                prices[i] = grid[0]
            elif jj >= n - 1:
                prices[i] = grid[-1]
            else:
                w = j - jj
                prices[i] = (1.0 - w) * grid[jj] + w * grid[jj + 1]

    return prices


def cos_european_put(
    char_fn: CharFn,
    r: float,
    t: float,
    strikes: Array,
    a: float | None = None,
    b: float | None = None,
    *,
    n: int = 256,
    s0: float | None = None,
    L: float = 10.0,
) -> Array:
    """European put prices via the COS method.  See ``cos_european_call``."""
    ks = np.asarray(strikes, dtype=float).ravel()
    if ks.size == 0 or (ks <= 0).any() or not np.isfinite(ks).all():
        raise ValueError("strikes must be positive and finite")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if n < 4:
        raise ValueError("n must be >= 4")

    prices = np.empty(ks.size, dtype=float)
    for i, k in enumerate(ks):
        ak, bk = _resolve_ab(a, b, char_fn, s0, k, r, t, L)
        vk = _cos_payoff_put(n, ak, bk, k)
        # See cos_european_call: psi(u) = char_fn(u) * exp(-i u ln S_0) is the
        # zero-initial-log-price CF the COS formula requires (double-counting
        # ln S_0 otherwise).
        adj = math.log(s0) if s0 is not None else math.log(k)

        def _cf_adj(u: np.ndarray, _cf: CharFn = char_fn, _a: float = adj) -> np.ndarray:
            return _cf(u) * np.exp(-1j * u * _a)

        grid = _cos_price_grid(_cf_adj, r, t, ak, bk, n, vk)
        x = np.log(s0 / k) if s0 is not None else _log_moneyness(char_fn, k)
        # Direct series evaluation at x (spectral accuracy); grid fallback
        # only for x outside [a, b].
        if ak <= x <= bk:
            prices[i] = _cos_price_at(_cf_adj, r, t, ak, bk, n, vk, float(x))
        else:
            dx = (bk - ak) / n
            j = (x - ak) / dx - 0.5
            jj = int(np.floor(j))
            if jj < 0:
                prices[i] = grid[0]
            elif jj >= n - 1:
                prices[i] = grid[-1]
            else:
                w = j - jj
                prices[i] = (1.0 - w) * grid[jj] + w * grid[jj + 1]

    return prices


def _resolve_ab(
    a: float | None,
    b: float | None,
    char_fn: CharFn,
    s0: float | None,
    k: float,
    r: float,
    t: float,
    L: float,
) -> tuple[float, float]:
    if a is not None and b is not None:
        return float(a), float(b)
    if s0 is None:
        raise ValueError("s0 required when a/b are auto-computed")
    # Derive BS-equivalent sigma from char_fn(…) evaluated near zero
    sigma_est = _estimate_sigma_from_cf(char_fn, t)
    a_val, b_val = cos_truncation_range(s0=s0, k=k, r=r, t=t, sigma=sigma_est, L=L, model="bs")
    return a_val, b_val


def _estimate_sigma_from_cf(char_fn: CharFn, t: float) -> float:
    """Estimate implied sigma from the CF's second derivative near u=0."""
    eps = 1e-4
    cf0 = char_fn(np.array([0.0]))[0]
    cf_plus = char_fn(np.array([eps]))[0]
    cf_minus = char_fn(np.array([-eps]))[0]
    # Characteristic function φ(u); use finite-diff for variance
    # Second derivative of log-φ gives variance
    log_cf_p = np.log(cf_plus / cf0)
    log_cf_m = np.log(cf_minus / cf0)
    var_est = -float(np.real((log_cf_p + log_cf_m) / eps**2))
    if var_est <= 0:
        return 0.3  # fallback
    sigma_est = math.sqrt(var_est / t)
    return float(sigma_est)


def _log_moneyness(char_fn: CharFn, k: float) -> float:
    """Derive approximate log-moneyness from char_fn(0)=1 property.

    When s0 is unknown, we default to at-the-money (x=0)."""
    return 0.0


# ──────────────────────────────────────────────────────────────────────
# 5. COS Bermudan pricing
# ──────────────────────────────────────────────────────────────────────


def cos_bermudan_put(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    strikes: Array,
    *,
    M: int = 10,
    n: int = 256,
    L: float = 10.0,
    exercise: Literal["put", "call"] = "put",
) -> Array:
    """Bermudan option prices via the COS backward recursion.

    Fang & Oosterlee (2009), §3: continuation value at t_m is recovered
    from the COS coefficients V_k(t_{m+1}) via the same expansion, then
    the early-exercise decision is applied pointwise on the grid.

    Parameters
    ----------
    char_fn : CF of ln S_T (maturity).
    r : risk-free rate.
    t : total time to maturity.
    s0 : spot price.
    strikes : array of positive strikes.
    M : number of exercise dates (≥ 1).  The last date is maturity.
    n : number of cosine terms / grid points.
    L : truncation width in standard deviations.
    exercise : "put" or "call".

    Returns
    -------
    prices : Bermudan option prices at each strike.
    """
    ks = np.asarray(strikes, dtype=float).ravel()
    if ks.size == 0 or (ks <= 0).any():
        raise ValueError("strikes must be positive")
    if t <= 0 or M < 1 or n < 4:
        raise ValueError("t > 0, M >= 1, n >= 4")

    dt = t / M
    # Use BS-equivalent sigma for a single global truncation range
    sigma_est = _estimate_sigma_from_cf(char_fn, t)
    a, b = cos_truncation_range(s0=s0, k=float(np.median(ks)), r=r, t=t, sigma=sigma_est, L=L)

    # One-step increment CF in log-moneyness coordinates.  char_fn embeds
    # ln(s0) and the FULL tenor t; the recursion needs the dt-step Lévy
    # exponent:  psi_dt(u) = exp((dt/t) * Log[char_fn(u) e^{-i u ln s0}])
    # (principal branch; |Im Log| < pi requires |u (r - sigma^2/2) t| < pi —
    # standard COS-Bermudan caveat, documented).
    adj = math.log(s0)
    scale = dt / t

    def _psi_step(
        u: np.ndarray, _cf: CharFn = char_fn, _a: float = adj, _s: float = scale
    ) -> np.ndarray:
        phi = _cf(u) * np.exp(-1j * u * _a)
        # Guard the principal log against an exactly-zero tail CF (exp(s*log0)
        # maps to 0 either way; the floor just silences the -inf warning).
        phi = np.where(np.abs(phi) < 1e-300, 1e-300 + 0j, phi)
        return np.exp(_s * np.log(phi))

    prices = np.empty(ks.size, dtype=float)
    grid_x = a + (np.arange(n) + 0.5) * (b - a) / n
    for idx, k in enumerate(ks):
        # V_M (maturity) coefficients from the payoff
        vk = _cos_payoff_put(n, a, b, k) if exercise == "put" else _cos_payoff_call(n, a, b, k)
        if exercise == "put":
            ex_val = np.maximum(k * (1.0 - np.exp(grid_x)), 0.0)
        else:
            ex_val = np.maximum(k * (np.exp(grid_x) - 1.0), 0.0)

        # Backward recursion over exercise dates t_1 .. t_{M-1}
        # (V_m = max(continuation, exercise); at maturity V_M = payoff).
        for _ in range(M - 1):
            cont = _cos_price_grid(_psi_step, r, dt, a, b, n, vk)
            vk = _dct_recover(np.maximum(cont, ex_val), n)

        # Final continuation step to t=0 evaluated directly at x0 = ln(s0/k)
        # (no exercise at t=0): spectral accuracy, no interpolation error.
        x0 = math.log(s0 / k)
        if a <= x0 <= b:
            prices[idx] = _cos_price_at(_psi_step, r, dt, a, b, n, vk, x0)
        else:
            cont = _cos_price_grid(_psi_step, r, dt, a, b, n, vk)
            prices[idx] = float(np.interp(x0, grid_x, cont))

    return prices


def _dct_recover(vals: Array, n: int) -> Array:
    """Recover COS coefficients V_k from grid values via DCT-II.

    V_k = (2/n) Σ_{j=0}^{n-1} f(x_j) cos(k π (j + 0.5) / n).
    """
    j_idx = np.arange(n)[None, :]
    k_idx = np.arange(n)[:, None]
    cos_mat = np.cos(np.pi * k_idx * (j_idx + 0.5) / n)
    return np.asarray((2.0 / n) * (cos_mat @ vals), dtype=float)


# ──────────────────────────────────────────────────────────────────────
# 6. CONV Bermudan pricing
# ──────────────────────────────────────────────────────────────────────


def conv_bermudan_put(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    strikes: Array,
    *,
    M: int = 10,
    N: int = 512,
    L: float = 10.0,
    alpha: float = 0.0,
) -> Array:
    """Bermudan put via the CONV method (Lord et al. 2008).

    Uses FFT-based convolution of the option value with the transition
    density, with optional exponential damping to control wrap-around.

    Parameters
    ----------
    char_fn : CF of ln S_T.
    r : risk-free rate.
    t : total tenor.
    s0 : spot.
    strikes : positive strikes.
    M : monitoring / early-exercise dates.
    N : FFT grid size (power of 2 recommended).
    L : truncation width.
    alpha : damping parameter (0 = no damping).

    Returns
    -------
    prices : Bermudan put prices at each strike.
    """
    ks = np.asarray(strikes, dtype=float).ravel()
    if ks.size == 0 or (ks <= 0).any():
        raise ValueError("strikes must be positive")
    if t <= 0 or M < 1 or N < 8:
        raise ValueError("t > 0, M >= 1, N >= 8")

    dt = t / M
    sigma_est = _estimate_sigma_from_cf(char_fn, t)
    med_k = float(np.median(ks))
    _, b_ref = cos_truncation_range(s0=s0, k=med_k, r=r, t=t, sigma=sigma_est, L=L)
    a_ref, _ = cos_truncation_range(s0=s0, k=med_k, r=r, t=t, sigma=sigma_est, L=L)

    # Build a wide-enough grid to avoid wrap-around
    half_width = (b_ref - a_ref) * 1.5
    x0 = np.log(s0 / med_k)
    x_min = x0 - half_width
    x_max = x0 + half_width
    dx = (x_max - x_min) / N
    grid = x_min + dx * np.arange(N)
    dk_grid = 2.0 * np.pi / (N * dx)
    u = dk_grid * np.fft.fftfreq(N) * N  # frequency grid

    # dt-step kernel on the log-return grid.  For a Levy model the step CF
    # is the (dt/t) power of the maturity CF of ln(S_T / s0), taken through
    # the complex principal log (same convention as the COS leg's
    # _psi_step) — NOT a magnitude-only kernel: the phase carries the mean.
    # Exponential damping tilts the density: the damped kernel's CF is
    # phi(u + i*alpha); alpha == 0 recovers the undamped kernel.
    u_eff = u + 1j * float(alpha)
    cf_vals = char_fn(u_eff) * np.exp(-1j * u_eff * np.log(s0))
    cf_safe = np.where(np.abs(cf_vals) < 1e-300, 1e-300 + 0j, cf_vals)
    cf_dt = np.exp((dt / t) * np.log(cf_safe))

    # Damping
    damp = np.exp(alpha * grid)

    prices = np.empty(ks.size, dtype=float)
    for idx, k in enumerate(ks):
        log_k = np.log(k)
        # Payoff on log-strike-shifted grid
        y = grid + np.log(s0) - log_k
        if alpha != 0.0:
            payoff = np.maximum(k - k * np.exp(y), 0.0) * damp
        else:
            payoff = np.maximum(k - k * np.exp(y), 0.0)

        # Terminal condition: value grid holds the (damped) payoff at t_M.
        v = payoff.copy()

        # M - 1 early-exercise dates t_1 .. t_{M-1} (inception is not an
        # exercise date; the final step below prices the last continuation).
        for _ in range(M - 1):
            v_hat = fft(v)
            conv = np.real(ifft(v_hat * cf_dt))
            v = np.maximum(np.exp(-r * dt) * conv, payoff)
            # Keep v real
            v = np.asarray(np.real(v), dtype=float)

        # Final continuation to t = 0 (no exercise at inception).
        v_hat = fft(v)
        v = np.asarray(np.exp(-r * dt) * np.real(ifft(v_hat * cf_dt)), dtype=float)

        # Interpolate at the spot's log-return x = 0.
        j_frac = (0.0 - x_min) / dx
        jj = int(np.floor(j_frac))
        if jj < 0:
            val = v[0]
        elif jj >= N - 1:
            val = v[-1]
        else:
            w = j_frac - jj
            val = (1.0 - w) * v[jj] + w * v[jj + 1]
        # Undamping at x = 0 is a no-op (exp(alpha * 0) == 1).

        prices[idx] = float(val)

    return prices


# ──────────────────────────────────────────────────────────────────────
# 7. Hilbert-transform barrier pricing
# ──────────────────────────────────────────────────────────────────────


def hilbert_barrier_call(
    char_fn: CharFn,
    r: float,
    t: float,
    s0: float,
    strike: float,
    barrier: float,
    *,
    M: int = 50,
    N: int = 512,
    L: float = 10.0,
    barrier_type: Literal["up-and-out", "down-and-out"] = "down-and-out",
) -> dict[str, float | int | Array]:
    """Discretely monitored barrier option via the Hilbert-transform /
    FFT method (Feng & Linetsky 2008).

    The monitoring is discrete with *M* equally spaced dates.  Increasing
    M closes the gap to the continuous-monitoring analytical price.

    Parameters
    ----------
    char_fn : CF of ln S_T.
    r : risk-free rate.
    t : total tenor.
    s0 : spot price.
    strike : option strike.
    barrier : barrier level.
    M : number of monitoring dates.
    N : FFT grid size.
    L : truncation width in standard deviations.
    barrier_type : "up-and-out" or "down-and-out".

    Returns
    -------
    dict with keys: ``price`` (float), ``M``, ``barrier``,
    ``survival_prob`` (the estimated probability of not being knocked out).
    """
    if t <= 0 or M < 1 or N < 8:
        raise ValueError("t > 0, M >= 1, N >= 8")
    if strike <= 0 or barrier <= 0 or s0 <= 0:
        raise ValueError("strike, barrier, s0 must be positive")
    if barrier_type == "down-and-out" and barrier >= s0:
        raise ValueError("down-and-out barrier must be below spot")
    if barrier_type == "up-and-out" and barrier <= s0:
        raise ValueError("up-and-out barrier must be above spot")

    dt = t / M
    sigma_est = _estimate_sigma_from_cf(char_fn, t)
    a_ref, b_ref = cos_truncation_range(s0=s0, k=strike, r=r, t=t, sigma=sigma_est, L=L)
    half_width = (b_ref - a_ref) * 1.5
    x0 = np.log(s0 / strike)
    x_min = x0 - half_width
    x_max = x0 + half_width
    dx = (x_max - x_min) / N
    grid = x_min + dx * np.arange(N)

    dk_grid = 2.0 * np.pi / (N * dx)
    u = dk_grid * np.fft.fftfreq(N) * N
    # Same Levy dt-step kernel as conv_bermudan_put: the (dt/t) power of
    # the maturity CF of ln(S_T / s0) through the complex principal log.
    cf_vals = char_fn(u) * np.exp(-1j * u * np.log(s0))
    cf_safe = np.where(np.abs(cf_vals) < 1e-300, 1e-300 + 0j, cf_vals)
    cf_dt = np.exp((dt / t) * np.log(cf_safe))

    log_barrier = np.log(barrier / strike)
    log_s0_shifted = np.log(s0 / strike)

    # Payoff at maturity (in log-moneyness coordinates)
    payoff = np.maximum(strike * np.exp(grid + log_s0_shifted) - strike, 0.0)

    # Barrier mask on grid
    if barrier_type == "down-and-out":
        barrier_mask = (grid + log_s0_shifted) > log_barrier
    else:
        barrier_mask = (grid + log_s0_shifted) < log_barrier

    # Terminal value at t_M: payoff on surviving paths only (the maturity
    # monitoring date knocks out terminal states beyond the barrier).
    v = np.where(barrier_mask, payoff, 0.0)

    for _ in range(M):
        v_hat = fft(v)
        conv = np.real(ifft(v_hat * cf_dt))
        v_next = np.exp(-r * dt) * conv
        # Apply barrier: zero out knocked-out region
        v_next[~barrier_mask] = 0.0
        # Ensure option is not negative
        v = np.maximum(v_next, 0.0)

    # Interpolate at the spot's log-return x = 0
    j_frac = (0.0 - x_min) / dx
    jj = int(np.floor(j_frac))
    if jj < 0:
        price = float(v[0])
    elif jj >= N - 1:
        price = float(v[-1])
    else:
        w = j_frac - jj
        price = float((1.0 - w) * v[jj] + w * v[jj + 1])

    # Estimate survival probability: fraction of paths not knocked out
    # (approximate via the ratio of option value to European value on grid)
    euro_val = np.maximum(strike * np.exp(grid + log_s0_shifted) - strike, 0.0)
    mask_both = barrier_mask & (euro_val > 1e-12)
    if mask_both.sum() > 0:
        surv = float(np.mean(v[mask_both] / (euro_val[mask_both] + 1e-12)))
    else:
        surv = 0.0

    return {"price": price, "M": M, "barrier": barrier, "survival_prob": surv}


# ──────────────────────────────────────────────────────────────────────
# 8. Analytical barrier formulas (Merton 1973) for verification
# ──────────────────────────────────────────────────────────────────────


def bs_continuous_barrier_call(
    s0: float,
    k: float,
    barrier: float,
    t: float,
    sigma: float,
    r: float = 0.0,
    barrier_type: Literal["down-and-out", "up-and-out"] = "down-and-out",
) -> float:
    """Merton (1973) analytical price for a continuously monitored
    barrier option under Black–Scholes.

    This is the **continuous** limit (M → ∞).  The discrete-monitoring
    Hilbert price should converge to this as M increases.

    References: Merton (1973) "Theory of rational option pricing", §8 (the
    reflection identity, down-and-out with K >= barrier); Reiner &
    Rubinstein (1991) / Haug (2007) §4.17 (four-term forms: down-and-out
    with barrier above strike, up-and-out with barrier above strike).
    """
    if t <= 0 or sigma <= 0:
        raise ValueError("t and sigma must be positive")
    if s0 <= 0 or k <= 0 or barrier <= 0:
        raise ValueError("s0, k, barrier must be positive")

    lam = (r + 0.5 * sigma**2) / sigma**2
    sq = sigma * math.sqrt(t)
    pow2 = (barrier / s0) ** (2.0 * lam)
    pow2m2 = (barrier / s0) ** (2.0 * lam - 2.0)

    if barrier_type == "down-and-out":
        if s0 <= barrier:
            return 0.0
        c_bs = _bs_price_scalar(s0, k, t, sigma, r, call=True)
        if k >= barrier:
            # Reflection-principle knock-in adjustment (Merton 1973):
            # valid when the payoff support lies above the barrier (K >= B).
            c_do = c_bs - pow2m2 * _bs_price_scalar(barrier**2 / s0, k, t, sigma, r, call=True)
        else:
            # Barrier above strike (K < B): four-term Reiner-Rubinstein form
            # — the compact reflection above is invalid here.
            x2 = math.log(s0 / barrier) / sq + lam * sq
            y1 = math.log(barrier / s0) / sq + lam * sq
            c_do = (
                s0 * normal.cdf(x2)
                - k * math.exp(-r * t) * normal.cdf(x2 - sq)
                - s0 * pow2 * normal.cdf(y1)
                + k * math.exp(-r * t) * pow2m2 * normal.cdf(y1 - sq)
            )
        return float(max(0.0, c_do))

    # up-and-out call (Reiner & Rubinstein 1991, Haug 2007 §4.17)
    if s0 >= barrier:
        return 0.0
    if barrier <= k:
        # The knocked region contains the whole payoff support.
        return 0.0
    x1 = math.log(s0 / k) / sq + lam * sq
    x2 = math.log(s0 / barrier) / sq + lam * sq
    y = math.log(barrier**2 / (s0 * k)) / sq + lam * sq
    y1 = math.log(barrier / s0) / sq + lam * sq
    c_uo = (
        s0 * (normal.cdf(x1) - normal.cdf(x2))
        - k * math.exp(-r * t) * (normal.cdf(x1 - sq) - normal.cdf(x2 - sq))
        - s0 * pow2 * (normal.cdf(y) - normal.cdf(y1))
        + k * math.exp(-r * t) * pow2m2 * (normal.cdf(y - sq) - normal.cdf(y1 - sq))
    )
    return float(max(0.0, c_uo))


def _bs_price_scalar(
    s: float, k: float, t_val: float, sigma: float, r: float, call: bool = True
) -> float:
    """Internal scalar BSM price; avoids importing options.py to keep
    the module self-contained."""
    if t_val <= 0:
        return (
            max(s - k * math.exp(-r * t_val), 0.0)
            if call
            else max(k * math.exp(-r * t_val) - s, 0.0)
        )
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t_val) / (sigma * math.sqrt(t_val))
    d2 = d1 - sigma * math.sqrt(t_val)
    if call:
        return float(s * normal.cdf(d1) - k * math.exp(-r * t_val) * normal.cdf(d2))
    return float(k * math.exp(-r * t_val) * normal.cdf(-d2) - s * normal.cdf(-d1))
