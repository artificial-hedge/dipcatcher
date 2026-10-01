"""Fukasawa (2026) first-order implied-variance representation. No Sharpe.

Implements the conditional quadratic-variation (QV) representation of
Black-Scholes total implied variance from

    Fukasawa, Masaaki (2026). "Yet another asymptotic formula for implied
    volatility." arXiv:2609.13961 [q-fin.PR]. Citation verified against
    https://arxiv.org/abs/2609.13961 (fetched 2026-09-30): title, author and
    abstract match the lane spec exactly; no spec-vs-paper deviation.

Setting (paper Sec. 2.2). S is a strictly positive continuous local
martingale under a pricing measure Q (zero rates/dividends), written
S/S0 = E(M) for a continuous local martingale M, and
A_t = <M>_t = <log S>_t is the realized total variance of the log price.
Theorem 2.5: under a QV-perturbation assumption (Assumption 2.1: the
normalized Z = A/E[A] deviates from 1 at a relative scale r -> 0 with a
uniformly integrable weighted perturbation Y = (Z - 1)/(r sqrt(Z))) and an
independent-Brownian smoothing component (Assumption 2.3: M = J + N,
<J, N> = 0, and conditionally on a sigma-field H that contains the paths of
J, C = <J>, D = <N>, N is a centered Gaussian martingale with clock D >=
c_underline A, c_underline in (0, 1]), for bounded standardized log-strikes
z = (k + v/2) / sqrt(v), v = E[A]:

    w_hat(k, T) = E[A | S_T = S0 e^k] + o(v r),          (paper eq. 18)
    sigma_hat(k, T)^2 = E[A | S_T = K] / T + o(v r / T), (paper eq. 19)

where w_hat is the put-implied total variance (paper eq. 5, put inversion)
and the conditional expectation is the continuous density version

    m(k) = E[A h(k)] / E[h(k)],                          (paper eq. 53)
    h(k) = D^{-1/2} phi((k - J_T + A/2) / sqrt(D)),      (paper eq. 51)

the conditional density of log(S_T/S_0) given H: log(S_T/S_0) =
J_T + N_T - A/2 and N_T | H ~ N(0, D_T). The same conditioning gives a
Rao-Blackwellized put price (paper Sec. 3.3, stopped-Black-Scholes
machinery): conditional on H, S_T = S0 L_T E(N)_T with
L_T = exp(J_T - C_T/2), so

    E[(K - S_T)_+] = E[ P_BS(S0 L_T, K, D_T) ],          (paper eqs. 2, 59)

with P_BS(s, K, q) the zero-rate put at total variance q, d_{1,2} =
log(s/K)/sqrt(q) +/- sqrt(q)/2. Both estimands are therefore smooth
Monte-Carlo means over the factor path only -- no kernel bandwidth and no
independent Brownian leg need be simulated for the representation itself.

Three asymptotic regimes (each separately exercisable below):

1. Small volatility-of-volatility (paper Sec. 2.5): dV/V = b dt + a c dW
   with a -> 0 at fixed T gives w_hat - m(k) = o(a). Implemented via
   ``simulate_lognormal_factor_sv``: V_t = v_* exp(b t + a s_v B_t
   - (a s_v)^2 t / 2) -- the bounded-coefficient case of paper eq. 38 --
   driven by the same factor Brownian B that carries S's correlated leg.
   The a = 0 limit is the deterministic-volatility (local-vol) case: there
   m(k) = v and w_hat = v *exactly*, a closed-form reduction asserted in
   tests.
2. Fast mean-reverting volatility (paper Sec. 2.6): the factor is the
   accelerated ergodic diffusion X^n_t = X_{n t}; with a bounded variance
   function f(X) = v_* exp(beta tanh X) and a constant correlation rho,
   w_hat - m(k) = o(n^{-1/2}). Implemented via ``simulate_ou_factor_sv``
   with accel = n: stationary-initialized OU factors X^n with rate n*lam
   and diffusion sqrt(n)*nu; v_n = T * v_bar exactly in stationarity.
3. Short maturity (paper Sec. 2.4, Cor. 2.7 + Example 2.10): fixed model,
   T -> 0, strikes at the fluctuation scale k = x sqrt(T). Implemented
   with the geometric-BM factor V_t = v_* exp(a s_v B_t - (a s_v)^2 t/2)
   (kernel H = 1/2, T^{-1/2} fluctuation sd sigma_U = a s_v -- the same
   class as Ex. 2.10's OU kernel with tunable amplitude), so
   sigma_hat^2 - E[A/T | S_T = K] = o(T^{1/2}) (paper eq. 23/33), and
   both sides share the closed-form leading term
   v_0 + sigma_U rho x sqrt(v_0) sqrt(T) / 2 (eqs. 26/33 vs 37), which
   the convergence table reports as ``slope_sig2``/``slope_m``/
   ``slope_theory``.

Honesty
-------
Everything here is SYNTHETIC Monte-Carlo correctness material on
model-generated paths with known law -- never market evidence, no
live-trading claim, and no Sharpe/Sortino/Calmar/P&L-style headline is
produced anywhere (there is no PnL in this module; the only prices are
European puts inside the representation check). ``conditional_qv`` is a
Monte-Carlo plug-in of paper eq. 53: it estimates E[A | S_T = K] at MC
precision on the supplied paths; it is a numerical diagnostic, not a proof
certificate of the theorem. ``*_convergence`` helpers report raw and
order-scaled absolute residuals |w_hat - m(k)| so an observer can check the
residual shrinks strictly faster than the theorem's first-order scale --
they surface non-convergence rather than hiding it. Fail-closed throughout:
non-positive spot/maturity/variance, degenerate smoothing clock (D <= 0),
inconsistent clocks (A != C + D, the non-martingale guard), non-finite
paths, empty inputs, and standardized strikes beyond the theorem's
boundedness guard all raise ``ValueError``.

Composition notes (import, do not reimplement)
---------------------------------------------
- ``quant_fund.models.options.implied_vol`` owns the exact Black-Scholes
  implied-volatility inversion (Brent bracketing, no-arbitrage bounds);
  ``implied_total_variance`` delegates to it and converts to total
  variance w = T sigma^2. ``iv_approx`` owns the closed-form ATM
  approximations (Brenner-Subrahmanyam, Corrado-Miller) -- deliberately
  not used for inversion here (the representation is defined by exact put
  inversion, paper eq. 5) but cross-checked against the inverted value in
  tests.
- ``quant_fund.models.local_stoch_vol.heston_mc`` owns the seeded,
  antithetic Philox-drawn correlated Heston path engine.
  ``heston_factor_paths`` adapts it to the paper's Example 2.4 factor
  structure: with corr(dW_S, dW_V) = rho the smoothing sigma-field is
  H = sigma(W_V), and the correlated martingale leg has the exact closed
  form J_T = rho/xi (v_T - v_0 - kappa theta T + kappa A_T) -- an Ito
  identity from dW_V = (dv - kappa(theta - v)dt)/(xi sqrt(v)) -- while
  D_T = (1 - rho^2) A_T. No reimplementation of Heston dynamics.
- ``quant_fund.models.realized.realized_variance`` owns the pathwise
  realized-QV estimator (sum of squared returns); tests cross-check it
  against the integrated-variance A carried on the path object.
- ``models/stoch_vol`` (Kalman QMLE fitter) and ``models/rough_vol``
  (scaling diagnostics / RFSV) are estimators over observed series, not
  vol-path generators under a known law; the conditional-QV experiments
  need Example 2.4 factor structure (vol adapted to the correlated factor
  Brownian), which the dedicated simulators below provide. ``volatility``
  owns forecast-side vol models (EWMA/GARCH/HAR), not pathwise QV.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models.options import implied_vol as _implied_vol

Array = NDArray[np.float64]

__all__ = [
    "SYNTHETIC_LABEL",
    "FactorSVPaths",
    "FukasawaCondQV",
    "FukasawaEstimate",
    "bs_put_total_variance",
    "conditional_log_density",
    "conditional_qv",
    "conditional_qv_arrays",
    "fast_mean_reverting_convergence",
    "heston_factor_paths",
    "implied_total_variance",
    "implied_total_variance_mc",
    "model_put_price",
    "short_maturity_convergence",
    "simulate_lognormal_factor_sv",
    "simulate_ou_factor_sv",
    "small_volvol_convergence",
    "standardized_strike",
    "bench_fukasawa_iv",
]

SYNTHETIC_LABEL = (
    "SYNTHETIC: model-generated price/vol paths under a known law; "
    "correctness material only, never market evidence"
)

#: Bound on the standardized strike |z| = |(k + v/2) / sqrt(v)|: Theorem 2.5
#: holds for bounded standardized log-strikes, so we fail closed beyond it.
DEFAULT_Z_MAX = 8.0

_MIN_PATHS = 4
_MIN_STEPS = 4


# ---------------------------------------------------------------------------
# Validation helpers (fail-closed)
# ---------------------------------------------------------------------------


def _check_positive(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return v


def _check_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    return seed


def _check_maturity(T: float) -> float:
    return _check_positive(T, "maturity")


def _check_grid(T: float, n_steps: int) -> float:
    t = _check_maturity(T)
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < _MIN_STEPS:
        raise ValueError(f"n_steps must be an int >= {_MIN_STEPS}")
    return t / n_steps


def _check_n_paths(n_paths: int) -> int:
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < _MIN_PATHS:
        raise ValueError(f"n_paths must be an int >= {_MIN_PATHS}")
    return n_paths


def _check_rho(rho: float) -> float:
    r = float(rho)
    if not np.isfinite(r) or abs(r) >= 1.0:
        raise ValueError("rho must be finite and in (-1, 1); |rho| = 1 kills the smoothing clock")
    return r


def _finite_1d(values: object, name: str) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def _check_strikes(k: Array | float, v: float, z_max: float) -> Array:
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    if kk.size == 0 or not bool(np.all(np.isfinite(kk))):
        raise ValueError("k must be non-empty and finite")
    zm = _check_positive(z_max, "z_max")
    z = (kk + v / 2.0) / math.sqrt(v)
    if bool(np.any(np.abs(z) > zm)):
        raise ValueError(
            f"standardized strike |z| exceeds the boundedness guard z_max={zm} "
            "(Theorem 2.5 covers bounded standardized strikes only)"
        )
    return kk


# ---------------------------------------------------------------------------
# Path carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSVPaths:
    """Terminal quantities of a Brownian-factor stochastic-vol model.

    Paper Example 2.4 / Assumption 2.3 decomposition of the log-price
    martingale M = J + N on each simulated path:

    - ``a_total[i]`` = A_T = <log S>_T = int V dt (integrated variance);
    - ``j_total[i]`` = J_T = int sqrt(V) rho dB (correlated martingale leg);
    - ``c_total[i]`` = C_T = <J>_T = int V rho^2 dt;
    - ``d_total[i]`` = D_T = <N>_T = int V (1 - rho^2) dt > 0, the
      independent-Brownian smoothing clock;
    - ``log_s[i]`` = log(S_T / S0) = J_T + N_T - A_T / 2;
    - ``log_s_path``: optional full (n_paths, n_steps + 1) log-S path for
      realized-QV cross-checks against ``models.realized``.

    Invariant A_T = C_T + D_T holds by construction; the consistency check
    in :func:`conditional_qv_arrays` treats its violation as non-martingale
    input and fails closed.
    """

    a_total: Array
    j_total: Array
    c_total: Array
    d_total: Array
    log_s: Array
    maturity: float
    log_s_path: Array | None = None
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        t = _check_maturity(self.maturity)
        object.__setattr__(self, "maturity", t)
        for name in ("a_total", "j_total", "c_total", "d_total", "log_s"):
            arr = _finite_1d(getattr(self, name), name)
            if arr.size < _MIN_PATHS:
                raise ValueError(f"{name} needs >= {_MIN_PATHS} paths")
            object.__setattr__(self, name, arr)
        n = self.a_total.size
        for name in ("j_total", "c_total", "d_total", "log_s"):
            if getattr(self, name).size != n:
                raise ValueError("path arrays must share one length")
        if bool(np.any(self.a_total <= 0.0)):
            raise ValueError("a_total must be > 0 (degenerate quadratic variation)")
        if bool(np.any(self.d_total <= 0.0)):
            raise ValueError(
                "d_total must be > 0 (no independent-Brownian smoothing component; "
                "Assumption 2.3 fails, h(k) is degenerate)"
            )
        if bool(np.any(self.c_total < 0.0)):
            raise ValueError("c_total must be >= 0")
        if self.log_s_path is not None:
            p = np.asarray(self.log_s_path, dtype=float)
            if p.ndim != 2 or p.shape[0] != n:
                raise ValueError("log_s_path must be (n_paths, n_steps + 1)")
            if not bool(np.all(np.isfinite(p))):
                raise ValueError("log_s_path must be finite")
            object.__setattr__(self, "log_s_path", p)

    @property
    def n_paths(self) -> int:
        return int(self.a_total.size)

    @property
    def v_mean(self) -> float:
        """v = E[A_T], the expected total realized variance."""
        return float(np.mean(self.a_total))


# ---------------------------------------------------------------------------
# Black-Scholes put in total-variance parametrization (paper eq. 2)
# ---------------------------------------------------------------------------


def bs_put_total_variance(s: Array | float, K: float, q: Array | float) -> Array:
    """P_BS(s, K, q) = K Phi(-d2) - s Phi(-d1), paper eq. (2), vectorized.

    ``q`` is the remaining total variance; at q = 0 the intrinsic value
    (K - s)+ is returned (the paper's convention). ``options.bs_price``
    owns the sigma-parametrized scalar version; this is the same formula
    reparametrized by q = sigma^2 T as the paper uses, vectorized so a
    whole path bundle can be priced under its conditional clock D_T.
    """
    kk = _check_positive(K, "K")
    ss = np.asarray(s, dtype=float)
    qq = np.asarray(q, dtype=float)
    if not bool(np.all(np.isfinite(ss))) or not bool(np.all(ss > 0.0)):
        raise ValueError("s must be finite and > 0")
    if not bool(np.all(np.isfinite(qq))) or bool(np.any(qq < 0.0)):
        raise ValueError("q must be finite and >= 0")
    ss, qq = np.broadcast_arrays(ss, qq)
    intrinsic = np.maximum(kk - ss, 0.0)
    q_pos = qq > 0.0
    out = np.array(intrinsic, dtype=float, copy=True)
    if bool(np.any(q_pos)):
        sq = np.sqrt(qq[q_pos])
        d1 = np.log(ss[q_pos] / kk) / sq + sq / 2.0
        d2 = d1 - sq
        out[q_pos] = kk * norm.cdf(-d2) - ss[q_pos] * norm.cdf(-d1)
    return np.asarray(out, dtype=float)


def model_put_price(
    paths: FactorSVPaths,
    k: float,
    s0: float = 1.0,
    *,
    method: Literal["rao_blackwell", "payoff"] = "rao_blackwell",
) -> tuple[float, float]:
    """Model put price E[(K - S_T)+] with K = s0 e^k; returns (price, MC se).

    ``rao_blackwell`` (default): mean of P_BS(s0 L_i, K, D_i) over paths,
    L_i = exp(J_i - C_i/2) -- the conditional-Gaussian mixture of paper
    Sec. 3.3; the independent leg N is never simulated for pricing.
    ``payoff``: naive mean of (K - s0 e^{log_s_i})+, retained as a noisy
    cross-check of the Rao-Blackwellized identity.
    """
    if not isinstance(paths, FactorSVPaths):
        raise TypeError("paths must be a FactorSVPaths")
    kk = float(k)
    if not np.isfinite(kk):
        raise ValueError("k must be finite")
    s = _check_positive(s0, "s0")
    strike = s * math.exp(kk)
    if method == "rao_blackwell":
        l_t = np.exp(paths.j_total - 0.5 * paths.c_total)
        prices = bs_put_total_variance(s * l_t, strike, paths.d_total)
    elif method == "payoff":
        s_t = s * np.exp(paths.log_s)
        prices = np.maximum(strike - s_t, 0.0)
    else:
        raise ValueError("method must be 'rao_blackwell' or 'payoff'")
    price = float(np.mean(prices))
    se = float(np.std(prices, ddof=1) / math.sqrt(paths.n_paths))
    return price, se


def implied_total_variance(put_price: float, s0: float, k: float, T: float) -> float:
    """w_hat(k, T): total implied variance of a put, paper eq. (5).

    Exact inversion through ``options.implied_vol`` (Brent bracketing over
    sigma), converted to total variance w = T sigma^2. Fail-closed when the
    price violates put no-arbitrage bounds or the bracket fails.
    """
    p = float(put_price)
    s = _check_positive(s0, "s0")
    t = _check_maturity(T)
    kk = float(k)
    if not np.isfinite(kk):
        raise ValueError("k must be finite")
    if not np.isfinite(p) or p < 0.0:
        raise ValueError("put_price must be finite and >= 0")
    sigma = _implied_vol(p, s, s * math.exp(kk), t, r=0.0, call=False)
    return float(t * sigma * sigma)


def implied_total_variance_mc(paths: FactorSVPaths, k: float, s0: float = 1.0) -> float:
    """Rao-Blackwellized model put price -> put-implied total variance."""
    price, _ = model_put_price(paths, k, s0, method="rao_blackwell")
    return implied_total_variance(price, s0, float(k), paths.maturity)


# ---------------------------------------------------------------------------
# Conditional-QV estimator (paper eqs. 51-53)
# ---------------------------------------------------------------------------


def _h_kernel(a: Array, j: Array, d: Array, k: Array) -> Array:
    """h_i(k) = D_i^{-1/2} phi((k - J_i + A_i/2) / sqrt(D_i)), eq. 51.

    Returns the (n_paths, n_k) kernel matrix; the continuous density version
    of E[A | S_T = s0 e^k] is E[A h(k)] / E[h(k)] (eq. 53).
    """
    sd = np.sqrt(d)[:, None]
    z = (k[None, :] - j[:, None] + 0.5 * a[:, None]) / sd
    return np.asarray(norm.pdf(z) / sd, dtype=float)


def conditional_qv_arrays(
    a_total: Array,
    j_total: Array,
    c_total: Array,
    d_total: Array,
    k: Array | float,
    *,
    z_max: float = DEFAULT_Z_MAX,
) -> FukasawaCondQV:
    """m(k) = E[A | S_T = s0 e^k] from raw path quantities (paper eq. 53).

    Works on any simulator's terminal quantities. Fail-closed non-martingale
    guard: the Ito decomposition requires A = <J> + <N> = C + D; a violation
    beyond float tolerance means the inputs are not the J/N decomposition of
    a continuous local martingale and we refuse rather than guess.
    """
    a = _finite_1d(a_total, "a_total")
    j = _finite_1d(j_total, "j_total")
    c = _finite_1d(c_total, "c_total")
    d = _finite_1d(d_total, "d_total")
    if not (a.size == j.size == c.size == d.size):
        raise ValueError("a_total, j_total, c_total, d_total must share one length")
    if a.size < _MIN_PATHS:
        raise ValueError(f"need >= {_MIN_PATHS} paths")
    if bool(np.any(a <= 0.0)):
        raise ValueError("a_total must be > 0")
    if bool(np.any(d <= 0.0)):
        raise ValueError("d_total must be > 0 (degenerate smoothing clock)")
    if bool(np.any(c < 0.0)):
        raise ValueError("c_total must be >= 0")
    if not bool(np.allclose(a, c + d, rtol=1e-8, atol=1e-12)):
        raise ValueError(
            "inconsistent clocks: a_total != c_total + d_total -- not the J/N "
            "decomposition of a continuous local martingale (fail-closed)"
        )
    v = float(np.mean(a))
    kk = _check_strikes(k, v, z_max)
    h = _h_kernel(a, j, d, kk)
    wsum = h.sum(axis=0)
    if bool(np.any(wsum <= 0.0)):
        raise ValueError(
            "strike outside the simulated terminal-density support "
            "(E[h(k)] underflows to 0); narrow the strike range"
        )
    m_num = (a[:, None] * h).sum(axis=0)
    density = wsum / a.size
    m = m_num / wsum
    # Delta-method MC standard error of the self-normalized ratio.
    resid = a[:, None] - m[None, :]
    # mean(resid * h) = 0 by construction, so Var(resid*h) ~= mean((resid*h)^2)
    m_se = np.sqrt(np.sum((resid * h) ** 2, axis=0)) / (a.size * density)
    return FukasawaCondQV(
        k=kk,
        m=np.asarray(m, dtype=float),
        density=np.asarray(density, dtype=float),
        m_se=np.asarray(m_se, dtype=float),
        v_mean=v,
        n_paths=int(a.size),
    )


@dataclass(frozen=True)
class FukasawaCondQV:
    """Conditional-QV estimate at a strike grid (paper eq. 53).

    ``m[j]`` = estimate of E[A | S_T = s0 e^{k[j]}]; ``density[j]`` =
    E[h(k_j)], the terminal log-return density at k; ``m_se`` is the
    delta-method MC standard error of the ratio estimator; ``v_mean`` =
    E[A] over the path bundle.
    """

    k: Array
    m: Array
    density: Array
    m_se: Array
    v_mean: float
    n_paths: int


def conditional_qv(
    paths: FactorSVPaths, k: Array | float, *, z_max: float = DEFAULT_Z_MAX
) -> FukasawaCondQV:
    """m(k) = E[A | S_T = s0 e^k] on a FactorSVPaths bundle (eq. 53)."""
    if not isinstance(paths, FactorSVPaths):
        raise TypeError("paths must be a FactorSVPaths")
    return conditional_qv_arrays(
        paths.a_total, paths.j_total, paths.c_total, paths.d_total, k, z_max=z_max
    )


def conditional_log_density(
    paths: FactorSVPaths, k: Array | float, *, z_max: float = DEFAULT_Z_MAX
) -> Array:
    """Terminal log-return density at k: E[h(k)] (denominator of eq. 53)."""
    return conditional_qv(paths, k, z_max=z_max).density


def standardized_strike(k: Array | float, v: float) -> Array:
    """z = (k + v/2) / sqrt(v), the bounded-strike coordinate of eq. (17)."""
    vv = _check_positive(v, "v")
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    if not bool(np.all(np.isfinite(kk))):
        raise ValueError("k must be finite")
    return np.asarray((kk + vv / 2.0) / math.sqrt(vv), dtype=float)


@dataclass(frozen=True)
class FukasawaEstimate:
    """Representation check at one strike: w_hat vs m(k) (paper eq. 18).

    ``residual = w_hat - m``; the theorem says it is o(v r) along the
    regime. ``residual_se`` combines the RB put-price MC error (mapped to
    total-variance units via the BS vega slope 2 w_hat / price-scale --
    approximated by the local derivative) and ``m_se`` in quadrature.
    """

    k: float
    z: float
    v_mean: float
    w_hat: float
    m: float
    m_se: float
    put_price: float
    put_price_se: float
    density: float
    residual: float
    residual_se: float


def fukasawa_residual(
    paths: FactorSVPaths, k: float, s0: float = 1.0, *, z_max: float = DEFAULT_Z_MAX
) -> FukasawaEstimate:
    """|w_hat(k, T) - m(k)| machinery at a single strike (eq. 18)."""
    if not isinstance(paths, FactorSVPaths):
        raise TypeError("paths must be a FactorSVPaths")
    s = _check_positive(s0, "s0")
    # bounded-strike guard before pricing: deep strikes invert to garbage
    _check_strikes(float(k), paths.v_mean, z_max)
    price, p_se = model_put_price(paths, float(k), s, method="rao_blackwell")
    w = implied_total_variance(price, s, float(k), paths.maturity)
    cq = conditional_qv(paths, np.asarray([float(k)]), z_max=z_max)
    m = float(cq.m[0])
    # d(sigma^2)/d(price) = 1 / vega_T; in total-variance units
    # d w_hat/dP = 1 / (dP/dq) with dP/dq = K phi(d2) / (2 sqrt(q)) (eq. 3).
    sq = math.sqrt(max(w, 1e-16))
    kk = float(k)
    d2 = -kk / sq - sq / 2.0
    vega_q = s * math.exp(kk) * float(norm.pdf(d2)) / (2.0 * sq)
    w_se = p_se / max(vega_q, 1e-16)
    r_se = math.sqrt(w_se * w_se + float(cq.m_se[0]) ** 2)
    z = float(standardized_strike(float(k), cq.v_mean)[0])
    return FukasawaEstimate(
        k=float(k),
        z=z,
        v_mean=cq.v_mean,
        w_hat=w,
        m=m,
        m_se=float(cq.m_se[0]),
        put_price=price,
        put_price_se=p_se,
        density=float(cq.density[0]),
        residual=w - m,
        residual_se=r_se,
    )


# ---------------------------------------------------------------------------
# Simulators: Brownian-factor SV models with known law
# ---------------------------------------------------------------------------


def _factor_step_contribs(
    v_path: Array,
    rho_v: Array | float,
    d_b: Array,
    d_b_perp: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Per-step increments (dA, dJ, dC, dD, dlogS) on a vol grid + Brownians.

    v_path: (n_paths, n_steps + 1) spot variance on a uniform grid.
    rho_v: broadcastable correlation (scalar, (n_steps,), or (P, S) states).
    d_b / d_b_perp: (n_paths, n_steps) increments of the factor and the
    independent smoothing Brownian. Prefix sums of these give the nested
    path restrictions used by the short-maturity table.
    """
    v = v_path[:, :-1]
    rho = np.broadcast_to(np.asarray(rho_v, dtype=float), v.shape)
    sq_v = np.sqrt(v)
    da = v * dt
    dj = sq_v * rho * d_b
    dc = v * rho * rho * dt
    dd = v * (1.0 - rho * rho) * dt
    dls = sq_v * (rho * d_b + np.sqrt(1.0 - rho * rho) * d_b_perp) - 0.5 * v * dt
    return da, dj, dc, dd, dls


def _accumulate_factor_sv(
    v_path: Array,
    rho_v: Array | float,
    d_b: Array,
    d_b_perp: Array,
    dt: float,
    *,
    return_log_path: bool,
) -> tuple[Array, Array, Array, Array, Array, Array | None]:
    """Integrate the J/N decomposition on a vol grid + Brownian increments.

    Left-point sums keep A = C + D exactly on the grid.
    """
    da, dj, dc, dd, dls = _factor_step_contribs(v_path, rho_v, d_b, d_b_perp, dt)
    log_s_path: Array | None = None
    if return_log_path:
        log_s_path = np.concatenate(
            [np.zeros((v_path.shape[0], 1)), np.cumsum(dls, axis=1)], axis=1
        )
    return (
        da.sum(axis=1),
        dj.sum(axis=1),
        dc.sum(axis=1),
        dd.sum(axis=1),
        dls.sum(axis=1),
        log_s_path,
    )


def _factor_sv_paths(
    v_path: Array,
    rho_v: Array | float,
    d_b: Array,
    d_bp: Array,
    dt: float,
    maturity: float,
    *,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """Bundle the left-point J/N/A sums on a vol grid into FactorSVPaths."""
    a, j, c, d, log_s, lp = _accumulate_factor_sv(
        v_path, rho_v, d_b, d_bp, dt, return_log_path=return_log_path
    )
    return FactorSVPaths(
        a_total=a,
        j_total=j,
        c_total=c,
        d_total=d,
        log_s=log_s,
        maturity=maturity,
        log_s_path=lp,
    )


def _det_vol_paths(
    v_level: float, d_b: Array, d_bp: Array, rho: float, maturity: float
) -> FactorSVPaths:
    """Deterministic-variance paths V_t = v_level on a shared bundle."""
    n_p, n_steps = d_b.shape
    dt = maturity / n_steps
    v_path = np.full((n_p, n_steps + 1), float(v_level))
    return _factor_sv_paths(v_path, rho, d_b, d_bp, dt, maturity)


def _lognormal_step_contribs(
    d_b: Array,
    d_bp: Array,
    *,
    v_star: float,
    volvol: float,
    a_scale: float,
    drift: float,
    rho: float,
    maturity: float,
) -> tuple[Array, Array, Array, Array, Array, float]:
    """Per-step increments of the geometric-BM (lognormal) factor model.

    Returns (dA, dJ, dC, dD, dlogS, dt) for prefix-sum nesting.
    """
    n_p, n_steps = d_b.shape
    dt = maturity / n_steps
    asv = a_scale * volvol
    b_path = np.concatenate([np.zeros((n_p, 1)), np.cumsum(d_b, axis=1)], axis=1)
    tgrid = np.arange(n_steps + 1) * dt
    v_path = v_star * np.exp(
        drift * tgrid[None, :] + asv * b_path - 0.5 * asv * asv * tgrid[None, :]
    )
    da, dj, dc, dd, dls = _factor_step_contribs(v_path, rho, d_b, d_bp, dt)
    return da, dj, dc, dd, dls, dt


def _lognormal_paths(
    d_b: Array,
    d_bp: Array,
    *,
    v_star: float,
    volvol: float,
    a_scale: float,
    drift: float,
    rho: float,
    maturity: float,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """Small-volvol model paths on a caller-supplied increment bundle."""
    da, dj, dc, dd, dls, dt = _lognormal_step_contribs(
        d_b,
        d_bp,
        v_star=v_star,
        volvol=volvol,
        a_scale=a_scale,
        drift=drift,
        rho=rho,
        maturity=maturity,
    )
    lp: Array | None = None
    if return_log_path:
        lp = np.concatenate([np.zeros((d_b.shape[0], 1)), np.cumsum(dls, axis=1)], axis=1)
    return FactorSVPaths(
        a_total=da.sum(axis=1),
        j_total=dj.sum(axis=1),
        c_total=dc.sum(axis=1),
        d_total=dd.sum(axis=1),
        log_s=dls.sum(axis=1),
        maturity=maturity,
        log_s_path=lp,
    )


def simulate_lognormal_factor_sv(
    *,
    v_star: float,
    volvol: float,
    a_scale: float,
    rho: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    drift: float = 0.0,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """Small-volatility-of-volatility model (paper Sec. 2.5, eq. 38).

    V_t = v_* exp(drift t + a s_v B_t - (a s_v)^2 t / 2): the bounded-
    coefficient geometric-BM case of dV/V = b dt + a c dW, driven by the
    SAME factor Brownian B that carries S's correlated leg,
    dS/S = sqrt(V) (rho dB + sqrt(1 - rho^2) dB_perp). ``a_scale`` = a is
    the vol-of-vol perturbation scale; a = 0 gives deterministic variance
    v_* e^{drift t}, where the representation is exact (m(k) = w_hat = v).
    """
    vs = _check_positive(v_star, "v_star")
    sv = _check_positive(volvol, "volvol")
    a_s = float(a_scale)
    if not np.isfinite(a_s) or a_s < 0.0:
        raise ValueError("a_scale must be finite and >= 0")
    r = _check_rho(rho)
    t = _check_maturity(maturity)
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")
    n_p = _check_n_paths(n_paths)
    dt = _check_grid(t, n_steps)
    rng = np.random.default_rng(_check_seed(seed))
    d_b = rng.standard_normal((n_p, n_steps)) * math.sqrt(dt)
    d_bp = rng.standard_normal((n_p, n_steps)) * math.sqrt(dt)
    return _lognormal_paths(
        d_b,
        d_bp,
        v_star=vs,
        volvol=sv,
        a_scale=a_s,
        drift=drift,
        rho=r,
        maturity=t,
        return_log_path=return_log_path,
    )


def simulate_ou_factor_sv(
    *,
    v_star: float,
    beta: float,
    lam: float,
    nu: float,
    accel: float,
    rho: float | Callable[[Array], Array],
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """OU-factor stochastic vol (paper Ex. 2.10 / Sec. 2.6 acceleration).

    Factor X solves dX = -accel*lam X dt + sqrt(accel) nu dW with
    stationary init X_0 ~ N(0, nu^2 / (2 lam)) (the invariant law is
    accel-free, so accel = n reproduces X_{nt} in distribution); exact OU
    transitions are used, stable at any accel. Vol V = f(X) =
    v_* exp(beta tanh X) (bounded above and away from 0, satisfying the
    moment conditions of paper Sec. 2.6 for every p). S follows
    dS/S = sqrt(f(X)) (rho(X) dW + sqrt(1 - rho(X)^2) dB_perp) -- W is the
    factor driver, so the J/N split is the paper's Example 2.4 with
    c_underline = 1 - max rho^2 > 0.

    ``accel = 1`` is the fixed-model case used by short-maturity
    asymptotics (Cor. 2.7, H = 1/2 for OU kernels, paper line after
    eq. 36); ``accel = n`` is the fast-mean-reverting sequence (Sec. 2.6,
    error o(n^{-1/2})).
    """
    vs = _check_positive(v_star, "v_star")
    bt = float(beta)
    if not np.isfinite(bt):
        raise ValueError("beta must be finite")
    lm = _check_positive(lam, "lam")
    nu_v = _check_positive(nu, "nu")
    acc = _check_positive(accel, "accel")
    if not callable(rho):
        _check_rho(float(rho))
    t = _check_maturity(maturity)
    n_p = _check_n_paths(n_paths)
    _check_grid(t, n_steps)
    rng = np.random.default_rng(_check_seed(seed))
    z_x0 = rng.standard_normal(n_p)
    z_b = rng.standard_normal((n_p, n_steps))
    z_p = rng.standard_normal((n_p, n_steps))
    return _ou_paths(
        z_x0,
        z_b,
        z_p,
        v_star=vs,
        beta=bt,
        lam=lm,
        nu=nu_v,
        accel=acc,
        rho=rho,
        maturity=t,
        return_log_path=return_log_path,
    )


def _ou_step_contribs(
    z_x0: Array,
    z_b: Array,
    z_p: Array,
    *,
    v_star: float,
    beta: float,
    lam: float,
    nu: float,
    accel: float,
    rho: float | Callable[[Array], Array],
    maturity: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Per-step increments (dA, dJ, dC, dD, dlogS) of the OU-factor model.

    ``z_x0`` seeds the stationary draw, ``z_b`` drives the factor Brownian
    and the J leg, ``z_p`` the independent smoothing Brownian. Exact OU
    transitions keep any stiffness ``accel * lam`` stable. Prefix sums of
    these contributions give nested path restrictions at any grid cut.
    """
    n_p, n_steps = z_b.shape
    if z_x0.shape != (n_p,) or z_p.shape != (n_p, n_steps):
        raise ValueError("normal shapes inconsistent")
    dt = maturity / n_steps
    sqdt = math.sqrt(dt)
    kappa = accel * lam
    decay = math.exp(-kappa * dt)
    # Exact OU transition driven by the SAME z_b stream that J integrates:
    # x <- decay x + sqrt(acc) nu w_eff z_b, with w_eff chosen so the step
    # innovation variance equals the exact OU value stat_var (1 - decay^2).
    # w_eff -> 1 as kappa dt -> 0, so it is the deterministic smoothing of
    # the increment by the kernel e^{-kappa (t-s)} inside the step.
    w_eff = math.sqrt((1.0 - decay * decay) / (2.0 * kappa * dt))
    stat_sd = nu / math.sqrt(2.0 * lam)
    x = z_x0 * stat_sd
    da = np.empty((n_p, n_steps))
    dj = np.empty((n_p, n_steps))
    dc = np.empty((n_p, n_steps))
    dd = np.empty((n_p, n_steps))
    dls = np.empty((n_p, n_steps))
    for i in range(n_steps):
        v_t = v_star * np.exp(beta * np.tanh(x))
        r_v = np.asarray(rho(x) if callable(rho) else np.full(n_p, float(rho)), dtype=float)
        if not bool(np.all(np.isfinite(r_v))) or bool(np.any(np.abs(r_v) >= 1.0)):
            raise ValueError("rho(x) must stay finite and |rho| < 1 on all paths")
        sq_v = np.sqrt(v_t)
        inc_b = z_b[:, i] * sqdt
        inc_p = z_p[:, i] * sqdt
        da[:, i] = v_t * dt
        dj[:, i] = sq_v * r_v * inc_b
        dc[:, i] = v_t * r_v * r_v * dt
        dd[:, i] = v_t * (1.0 - r_v * r_v) * dt
        dls[:, i] = sq_v * (r_v * inc_b + np.sqrt(1.0 - r_v * r_v) * inc_p) - 0.5 * v_t * dt
        x = decay * x + math.sqrt(accel) * nu * w_eff * inc_b
    return da, dj, dc, dd, dls


def _ou_paths(
    z_x0: Array,
    z_b: Array,
    z_p: Array,
    *,
    v_star: float,
    beta: float,
    lam: float,
    nu: float,
    accel: float,
    rho: float | Callable[[Array], Array],
    maturity: float,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """OU-factor model on caller-supplied standard normals (CRN shareable)."""
    da, dj, dc, dd, dls = _ou_step_contribs(
        z_x0,
        z_b,
        z_p,
        v_star=v_star,
        beta=beta,
        lam=lam,
        nu=nu,
        accel=accel,
        rho=rho,
        maturity=maturity,
    )
    lp: Array | None = None
    if return_log_path:
        lp = np.concatenate([np.zeros((z_b.shape[0], 1)), np.cumsum(dls, axis=1)], axis=1)
    return FactorSVPaths(
        a_total=da.sum(axis=1),
        j_total=dj.sum(axis=1),
        c_total=dc.sum(axis=1),
        d_total=dd.sum(axis=1),
        log_s=dls.sum(axis=1),
        maturity=maturity,
        log_s_path=lp,
    )


def _prefix_factor_paths(
    da: Array,
    dj: Array,
    dc: Array,
    dd: Array,
    dls: Array,
    steps: int,
    dt: float,
) -> FactorSVPaths:
    """Nested path restriction: prefix sums through step ``steps``."""
    if steps < 1 or steps > da.shape[1]:
        raise ValueError(f"steps must be in [1, {da.shape[1]}]")
    s = slice(0, steps)
    return FactorSVPaths(
        a_total=da[:, s].sum(axis=1),
        j_total=dj[:, s].sum(axis=1),
        c_total=dc[:, s].sum(axis=1),
        d_total=dd[:, s].sum(axis=1),
        log_s=dls[:, s].sum(axis=1),
        maturity=steps * dt,
    )


def ou_vol_stat_mean(v_star: float, beta: float, lam: float, nu: float) -> float:
    """v_bar = E_pi[f(X)] under the stationary law N(0, nu^2/(2 lam)).

    Gauss-Hermite quadrature on the invariant OU law; this is the exact
    n -> infinity limit of E[A^n]/T in Sec. 2.6 (eq. 45) and the vol level
    of the deterministic anchor model used for paired residuals.
    """
    vs = _check_positive(v_star, "v_star")
    bt = float(beta)
    lm = _check_positive(lam, "lam")
    nu_v = _check_positive(nu, "nu")
    xs, ws = np.polynomial.hermite_e.hermegauss(96)
    xstat = xs * nu_v / math.sqrt(2.0 * lm)
    return float(np.sum(ws * vs * np.exp(bt * np.tanh(xstat))) / math.sqrt(2.0 * math.pi))


def heston_factor_paths(
    *,
    spot: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    return_log_path: bool = False,
) -> FactorSVPaths:
    """Adapt ``local_stoch_vol.heston_mc`` to the paper's factor structure.

    heston_mc simulates dS = -V/2 S dt + sqrt(V) S dW_S (r = q = 0),
    dv = kappa(theta - v)dt + xi sqrt(v) dW_V, corr(dW_S, dW_V) = rho.
    Writing W_S = rho W_V + sqrt(1 - rho^2) B~ makes Assumption 2.3 exact
    with H = sigma(W_V): J = rho int sqrt(V) dW_V, N = sqrt(1 - rho^2)
    int sqrt(V) dB~, D_T = (1 - rho^2) A_T. Ito's formula on the vol SDE
    recovers J in closed form from the v path alone:

        J_T = rho/xi (v_T - v_0 - kappa theta T + kappa A_T),

    used verbatim here (no reimplementation of the dynamics). ``xi > 0``
    is required for the recovery; xi = 0 (deterministic vol) is covered
    exactly by ``simulate_lognormal_factor_sv(a_scale=0)`` instead.
    """
    from quant_fund.models.local_stoch_vol import heston_mc

    s0 = _check_positive(spot, "spot")
    kp = _check_positive(kappa, "kappa")
    th = _check_positive(theta, "theta")
    xv = _check_positive(
        xi, "xi (needed > 0 for the J_T recovery; use lognormal a=0 for deterministic vol)"
    )
    r = _check_rho(rho)
    v_init = _check_positive(v0, "v0")
    t = _check_maturity(maturity)
    n_p = _check_n_paths(n_paths)
    dt = _check_grid(t, n_steps)
    if n_p % 2 != 0:
        # heston_mc's antithetic engine needs an even path count
        n_p += 1
    times = np.linspace(0.0, t, n_steps + 1)
    out = heston_mc(
        spot=s0,
        r=0.0,
        q=0.0,
        kappa=kp,
        theta=th,
        xi=xv,
        rho=r,
        v0=v_init,
        times=times,
        n_paths=n_p,
        seed=_check_seed(seed),
    )
    s_paths = np.asarray(out["S"], dtype=float)
    v_paths = np.asarray(out["v"], dtype=float)
    if s_paths.ndim != 2 or v_paths.shape != s_paths.shape:
        raise ValueError("unexpected heston_mc output shape")
    a = np.sum(v_paths[:, :-1], axis=1) * dt
    j = (r / xv) * (v_paths[:, -1] - v_init - kp * th * t + kp * a)
    c = r * r * a
    d = (1.0 - r * r) * a
    log_s = np.log(np.maximum(s_paths[:, -1], 1e-300) / s0)
    lp = np.log(np.maximum(s_paths, 1e-300) / s0) if return_log_path else None
    return FactorSVPaths(
        a_total=a,
        j_total=j,
        c_total=c,
        d_total=d,
        log_s=log_s,
        maturity=t,
        log_s_path=lp,
    )


# ---------------------------------------------------------------------------
# Regime convergence tables
# ---------------------------------------------------------------------------


def small_volvol_convergence(
    a_scales: Array | list[float],
    *,
    k: float,
    v_star: float = 0.04,
    volvol: float = 1.0,
    drift: float = 0.0,
    rho: float = -0.7,
    maturity: float = 0.5,
    n_steps: int = 128,
    n_paths: int = 100_000,
    seed: int = 2609,
    s0: float = 1.0,
    z_max: float = DEFAULT_Z_MAX,
) -> dict[str, Array | float]:
    """o(a) check of Sec. 2.5 (eq. 41): residual/a_scale must shrink.

    Common-random-number design: ONE (dW, dW_perp) increment bundle drives
    every row plus an a = 0 anchor -- the deterministic-vol limit where the
    representation is exact. ``paired_residual = res(a) - res(0)`` cancels
    the shared Monte-Carlo noise so the theorem's remainder is visible at
    affordable path counts; ``scaled_residual`` = |paired| / a should
    decrease as a -> 0 (o(a) restated).
    """
    aa = _finite_1d(a_scales, "a_scales")
    if bool(np.any(aa <= 0.0)):
        raise ValueError("a_scales must be > 0 (a = 0 is the exact deterministic limit)")
    vs = _check_positive(v_star, "v_star")
    sv = _check_positive(volvol, "volvol")
    r = _check_rho(rho)
    t = _check_maturity(maturity)
    s = _check_positive(s0, "s0")
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")
    n_p = _check_n_paths(n_paths)
    dt = _check_grid(t, n_steps)
    rng = np.random.default_rng(_check_seed(seed))
    d_b = rng.standard_normal((n_p, n_steps)) * math.sqrt(dt)
    d_bp = rng.standard_normal((n_p, n_steps)) * math.sqrt(dt)

    def _paths(a_s: float) -> FactorSVPaths:
        return _lognormal_paths(
            d_b,
            d_bp,
            v_star=vs,
            volvol=sv,
            a_scale=a_s,
            drift=drift,
            rho=r,
            maturity=t,
        )

    anchor = fukasawa_residual(_paths(0.0), float(k), s, z_max=z_max)
    w = np.empty(aa.size)
    m = np.empty(aa.size)
    res = np.empty(aa.size)
    paired = np.empty(aa.size)
    se = np.empty(aa.size)
    for i, a_s in enumerate(aa):
        est = fukasawa_residual(_paths(float(a_s)), float(k), s, z_max=z_max)
        w[i], m[i], res[i] = est.w_hat, est.m, est.residual
        paired[i] = est.residual - anchor.residual
        se[i] = math.sqrt(est.residual_se**2 + anchor.residual_se**2)
    return {
        "a_scale": aa,
        "w_hat": w,
        "m": m,
        "residual": res,
        "paired_residual": paired,
        "abs_residual": np.abs(res),
        "scaled_residual": np.abs(paired) / aa,
        "residual_se": se,
        "k": np.full(aa.size, float(k)),
        "anchor_residual": float(anchor.residual),
        "anchor_se": float(anchor.residual_se),
    }


def fast_mean_reverting_convergence(
    accels: Array | list[float],
    *,
    k: float,
    v_star: float = 0.04,
    beta: float = 0.6,
    lam: float = 1.0,
    nu: float = 1.0,
    rho: float = -0.5,
    maturity: float = 0.5,
    n_steps: int = 1024,
    n_paths: int = 60_000,
    seed: int = 2609,
    s0: float = 1.0,
    z_max: float = DEFAULT_Z_MAX,
) -> dict[str, Array | float]:
    """o(n^{-1/2}) check of Sec. 2.6 (eq. 49): residual*sqrt(n) must shrink.

    Common-random-number design: one (z_x0, z_b, z_p) triple drives every
    row plus an exact n -> infinity anchor -- the deterministic-vol model
    at v_bar = E_pi[f(X)] (eq. 45; the representation is exact there).
    ``paired_residual = res(n) - res(inf)`` cancels shared MC noise;
    ``scaled_residual`` = |paired| * sqrt(n) should decrease as n grows
    (o(n^{-1/2}) restated). A fine grid is used by default so the fast
    factor is resolved (kappa*dt stays small).
    """
    nn = _finite_1d(accels, "accels")
    if bool(np.any(nn < 1.0)):
        raise ValueError("accels must be >= 1")
    vs = _check_positive(v_star, "v_star")
    bt = float(beta)
    if not np.isfinite(bt):
        raise ValueError("beta must be finite")
    lm = _check_positive(lam, "lam")
    nu_v = _check_positive(nu, "nu")
    r = _check_rho(rho)
    t = _check_maturity(maturity)
    s = _check_positive(s0, "s0")
    n_p = _check_n_paths(n_paths)
    _check_grid(t, n_steps)
    rng = np.random.default_rng(_check_seed(seed))
    z_x0 = rng.standard_normal(n_p)
    z_b = rng.standard_normal((n_p, n_steps))
    z_p = rng.standard_normal((n_p, n_steps))
    v_bar = ou_vol_stat_mean(vs, bt, lm, nu_v)
    anchor = fukasawa_residual(
        _det_vol_paths(v_bar, z_b * math.sqrt(t / n_steps), z_p * math.sqrt(t / n_steps), r, t),
        float(k),
        s,
        z_max=z_max,
    )
    w = np.empty(nn.size)
    m = np.empty(nn.size)
    res = np.empty(nn.size)
    paired = np.empty(nn.size)
    se = np.empty(nn.size)
    for i, acc in enumerate(nn):
        paths = _ou_paths(
            z_x0,
            z_b,
            z_p,
            v_star=vs,
            beta=bt,
            lam=lm,
            nu=nu_v,
            accel=float(acc),
            rho=r,
            maturity=t,
        )
        est = fukasawa_residual(paths, float(k), s, z_max=z_max)
        w[i], m[i], res[i] = est.w_hat, est.m, est.residual
        paired[i] = est.residual - anchor.residual
        se[i] = math.sqrt(est.residual_se**2 + anchor.residual_se**2)
    return {
        "accel": nn,
        "w_hat": w,
        "m": m,
        "residual": res,
        "paired_residual": paired,
        "abs_residual": np.abs(res),
        "scaled_residual": np.abs(paired) * np.sqrt(nn),
        "residual_se": se,
        "k": np.full(nn.size, float(k)),
        "v_bar": float(v_bar),
        "anchor_residual": float(anchor.residual),
        "anchor_se": float(anchor.residual_se),
    }


def short_maturity_convergence(
    maturities: Array | list[float],
    *,
    x: float = 1.0,
    v_star: float = 0.16,
    volvol: float = 1.0,
    a_scale: float = 1.0,
    drift: float = 0.0,
    rho: float = -0.7,
    n_steps: int = 256,
    n_paths: int = 100_000,
    seed: int = 2609,
    s0: float = 1.0,
    z_max: float = DEFAULT_Z_MAX,
) -> dict[str, Array | float]:
    """o(T^{1/2}) check of Cor. 2.7/Ex. 2.10 at k = x sqrt(T) (H = 1/2 factor).

    Uses the geometric-BM (lognormal) factor V_t = v_* exp(a s_v B_t -
    (a s_v)^2 t/2): its fluctuation kernel is H = 1/2 with T^{-1/2}
    asymptotic sd sigma_U = a s_v (U_t -> sigma_U Z), the same class as
    Ex. 2.10's OU kernel but with the fluctuation amplitude directly
    tunable — so the first-order error is large enough to resolve under
    nested common random numbers.

    Residuals are on the ANNUALIZED scale sigma_hat^2 - E[A/T | S_T = K]
    (eq. 33), where the remainder is o(T^{1/2}). ONE Brownian bundle is
    simulated on [0, max(T)] and every row is a nested prefix cut at a
    grid step; an anchor row at min(maturities)/4 approximates the T -> 0
    limit (the annualized residual vanishes there). ``paired_residual`` =
    res_ann(T) - res_ann(T_anchor) removes the shared MC noise.

    First-order diagnostics (paper eq. 26/33 vs eq. 37, H = 1/2): both
    sigma_hat^2 and E[A/T | S] approach v_0 = m(0) with leading
    coefficient sigma_U rho x sqrt(v0) / 2 * sqrt(T):

      slope_sig2 = (sigma_hat^2 - v0) / sqrt(T),
      slope_m    = (E[A/T|S] - v0) / sqrt(T),
      slope_theory = a s_v rho sqrt(v0) x / 2

    and the theorem is the statement that the two slopes agree to first
    order while |residual|/sqrt(T) stays small — the remainder is
    sub-leading relative to the shared O(sqrt(T)) term.
    """
    tt = _finite_1d(maturities, "maturities")
    if bool(np.any(tt <= 0.0)):
        raise ValueError("maturities must be > 0")
    xx = float(x)
    if not np.isfinite(xx):
        raise ValueError("x must be finite")
    vs = _check_positive(v_star, "v_star")
    sv = _check_positive(volvol, "volvol")
    a_s = float(a_scale)
    if not np.isfinite(a_s) or a_s < 0.0:
        raise ValueError("a_scale must be finite and >= 0")
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")
    r = _check_rho(rho)
    s = _check_positive(s0, "s0")
    n_p = _check_n_paths(n_paths)
    n_s = int(n_steps)
    t_max = float(np.max(tt))
    _check_grid(t_max, n_s)
    dt = t_max / n_s
    # Snap each maturity to the simulation grid; reported maturity is the
    # snapped value. Nested prefix sums of ONE path bundle give the rows,
    # so every pair of rows shares its noise nearly exactly.
    steps = np.rint(tt / dt).astype(int)
    if bool(np.any(steps < _MIN_STEPS)):
        raise ValueError(
            f"every maturity must span >= {_MIN_STEPS} grid steps "
            "(t_i / max(maturities) * n_steps >= 4)"
        )
    rng = np.random.default_rng(_check_seed(seed))
    d_b = rng.standard_normal((n_p, n_s)) * math.sqrt(dt)
    d_bp = rng.standard_normal((n_p, n_s)) * math.sqrt(dt)
    da, dj, dc, dd, dls, _ = _lognormal_step_contribs(
        d_b,
        d_bp,
        v_star=vs,
        volvol=sv,
        a_scale=a_s,
        drift=drift,
        rho=r,
        maturity=t_max,
    )
    # Anchor row approximates the T -> 0 limit (annualized residual -> 0):
    # a quarter of the smallest tabulated maturity on the same nested paths.
    t_anchor_steps = max(1, int(np.min(steps)) // 4)
    t_anchor = t_anchor_steps * dt
    anchor = fukasawa_residual(
        _prefix_factor_paths(da, dj, dc, dd, dls, t_anchor_steps, dt),
        xx * math.sqrt(t_anchor),
        s,
        z_max=z_max,
    )
    v0 = vs  # m(0) = lim E[V_t] at T -> 0 for the lognormal factor
    sigma_u = a_s * sv  # T^{-1/2} vol-fluctuation sd of the factor kernel
    eff_tt = steps * dt
    sq_t = np.sqrt(eff_tt)
    sig2 = np.empty(tt.size)
    m_ann = np.empty(tt.size)
    res = np.empty(tt.size)
    paired = np.empty(tt.size)
    se = np.empty(tt.size)
    kk = np.empty(tt.size)
    for i, (st, t) in enumerate(zip(steps, eff_tt, strict=True)):
        k = xx * math.sqrt(float(t))
        est = fukasawa_residual(
            _prefix_factor_paths(da, dj, dc, dd, dls, int(st), dt),
            k,
            s,
            z_max=z_max,
        )
        sig2[i] = est.w_hat / float(t)
        m_ann[i] = est.m / float(t)
        res[i] = est.residual / float(t)
        paired[i] = res[i] - anchor.residual / t_anchor
        se[i] = math.sqrt((est.residual_se / float(t)) ** 2 + (anchor.residual_se / t_anchor) ** 2)
        kk[i] = k
    return {
        "maturity": eff_tt,
        "sigma2_hat": sig2,
        "m_annualized": m_ann,
        "residual": res,
        "paired_residual": paired,
        "abs_residual": np.abs(res),
        "scaled_residual": np.abs(paired) / sq_t,
        "residual_se": se,
        "k": kk,
        "slope_sig2": (sig2 - v0) / sq_t,
        "slope_m": (m_ann - v0) / sq_t,
        "slope_theory": float(sigma_u * r * math.sqrt(v0) * xx / 2.0),
        "v0": float(v0),
        "anchor_maturity": float(t_anchor),
        "anchor_residual": float(anchor.residual / t_anchor),
        "anchor_se": float(anchor.residual_se / t_anchor),
    }


# ---------------------------------------------------------------------------
# Bench helper (flat dict, seeded, ~seconds)
# ---------------------------------------------------------------------------


def bench_fukasawa_iv(
    *,
    seed: int = 2609,
    n_paths: int = 30_000,
    n_steps: int = 192,
) -> dict[str, float]:
    """Seeded SYNTHETIC correctness bench (~seconds), flat metric dict.

    Exercises all three regimes at a moderate grid: small-volvol residual
    scaling across a halving, fast-mean-reverting sqrt(n)-scaled residuals,
    short-maturity sqrt(T)-scaled residuals, plus the exact deterministic-
    vol reduction and the eq. 53 normalization identity
    int m(k) E[h(k)] dk = E[A]. All keys are research diagnostics --
    proper-score-adjacent representation errors only; no PnL, no Sharpe.
    """
    tic = time.perf_counter()
    out: dict[str, float] = {}

    sv = small_volvol_convergence(
        [0.50, 0.25],
        k=0.10,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    sv_abs = np.asarray(sv["abs_residual"], dtype=float)
    sv_sc = np.asarray(sv["scaled_residual"], dtype=float)
    out["small_volvol_abs_resid_a50"] = float(sv_abs[0])
    out["small_volvol_abs_resid_a25"] = float(sv_abs[1])
    out["small_volvol_scaled_resid_a50"] = float(sv_sc[0])
    out["small_volvol_scaled_resid_a25"] = float(sv_sc[1])

    fm = fast_mean_reverting_convergence(
        [4.0, 16.0],
        k=0.10,
        n_paths=n_paths,
        n_steps=4 * n_steps,
        seed=seed + 101,
    )
    fm_sc = np.asarray(fm["scaled_residual"], dtype=float)
    out["fast_mr_scaled_resid_n4"] = float(fm_sc[0])
    out["fast_mr_scaled_resid_n16"] = float(fm_sc[1])

    sm = short_maturity_convergence(
        [0.08, 0.04, 0.02],
        x=0.4,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed + 202,
    )
    sm_t = np.asarray(sm["maturity"], dtype=float)
    sm_sc = np.asarray(sm["scaled_residual"], dtype=float)
    sm_sl = np.asarray(sm["slope_m"], dtype=float)
    out["short_mat_scaled_resid_t08"] = float(sm_sc[0])
    out["short_mat_scaled_resid_t02"] = float(sm_sc[-1])
    out["short_mat_slope_m_gap"] = float(abs(sm_sl[int(np.argmax(sm_t))] - sm["slope_theory"]))
    out["short_mat_slope_theory"] = float(sm["slope_theory"])

    det = simulate_lognormal_factor_sv(
        v_star=0.04,
        volvol=1.0,
        a_scale=0.0,
        rho=-0.5,
        maturity=0.5,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed + 303,
    )
    est = fukasawa_residual(det, 0.10)
    out["deterministic_vol_abs_resid"] = float(abs(est.residual))

    cq = conditional_qv(det, np.linspace(-0.4, 0.4, 81), z_max=DEFAULT_Z_MAX)
    dk = 0.8 / 80.0
    out["normalization_gap"] = float(abs(np.sum(cq.m * cq.density) * dk / cq.v_mean - 1.0))
    out["bench_n_paths"] = float(n_paths)
    out["bench_n_steps"] = float(n_steps)
    out["runtime_seconds"] = float(time.perf_counter() - tic)
    out["synthetic"] = 1.0
    return out
