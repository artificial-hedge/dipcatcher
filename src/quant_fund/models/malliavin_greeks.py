"""Malliavin calculus Greeks for Monte Carlo simulation.

Computes Delta, Vega and Gamma of path-dependent payoffs via Malliavin
integration-by-parts weights on the Wiener space, for general SDEs simulated
with the Euler–Maruyama scheme.  The tangent (variation) process Y_t is tracked
alongside the state, and the weights are assembled from the discrete path.

References
----------
1. Fournié, Lasry, Lebuchoux, Lions & Touzi (1999),
   "Applications of Malliavin calculus to Monte Carlo methods in finance,"
   *Finance & Stochastics*, 3, 391–412.
   — foundational paper: Delta = E[f·(1/T)∫(Y_t/σ(S_t))dW_t]  [Eq. 2.10].

2. Benhamou (2000), "Fast Greeks by simulation: forward automatic
   differentiation of the likelihood ratio method and the Malliavin derivative
   approach," *working paper*.
   — discrete Euler‑scheme weights; iterated‑weight Gamma [Algorithm 1].

3. Detemple, Garcia & Rindisbacher (2005), "A Monte Carlo Method for
   Optimal Portfolios," *J. Finance*, 58, 401–446.
   — Malliavin integration‑by‑parts in portfolio contexts.

Numerics
--------
- Euler–Maruyama with tangent process: O(Δt) discretisation bias.
  Recommended: 100–250 steps for 1yr option, MC 20k–50k paths.
- Zero volatility, degenerate paths, or non‑differentiable SDE params
  raise ValueError (fail‑closed).

SYNTHETIC research only — no live‑trading claims.  Seeded pseudo‑random.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

_ModelFunc = Callable[[float], float]


class ModelSpec(NamedTuple):
    """Drift, diffusion, and their first derivatives for an SDE.

    dS_t = drift(S_t) dt + diff(S_t) dW_t

    Attributes
    ----------
    drift : S → μ(S)
    diff : S → σ(S)
    drift_deriv : S → μ'(S)
    diff_deriv : S → σ'(S)
    diff_param_deriv : S → ∂σ(S;θ)/∂θ  (for Vega, θ is the vol scale)
    """

    drift: _ModelFunc
    diff: _ModelFunc
    drift_deriv: _ModelFunc
    diff_deriv: _ModelFunc
    diff_param_deriv: _ModelFunc


def _make_model_spec(
    name: str,
    *,
    r: float = 0.0,
    sigma: float = 1.0,
    kappa: float = 1.0,
    theta: float = 0.04,
) -> ModelSpec:
    """Return a ModelSpec for a named diffusion."""
    if name == "bachelier":
        return ModelSpec(
            drift=lambda s: 0.0,
            diff=lambda s: sigma,
            drift_deriv=lambda s: 0.0,
            diff_deriv=lambda s: 0.0,
            diff_param_deriv=lambda s: 1.0,
        )
    if name == "bs":
        return ModelSpec(
            drift=lambda s: r * s,
            diff=lambda s: sigma * s,
            drift_deriv=lambda s: r,
            diff_deriv=lambda s: sigma,
            diff_param_deriv=lambda s: s,  # ∂(σ⋅s)/∂σ = s
        )
    if name == "cir":
        return ModelSpec(
            drift=lambda s: kappa * (theta - s),
            diff=lambda s: sigma * math.sqrt(max(s, 0.0)),
            drift_deriv=lambda s: -kappa,
            diff_deriv=lambda s: 0.5 * sigma / math.sqrt(max(s, 1e-300)),
            diff_param_deriv=lambda s: math.sqrt(max(s, 0.0)),
        )
    raise ValueError(f"unknown model: {name!r}")


# ---------------------------------------------------------------------------
# Simulation output
# ---------------------------------------------------------------------------


class TangentPaths(NamedTuple):
    """Result of an Euler–Maruyama simulation with tangent processes.

    All arrays have shape (n_paths, n_steps + 1) for state/tangent variables
    and (n_paths, n_steps) for Brownian increments.
    """

    S: Array  # state path  S_0 ... S_N
    Y: Array  # Delta tangent  ∂S/∂S₀
    Y_vega: Array  # Vega tangent  ∂S/∂σ
    Z: Array  # second variation  ∂²S/∂S₀²
    dW: Array  # Brownian increments  ΔW_0 ... ΔW_{N-1}
    time_grid: Array  # t_0 ... t_N


# ---------------------------------------------------------------------------
# Core Euler–Maruyama with tangent processes
# ---------------------------------------------------------------------------


def simulate_euler_with_tangent(
    s0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    model: str = "bs",
    rng: np.random.Generator | int | None = None,
    **model_kw: float,
) -> TangentPaths:
    """Euler–Maruyama with Delta-, Vega- and Gamma‑tangent processes.

    Supported models: ``"bs"`` (GBM), ``"bachelier"``, ``"cir"``.

    Parameters
    ----------
    s0 : float
        Initial state. Must be > 0 for BS/CIR.
    T : float
        Time horizon (> 0).
    n_steps : int
        Number of Euler steps.
    n_paths : int
        Number of Monte Carlo paths.
    model : str
        Named model.
    rng : Generator | int | None
        Seeded RNG.
    **model_kw : float
        Model parameters: ``r``, ``sigma``, ``kappa``, ``theta``.

    Returns
    -------
    TangentPaths
    """
    if not (np.isfinite(s0) and s0 > 0):
        raise ValueError("s0 must be positive and finite")
    if not (np.isfinite(T) and T > 0):
        raise ValueError("T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    ms = _make_model_spec(model, **model_kw)

    # Safety: zero diffusion at s₀ ⇒ degenerate
    if ms.diff(s0) <= 0.0 or not np.isfinite(ms.diff(s0)):
        raise ValueError("diffusion coefficient must be positive at s0")

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    Y = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    Yv = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    Z = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    dW_arr = np.empty((n_paths, n_steps), dtype=np.float64)

    S[:, 0] = s0
    Y[:, 0] = 1.0
    Yv[:, 0] = 0.0
    Z[:, 0] = 0.0

    drift = ms.drift
    diff = ms.diff
    drift_deriv = ms.drift_deriv
    diff_deriv = ms.diff_deriv
    diff_param_deriv = ms.diff_param_deriv

    dW_all = rng.standard_normal((n_paths, n_steps), dtype=np.float64) * sqrt_dt
    dW_arr[:] = dW_all

    for k in range(n_steps):
        dW = dW_all[:, k]
        s_prev = S[:, k]
        y_prev = Y[:, k]
        yv_prev = Yv[:, k]
        z_prev = Z[:, k]

        # Coerce to scalar arrays safely
        dr = np.array([drift(float(s)) for s in s_prev], dtype=np.float64)
        df = np.array([diff(float(s)) for s in s_prev], dtype=np.float64)

        if not np.all(df > 0.0):
            raise ValueError(f"diffusion coefficient ≤ 0 at step {k}")

        dr_d = np.array([drift_deriv(float(s)) for s in s_prev], dtype=np.float64)
        df_d = np.array([diff_deriv(float(s)) for s in s_prev], dtype=np.float64)
        df_param = np.array([diff_param_deriv(float(s)) for s in s_prev], dtype=np.float64)

        # Euler updates
        S[:, k + 1] = np.maximum(s_prev + dr * dt + df * dW, 1e-300)

        # Tangent Y (Δ): dY = μ' · Y · dt + σ' · Y · dW
        Y[:, k + 1] = y_prev + dr_d * y_prev * dt + df_d * y_prev * dW

        # Vega tangent Yᵛ: dYᵛ = μ'·Yᵛ·dt + σ'·Yᵛ·dW + ∂σ/∂σ · dW
        Yv[:, k + 1] = yv_prev + dr_d * yv_prev * dt + df_d * yv_prev * dW + df_param * dW

        # Second variation Z: dZ = μ'·Z·dt + σ'·Z·dW  (for BS, Z ≡ 0)
        Z[:, k + 1] = z_prev + dr_d * z_prev * dt + df_d * z_prev * dW

    time_grid = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)

    return TangentPaths(
        S=S,
        Y=Y,
        Y_vega=Yv,
        Z=Z,
        dW=dW_arr,
        time_grid=time_grid,
    )


# ---------------------------------------------------------------------------
# Diffusion‑coefficient helper
# ---------------------------------------------------------------------------


def _sigma_grid(paths: TangentPaths, ms: ModelSpec) -> Array:
    """σ(S_k) for k = 0 … N−1  → shape (n_paths, n_steps)."""
    diff_fn = ms.diff
    S_int = paths.S[:, :-1]
    np_vec = np.vectorize(diff_fn, otypes=[np.float64])
    return np.asarray(np_vec(S_int), dtype=np.float64)


# ---------------------------------------------------------------------------
# Malliavin weights (discrete Euler scheme)
# ---------------------------------------------------------------------------


def delta_weight(
    paths: TangentPaths,
    model: str,
    **model_kw: float,
) -> Array:
    """Malliavin Delta weight.

    π_Δ = (1/T) Σ_{k=0}^{N-1}  Y_k · ΔW_k  /  σ(S_k)

    References
    ----------
    Fournié et al. (1999), Eq. (2.10).
    """
    ms = _make_model_spec(model, **model_kw)
    T = float(paths.time_grid[-1])
    dW = paths.dW
    sigma_vals = _sigma_grid(paths, ms)
    Y_int = paths.Y[:, :-1]
    terms = Y_int * dW / np.maximum(sigma_vals, 1e-300)
    return np.sum(terms, axis=1) / T


def vega_weight(
    paths: TangentPaths,
    model: str,
    **model_kw: float,
) -> Array:
    """Malliavin Vega weight (∂/∂σ).

    π_σ = (1/T) Σ_{k=0}^{N-1}  Yᵛ_k · ΔW_k  /  σ(S_k)

    where Yᵛ = ∂S/∂σ is the Vega tangent process.

    References
    ----------
    Fournié et al. (1999), Eq. (2.16) adapted for general σ parameter.
    """
    ms = _make_model_spec(model, **model_kw)
    T = float(paths.time_grid[-1])
    dW = paths.dW
    sigma_vals = _sigma_grid(paths, ms)
    Yv_int = paths.Y_vega[:, :-1]
    terms = Yv_int * dW / np.maximum(sigma_vals, 1e-300)
    return np.sum(terms, axis=1) / T


def gamma_weight(
    paths: TangentPaths,
    model: str,
    **model_kw: float,
) -> Array:
    """Malliavin Gamma weight via iterated formula.

    π_γ = π_Δ²  −  (1/T²) Σ_k  (Y_k / σ(S_k))² · Δt

    where Δt = T / n_steps.

    This is the Euler‑scheme form from Benhamou (2000), Algorithm 1.
    For BS with r=0 the correction reduces to the term −1/(σ² S₀² T).

    References
    ----------
    Benhamou (2000), Algorithm 1.
    """
    ms = _make_model_spec(model, **model_kw)
    T = float(paths.time_grid[-1])
    dW = paths.dW
    n_paths, n_steps = dW.shape
    dt = T / n_steps

    pi_d = delta_weight(paths, model, **model_kw)

    sigma_vals = _sigma_grid(paths, ms)
    Y_int = paths.Y[:, :-1]

    # Discretisation correction: (1/T²) Σ (Y_k/σ_k)² · Δt
    y_over_sig = Y_int / np.maximum(sigma_vals, 1e-300)
    sq_sum = np.sum(np.square(y_over_sig), axis=1)
    correction = np.asarray(sq_sum * dt / (T * T), dtype=np.float64)

    return np.asarray(pi_d * pi_d - correction, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public interface: malliavin_greeks
# ---------------------------------------------------------------------------


class GreeksResult(NamedTuple):
    delta: float
    vega: float
    gamma: float
    delta_se: float
    vega_se: float
    gamma_se: float
    price: float
    price_se: float


def _payoff(payoff_fn: Callable[[Array, float], Array], S_T: Array, strike: float) -> Array:
    """Call payoff_fn and validate shape."""
    result = np.asarray(payoff_fn(S_T, strike), dtype=np.float64)
    n = S_T.shape[0]
    if result.shape != (n,):
        raise ValueError(f"payoff_fn must return shape ({n},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("payoff_fn returned non-finite values")
    return result


def malliavin_greeks(
    s0: float,
    T: float,
    payoff_fn: Callable[[Array, float], Array],
    n_steps: int = 126,
    n_paths: int = 30000,
    model: str = "bs",
    rng: np.random.Generator | int | None = None,
    discount: float = 1.0,
    **model_kw: float,
) -> GreeksResult:
    """Estimate Greeks via Malliavin integration‑by‑parts.

    Simulates the Euler–Maruyama scheme with tangent processes for the
    chosen model, then weights the payoff f(S_T) with Malliavin weights
    to obtain Delta, Vega, and Gamma estimates together with standard
    errors.

    Parameters
    ----------
    s0 : float
        Initial state.
    T : float
        Time horizon.
    payoff_fn : Callable[[Array, float], Array]
        Payoff f(S_T, K) → (n_paths,) array.
    n_steps : int
        Euler steps (default 126).
    n_paths : int
        MC paths (default 30k).
    model : str
        ``"bs"``, ``"bachelier"``, or ``"cir"``.
    rng : Generator | int | None
    discount : float
        Discount factor DF = e^{-rT} to apply to the payoff.
    **model_kw : float
        Passed to model spec (``r``, ``sigma``, ``kappa``, ``theta``).
        Also accepts ``strike`` for the payoff.

    Returns
    -------
    GreeksResult
        Named tuple with (delta, vega, gamma, delta_se, vega_se,
        gamma_se, price, price_se).
    """
    # Extract strike from model_kw before passing to model
    strike = float(model_kw.pop("strike", s0))
    model_params = {k: v for k, v in model_kw.items() if k in {"r", "sigma", "kappa", "theta"}}

    # Seed
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    paths = simulate_euler_with_tangent(
        s0, T, n_steps, n_paths, model=model, rng=rng, **model_params
    )

    # Discounted payoff
    S_T = paths.S[:, -1]
    raw_payoffs = _payoff(payoff_fn, S_T, strike)
    payoffs_disc = raw_payoffs * discount

    # Price estimate
    price = float(np.mean(payoffs_disc))
    price_se = float(np.std(payoffs_disc, ddof=1) / math.sqrt(n_paths))

    # Delta
    pi_d = delta_weight(paths, model, **model_params)
    d_ests = payoffs_disc * pi_d
    delta = float(np.mean(d_ests))
    delta_se = float(np.std(d_ests, ddof=1) / math.sqrt(n_paths))

    # Vega
    pi_v = vega_weight(paths, model, **model_params)
    v_ests = payoffs_disc * pi_v
    vega = float(np.mean(v_ests))
    vega_se = float(np.std(v_ests, ddof=1) / math.sqrt(n_paths))

    # Gamma
    pi_g = gamma_weight(paths, model, **model_params)
    g_ests = payoffs_disc * pi_g
    gamma = float(np.mean(g_ests))
    gamma_se = float(np.std(g_ests, ddof=1) / math.sqrt(n_paths))

    return GreeksResult(
        delta=delta,
        vega=vega,
        gamma=gamma,
        delta_se=delta_se,
        vega_se=vega_se,
        gamma_se=gamma_se,
        price=price,
        price_se=price_se,
    )


# ---------------------------------------------------------------------------
# Analytic-reference helpers (correctness checking, SYNTHETIC only)
# ---------------------------------------------------------------------------


def bs_analytic_delta(
    S0: float, K: float, T: float, sigma: float, r: float = 0.0, call: bool = True
) -> float:
    """BSM analytic Delta."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(stats.norm.cdf(d1) if call else stats.norm.cdf(d1) - 1.0)


def bs_analytic_vega(S0: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """BSM analytic Vega (∂/∂σ)."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(S0 * stats.norm.pdf(d1) * math.sqrt(T))


def bs_analytic_gamma(S0: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """BSM analytic Gamma."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(stats.norm.pdf(d1) / (S0 * sigma * math.sqrt(T)))


def bs_analytic_price(
    S0: float, K: float, T: float, sigma: float, r: float = 0.0, call: bool = True
) -> float:
    """BSM analytic price."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return float(S0 * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2))
    return float(K * math.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1))


def bs_digital_delta(
    S0: float, K: float, T: float, sigma: float, r: float = 0.0, call: bool = True
) -> float:
    """Analytic Delta of a digital (binary) option.

    Payoff: 1_{S_T > K} (call) or 1_{S_T < K} (put).
    δ = φ(d₂) / (S₀ σ √T)  for call; negated for put.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d2 = (math.log(S0 / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    pdf_d2 = float(stats.norm.pdf(d2))
    factor = pdf_d2 / (S0 * sigma * math.sqrt(T))
    return factor if call else -factor
