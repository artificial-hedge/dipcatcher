"""Propagator (transient-impact) model estimation and inversion.

Bouchaud-Farmer-Lillo propagator dynamics: signed order flow
``eps_t * v_t`` moves the mid-price through a decaying memory kernel,

    R_l = sum_k G(l - k) * v_k * eps_k ,   G(l) ~ c * (l0 + l)^{-beta}

Unlike static square-root formulas, the kernel is *estimated* here:
least-squares deconvolution of a simulated/observed response against a
signed-flow series, with optional ridge and power-law projection; then
decomposed into transient vs permanent mass and inverted into an
expected-cost curve for execution schedules.

References
----------
- Bouchaud, Farmer & Lillo (2009). How markets slowly digest changes in
  supply and demand. In *Handbook of Financial Markets: Dynamics and
  Evolution*. arXiv:0809.0822.
- Taranto, Bormetti, Bouchaud, Lillo & Toth (2018). Linear response theory
  and finite-time propagators. *Physical Review Letters* 120.
  arXiv:1802.07015.
- Lillo, Mike & Farmer (2005). Theory for long memory in supply and demand.
  *Physical Review E* 71. arXiv:cond-mat/0502700.
- Gomes & Waelbroeck (2015). Is market impact a measure of the information
  value of trades? *Quantitative Finance* 15(5). arXiv:1410.8214.

Honesty
-------
All benches run on seeded SYNTHETIC flow generated in-module under a known
power-law propagator — recovery/calibration correctness only, never market
evidence.

Composition notes
-----------------
- ``execution/impact.py::propagator_kernel``/``propagator_price_path``
  provide the functional form and a price-path simulator; this module is the
  estimation/inversion counterpart (kernel recovery, decay diagnostics,
  schedule-cost inversion) — complementary, no shared code path.
- ``microstructure/book_metrics.py``/``kyle_ofi`` measure instantaneous
  impact and order-flow imbalance; the propagator view adds memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import lstsq
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_1d(name: str, a: FloatArray, min_len: int = 4) -> FloatArray:
    v = np.asarray(a, dtype=float).ravel()
    if v.size < min_len:
        raise ValueError(f"{name} needs >= {min_len} points")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} contains non-finite values")
    return v


def propagator_response(signed_flow: FloatArray, kernel: FloatArray) -> FloatArray:
    """Response series R_l = sum_k G(l-k) flow_k (causal convolution)."""
    flow = _check_1d("signed_flow", signed_flow)
    g = _check_1d("kernel", kernel, min_len=1)
    if (g < 0).any():
        raise ValueError("kernel must be non-negative")
    n = flow.size
    out = np.zeros(n)
    for lag in range(min(g.size, n)):
        out[lag:] += g[lag] * flow[: n - lag]
    return out


def power_law_kernel(lags: FloatArray, c: float, beta: float, l0: float = 1.0) -> FloatArray:
    """G(l) = c * (l0 + l)^{-beta}."""
    lags = _check_1d("lags", lags, min_len=1)
    if c <= 0 or beta < 0 or l0 <= 0:
        raise ValueError("need c>0, beta>=0, l0>0")
    return c * np.power(l0 + lags, -beta)


def fit_kernel_ls(
    signed_flow: FloatArray,
    response: FloatArray,
    n_lags: int,
    ridge: float = 1e-4,
) -> FloatArray:
    """Least-squares deconvolution: estimate G[0..n_lags) from (flow, R).

    Solves the triangular Toeplitz regression R_l = sum_{k<=n_lags} G_k
    flow_{l-k} + eps with ridge regularization, then clamps negatives to 0
    (kernel non-negativity is part of the model).
    """
    flow = _check_1d("signed_flow", signed_flow)
    resp = _check_1d("response", response)
    if flow.size != resp.size:
        raise ValueError("flow and response must have equal length")
    if not 1 <= n_lags <= flow.size // 4:
        raise ValueError("n_lags must be in [1, len/4]")
    n = flow.size
    # design matrix X[l, k] = flow[l - k] for k < n_lags (l >= k)
    x = np.zeros((n - n_lags, n_lags))
    for k in range(n_lags):
        x[:, k] = flow[n_lags - k : n - k]
    y = resp[n_lags:]
    gram = x.T @ x + ridge * np.eye(n_lags) * np.trace(x.T @ x) / n_lags
    rhs = x.T @ y
    g = np.linalg.solve(gram, rhs)
    return np.maximum(g, 0.0)


def fit_power_law(lags: FloatArray, kernel_hat: FloatArray) -> tuple[float, float]:
    """Log-log fit G(l) ~ c*(l0+l)^-beta via weighted least squares.

    Returns (c, beta). Lags with kernel_hat <= 0 are dropped (log domain).
    """
    lags = _check_1d("lags", lags, min_len=2)
    g = _check_1d("kernel_hat", kernel_hat, min_len=2)
    if lags.size != g.size:
        raise ValueError("lags and kernel_hat must have equal length")
    mask = g > 0
    if mask.sum() < 2:
        raise ValueError("too few positive kernel points")
    xlog = np.log(1.0 + lags[mask])
    ylog = np.log(g[mask])
    # slope = -beta
    design = np.column_stack([np.ones(mask.sum()), xlog])
    coef, *_ = lstsq(design, ylog, rcond=None)
    log_c, slope = float(coef[0]), float(coef[1])
    return float(np.exp(log_c)), -slope


@dataclass(frozen=True)
class ImpactDiagnostics:
    """Decay diagnostics for a fitted propagator kernel."""

    half_life: float  # first lag where G <= G(0)/2 (interpolated)
    beta: float  # fitted power-law exponent
    amplitude: float  # fitted c
    permanent_mass: float  # sum_k G(k) over the fitted horizon
    transient_share: float  # share of total mass decaying within half_life


def kernel_diagnostics(lags: FloatArray, kernel_hat: FloatArray) -> ImpactDiagnostics:
    """Half-life, power-law fit, and transient-vs-permanent split."""
    lags = _check_1d("lags", lags, min_len=3)
    g = _check_1d("kernel_hat", kernel_hat, min_len=3)
    if lags.size != g.size:
        raise ValueError("lags and kernel_hat must have equal length")
    if g[0] <= 0:
        raise ValueError("G(0) must be positive")
    half = g[0] / 2.0
    below = np.nonzero(g <= half)[0]
    if below.size == 0:
        half_life = float(lags[-1])
    else:
        i = int(below[0])
        if i == 0:
            half_life = float(lags[0])
        else:
            # linear interpolation between lags[i-1], lags[i]
            frac = (g[i - 1] - half) / max(g[i - 1] - g[i], 1e-12)
            half_life = float(lags[i - 1] + frac * (lags[i] - lags[i - 1]))
    c, beta = fit_power_law(lags, g)
    total = float(g.sum())
    within = float(g[lags <= half_life].sum())
    return ImpactDiagnostics(
        half_life=half_life,
        beta=beta,
        amplitude=c,
        permanent_mass=total,
        transient_share=within / total if total > 0 else 0.0,
    )


def expected_cost_curve(kernel: FloatArray, horizon: int) -> FloatArray:
    """Expected propagator cost of a unit metaorder spread uniformly over T steps.

    For each spreading length T in 1..horizon, the uniform schedule pays
    ``(1/T^2) sum_{i,j} G(|i-j|)`` — a quadratic form on the schedule.
    Entry 0 is the all-at-once cost G(0); the curve decays toward the
    permanent component as T grows (transient impact decays between tranches).
    """
    g = _check_1d("kernel", kernel, min_len=1)
    if horizon < 1:
        raise ValueError("horizon >= 1 required")
    costs = np.empty(horizon)
    for t in range(1, horizon + 1):
        # uniform unit metaorder over t steps: w_i = 1/t
        # cost = sum_{i,j} w_i w_j G(|i-j|) = (G(0) + 2 sum_l (t-l)/t^2 G(l))/1
        total = g[0]
        for lag in range(1, min(g.size, t)):
            total += 2.0 * g[lag] * (t - lag)
        costs[t - 1] = total / t**2
    return costs


def simulate_impacted_series(
    n: int,
    kernel: FloatArray,
    sigma_flow: float = 1.0,
    persistence: float = 0.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Generate (signed_flow, response) under a known kernel.

    Flow is AR(persistence) in sign space: eps_t = persistence * eps_{t-1}
    + noise, signed ±sigma_flow magnitudes.
    """
    if n < 16:
        raise ValueError("n >= 16 required")
    g = _check_1d("kernel", kernel, min_len=1)
    if sigma_flow <= 0 or not 0 <= persistence < 1:
        raise ValueError("sigma_flow>0 and 0<=persistence<1 required")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n)
    flow = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = persistence * prev + raw[t]
        flow[t] = np.sign(prev) * sigma_flow * (0.5 + abs(prev))
    return flow, propagator_response(flow, g)


def bench_propagator_impact(seed: int = 20260131) -> dict[str, float]:
    """SYNTHETIC bench for the propagator estimator. Correctness only."""
    out: dict[str, float] = {}
    n, n_lags = 40_000, 40
    lags = np.arange(n_lags, dtype=float)
    g_true = power_law_kernel(lags, c=1.0, beta=0.5)
    flow, resp = simulate_impacted_series(n, g_true, sigma_flow=1.0, persistence=0.5, seed=seed)
    g_hat = fit_kernel_ls(flow, resp, n_lags=n_lags, ridge=1e-3)
    # kernel recovery: L2 relative error on the fitted window
    err = float(np.linalg.norm(g_hat - g_true) / np.linalg.norm(g_true))
    out["synthetic_kernel_recovery_relerr"] = err
    diag = kernel_diagnostics(lags, g_hat)
    out["synthetic_halflife_err"] = abs(diag.half_life - _true_halflife(g_true, lags))
    out["synthetic_beta_err"] = abs(diag.beta - 0.5)
    out["synthetic_transient_share"] = diag.transient_share
    # held-out cost-model fit: simulate longer, predict response tail
    flow2, resp2 = simulate_impacted_series(
        4_000, g_true, sigma_flow=1.0, persistence=0.5, seed=seed + 1
    )
    pred = propagator_response(flow2, g_hat)
    ssr = float(np.sum((resp2 - pred) ** 2))
    sst = float(np.sum((resp2 - resp2.mean()) ** 2))
    out["synthetic_cost_model_r2"] = 1.0 - ssr / sst if sst > 0 else 0.0
    # cost curve sanity: all-at-once >= long uniform spread
    curve = expected_cost_curve(g_hat, horizon=20)
    out["synthetic_cost_instant"] = float(curve[0])
    out["synthetic_cost_uniform"] = float(curve[-1])
    out["synthetic_cost_monotone_frac"] = float(np.mean(np.diff(curve) <= 1e-9))
    # determinism
    f2, r2 = simulate_impacted_series(500, g_true, seed=seed)
    f3, r3 = simulate_impacted_series(500, g_true, seed=seed)
    out["synthetic_determinism"] = float(np.array_equal(f2, f3) and np.array_equal(r2, r3))
    return out


def _true_halflife(g: FloatArray, lags: FloatArray) -> float:
    half = g[0] / 2.0
    below = np.nonzero(g <= half)[0]
    return float(lags[below[0]]) if below.size else float(lags[-1])
