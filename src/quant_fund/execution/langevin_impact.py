"""Generalized-Langevin latent-liquidity market impact (Itkin 2026). No Sharpe.

Implements the latent-liquidity market-impact model of

    Itkin, A. (2026). "A Generalized Langevin Model of Latent Liquidity and
    Concave Price Impact." arXiv:2609.37872 [q-fin.TR] (29 Sep 2026).
    Citation verified against https://arxiv.org/abs/2609.37872 and the full
    HTML text (fetched 2026-09-30); the spec's title/author/abstract match.

Model sketch (equation numbers refer to the paper). Order flow at signed rate
``q_t`` moves the log-price against constant displayed depth ``L_0`` while a
threshold-activated counterflow of latent traders opposes the induced
displacement ``D_t = X_t - m_t`` relative to the counterfactual reference
price ``m_t`` (Eqs. 2-4, 28, 30):

    dD_t = [q_t - rho(sgn(D_t) Y_t) * A_t(D_t)] / L_0 dt,   D_0 = 0,

where ``A(D) = sgn(D) int_0^{|D|} (|D| - chi) Pi(d chi)`` aggregates linear
individual responses over a threshold measure ``Pi`` (Eq. 20) and
``rho(sgn(D) Y)`` is a logistic pool intensity (Eq. 29) recording depletion
of the opposing side by past order flow. The latent state ``Y_t`` obeys an
overdamped generalized Langevin equation (Eq. 8) whose memory kernels are
finite sums of exponentials (Eqs. 10-11), so the Markovian lift of
Proposition 1 is exact, not a truncation: the state
``Z = (D, Y, h_1..h_N, g_1..g_M)`` evolves by the lifted system (Eqs. 15-16,
59)

    dY   = [-U'(Y) - sum_i a_i h_i + sum_j g_j] dt,
    dh_i = [dY/dt - gamma_i h_i] dt - sigma_{Y,i} dW_i,
    dg_j = [-lambda_j g_j + c_j q_t] dt,

with stationary-OU initialization ``eta_{i,0} ~ N(0, sigma_Y^2 / a_i)`` and
``h_{i,0} = -eta_{i,0}`` (``sigma_{Y,i} = sigma_Y sqrt(2 gamma_i / a_i)``).

Simulation scheme. The ``h``-modes and ``g``-modes are integrated exactly
under a piecewise-constant latent drift (exponential Euler, no history
retention); the displacement uses a semi-implicit backward step on the
counterflow feedback, solved per path by Newton iteration — the implicit
map is strictly increasing (slope >= 1), so the root is unique. Ambient
price noise cancels in the displacement by construction (Eq. 31); a
reference Brownian path ``m_t`` is still generated so that log-price
conventions can be checked end to end.

Closed-form material implemented here (all verified in tests):

- Prop. 2 / Eq. 25: exponential threshold density gives
  ``A(D) = q* sgn(D) (|D|/d_c - 1 + exp(-|D|/d_c))``, quadratic onset
  ``omega D|D|`` with ``omega = q*/(2 d_c^2)``, linear tail slope ``q*/d_c``.
- Prop. 3 / Eq. 35: round-trip cost identity
  ``C[q] = (L_0/2) E[D_H^2] + E[int q^cf_t D_t dt] >= 0``.
- Prop. 4 / Eq. 45: fresh-pool impact under quadratic counterflow,
  ``I_T(Q) = sqrt(Q/(omega T)) tanh(sqrt(omega Q T)/L_0)``; square-root
  range of Eq. 46 and counterflow time scale ``tau_cf`` of Eq. 47.
- Eq. 48: vol-scaled thresholds ``d_c(t) = c_d sigma_t sqrt(tau_d)`` make
  sqrt-regime impact proportional to volatility.
- Eq. 50 / Prop. 5 (Eqs. 53-54): hyperbolic post-execution relaxation and
  pathwise bounds for a stochastic pool with ``rho in [rho_min, rho_max]``.
- Remark 3 / Eq. 52: duration-free thresholds (``tau_d = c_tau T`` ->
  ``tanh`` form; elapsed noise ``s_t = sigma sqrt(c_tau t)`` -> Bessel
  ``I_1/I_0`` form, Appendix E).
- Sec. 2.6 nested hierarchy: Kyle (no counterflow), fresh pool
  (``rho == 1``), single-mode pool (geometric-mean rates, integrated
  strengths preserved), full GLE pool — the same variant axes the paper's
  Figures/Tables 3-6 compare. Baseline parameterization is Table 1.

Honesty. Everything here is a SYNTHETIC model object: Monte-Carlo outputs
are correctness diagnostics of the paper's construction (impact-vs-size
local exponent, vol scaling, duration invariance, depletion narrowing,
nonnegative round-trip cost, post-execution relaxation), never market
evidence. Per-path execution PnL-like quantities are namespaced
``sim_internal_*`` inside ``ImpactPaths.sim_internal`` and are deliberately
NOT surfaced by ``impact_curve``/``regime_scan``/``bench_langevin_impact``
report blobs. Impact is reported in log-price (displacement) units
``p_0 = sigma_X sqrt(tau_0)``, matching the paper's normalization — a
model-implied displacement, not a Sharpe-family performance metric. No
live-trading claims and no broker connectivity are made.

Composition notes. This lane composes, never re-implements:
``execution/almgren_chriss.slice_trades`` converts worked-order holdings to
child-flow rates (``trajectory_rate_schedule``);
``execution/impact.pow_law_total_impact`` supplies the Almgren-style
power-law benchmark delta reported next to model impacts in the bench; and
``models/volatility.ewma_variance`` supplies the EWMA sigma estimator used
by the vol-scaling experiment (``ewma_sigma``). Propagator/permanent-impact
machinery in ``execution/impact.py`` is NOT reused for the model core
because the paper's mechanism (threshold counterflow + GLE pool) is
dynamically distinct and must be implemented verbatim.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.special import i0 as _bess_i0
from scipy.special import i1 as _bess_i1

from quant_fund.execution.almgren_chriss import slice_trades
from quant_fund.execution.impact import pow_law_total_impact
from quant_fund.models.volatility import ewma_variance

Array = NDArray[np.float64]
RateFn = Callable[[float], float]

COUNTERFLOW_KINDS: tuple[str, str] = ("exponential", "quadratic")
THRESHOLD_SCALINGS: tuple[str, ...] = ("fixed", "vol", "duration", "elapsed")
SCHEMA = "langevin_impact.v1"

# Table-1 baseline reference units of the paper.
_BASELINE_IMPACTS_FRESH: dict[float, float] = {
    0.1: 0.465086,
    0.3: 0.269390,
    1.0: 0.144834,
    3.0: 0.082776,
    10.0: 0.045057,
    30.0: 0.025932,
}


def _positive_scalar(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return v


def _finite_scalar(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _positive_tuple(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    out = tuple(float(v) for v in values)
    if not all(np.isfinite(v) and v > 0.0 for v in out):
        raise ValueError(f"{name} must contain positive finite entries")
    return out


def _monotone_times(times: Array, name: str = "times") -> Array:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{name} must be a finite strictly increasing array of length >= 2")
    return t


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangevinImpactConfig:
    """Model vector ``Theta`` of the paper (Eq. 61), flattened.

    ``counterflow_scale`` is the threshold scale ``d_c`` when
    ``threshold_scaling == "fixed"`` and the dimensionless normalized
    threshold ``c_d`` otherwise (Eqs. 24, 27). ``threshold_scaling`` selects
    the noise-units convention: ``"fixed"`` (constant ``d_c``), ``"vol"``
    (``d_c = c_d sigma_X sqrt(tau_d)``), ``"duration"``
    (``d_c = c_d sigma_X sqrt(c_tau T_exec)``, Remark 3 first form) and
    ``"elapsed"`` (``d_c(t) = c_d sigma_X sqrt(c_tau (t - t_0))``, Remark 3
    second form). ``counterflow_kind`` selects the aggregate response:
    ``"exponential"`` uses the Eq. 25 closed form of the exponential
    threshold density; ``"quadratic"`` uses its quadratic onset
    ``A(D) = omega D|D|`` with ``omega = q* / (2 d_c^2)``.
    """

    depth: float = 1.0
    u2: float = 1.0
    u3: float = 0.0
    u4: float = 0.1
    intrinsic_weights: tuple[float, ...] = (0.5, 0.05)
    intrinsic_rates: tuple[float, ...] = (1.0, 0.1)
    flow_amplitudes: tuple[float, ...] = (5.0, 0.5)
    flow_rates: tuple[float, ...] = (2.0, 0.2)
    pool_y_rho: float = 2.0
    pool_floor: float = 0.3
    pool_enabled: bool = True
    counterflow_enabled: bool = True
    counterflow_kind: str = "exponential"
    counterflow_q_star: float = 100.0
    counterflow_scale: float = 1.0
    counterflow_atom: float = 0.0
    threshold_scaling: str = "fixed"
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    tau_d: float = 1.0
    c_tau: float = 1.0

    def __post_init__(self) -> None:
        _positive_scalar(self.depth, "depth")
        _positive_scalar(self.u2, "u2")
        _finite_scalar(self.u3, "u3")
        _positive_scalar(self.u4, "u4")
        a = _positive_tuple(self.intrinsic_weights, "intrinsic_weights")
        g = _positive_tuple(self.intrinsic_rates, "intrinsic_rates")
        if len(a) != len(g):
            raise ValueError("intrinsic_weights and intrinsic_rates must match length")
        c = _positive_tuple(self.flow_amplitudes, "flow_amplitudes")
        lam = _positive_tuple(self.flow_rates, "flow_rates")
        if len(c) != len(lam):
            raise ValueError("flow_amplitudes and flow_rates must match length")
        _positive_scalar(self.pool_y_rho, "pool_y_rho")
        fl = _finite_scalar(self.pool_floor, "pool_floor")
        if not 0.0 <= fl < 1.0:
            raise ValueError("pool_floor must be in [0, 1)")
        _positive_scalar(self.counterflow_q_star, "counterflow_q_star")
        _positive_scalar(self.counterflow_scale, "counterflow_scale")
        at = _finite_scalar(self.counterflow_atom, "counterflow_atom")
        if at < 0.0:
            raise ValueError("counterflow_atom must be non-negative")
        _positive_scalar(self.sigma_x, "sigma_x")
        sy = _finite_scalar(self.sigma_y, "sigma_y")
        if sy < 0.0:
            raise ValueError("sigma_y must be non-negative")
        _positive_scalar(self.tau_d, "tau_d")
        _positive_scalar(self.c_tau, "c_tau")
        if self.counterflow_kind not in COUNTERFLOW_KINDS:
            raise ValueError(f"counterflow_kind must be one of {COUNTERFLOW_KINDS}")
        if self.threshold_scaling not in THRESHOLD_SCALINGS:
            raise ValueError(f"threshold_scaling must be one of {THRESHOLD_SCALINGS}")

    @property
    def n_intrinsic(self) -> int:
        return len(self.intrinsic_rates)

    @property
    def n_flow(self) -> int:
        return len(self.flow_rates)


def baseline_config() -> LangevinImpactConfig:
    """Table-1 baseline parameterization (reference units tau_0=p_0=L_0=Q_0=1)."""
    return LangevinImpactConfig()


def kyle_config() -> LangevinImpactConfig:
    """Nested-hierarchy Kyle benchmark: Pi = 0, impact linear and permanent."""
    return replace(baseline_config(), counterflow_enabled=False, pool_enabled=False)


def fresh_pool_config(**overrides: float | bool | str) -> LangevinImpactConfig:
    """Fresh pool rho == 1: counterflow depends on displacement only."""
    base = replace(baseline_config(), pool_enabled=False, sigma_y=0.0)
    return replace(base, **overrides)  # type: ignore[arg-type]


def single_mode_config() -> LangevinImpactConfig:
    """Single-mode pool (Sec. 2.6): N = M = 1 at geometric-mean rates.

    Rates are the geometric means of the baseline rates and the weights
    preserve the integrated strengths ``sum a_i/gamma_i`` and
    ``sum c_j/lambda_j`` of the baseline kernels.
    """
    base = baseline_config()
    gam = math.sqrt(base.intrinsic_rates[0] * base.intrinsic_rates[1])
    lam = math.sqrt(base.flow_rates[0] * base.flow_rates[1])
    s_yy = sum(a / g for a, g in zip(base.intrinsic_weights, base.intrinsic_rates, strict=True))
    s_yx = sum(c / r for c, r in zip(base.flow_amplitudes, base.flow_rates, strict=True))
    return replace(
        base,
        intrinsic_weights=(s_yy * gam,),
        intrinsic_rates=(gam,),
        flow_amplitudes=(s_yx * lam,),
        flow_rates=(lam,),
    )


def broad_spectrum_config(n_modes: int = 4, breadth: float = 1000.0) -> LangevinImpactConfig:
    """Broader spectrum of Sec. 7.3: log-spaced rates around baseline
    geometric means, integrated strengths ``sum a_i/gamma_i`` and
    ``sum c_j/lambda_j`` preserved."""
    base = baseline_config()
    if int(n_modes) != n_modes or n_modes < 2:
        raise ValueError("n_modes must be an integer >= 2")
    _positive_scalar(breadth, "breadth")
    n = int(n_modes)
    g0 = math.sqrt(base.intrinsic_rates[0] * base.intrinsic_rates[1])
    l0 = math.sqrt(base.flow_rates[0] * base.flow_rates[1])
    span = math.log(breadth)
    gam = np.exp(np.linspace(math.log(g0) - span / 2, math.log(g0) + span / 2, n))
    lam = np.exp(np.linspace(math.log(l0) - span / 2, math.log(l0) + span / 2, n))
    s_yy = sum(a / g for a, g in zip(base.intrinsic_weights, base.intrinsic_rates, strict=True))
    s_yx = sum(c / r for c, r in zip(base.flow_amplitudes, base.flow_rates, strict=True))
    a = s_yy / np.sum(1.0 / gam)
    c = s_yx / np.sum(1.0 / lam)
    return replace(
        base,
        intrinsic_weights=tuple(float(a) for _ in range(n)),
        intrinsic_rates=tuple(float(v) for v in gam),
        flow_amplitudes=tuple(float(c) for _ in range(n)),
        flow_rates=tuple(float(v) for v in lam),
    )


# ---------------------------------------------------------------------------
# Counterflow and pool primitives (Eqs. 19-29)
# ---------------------------------------------------------------------------


def exponential_threshold_density(chi: Array, q_star: float, d_c: float) -> Array:
    """Exponential threshold density ``(q*/d_c^2) exp(-chi/d_c)`` (Eq. 24)."""
    _positive_scalar(q_star, "q_star")
    _positive_scalar(d_c, "d_c")
    x = np.asarray(chi, dtype=float)
    if np.any(x < 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("chi must be a finite non-negative array")
    return (q_star / d_c**2) * np.exp(-x / d_c)


def aggregate_counterflow(
    displacement: Array,
    thresholds: Array,
    density: Array,
    *,
    atom: float = 0.0,
) -> Array:
    """Numerical aggregate ``A(D) = sgn(D) int_0^{|D|} (|D|-chi) Pi(d chi)``.

    ``thresholds``/``density`` give a quadrature grid for the continuous
    part of Eq. 20; ``atom`` is the point mass ``pi_0`` at zero threshold.
    """
    d = np.asarray(displacement, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("displacement must be finite")
    chi = np.asarray(thresholds, dtype=float).reshape(-1)
    pi = np.asarray(density, dtype=float).reshape(-1)
    if chi.size != pi.size or chi.size < 2:
        raise ValueError("thresholds and density must match length >= 2")
    if np.any(np.diff(chi) < 0.0) or chi[0] < 0.0:
        raise ValueError("thresholds must be sorted and non-negative")
    a0 = _finite_scalar(atom, "atom")
    if a0 < 0.0:
        raise ValueError("atom must be non-negative")
    kernel = np.clip(np.abs(d)[..., None] - chi[None, :], 0.0, None)
    mass = np.trapezoid(kernel * pi[None, :], chi, axis=-1)
    return np.asarray(np.sign(d) * (mass + a0 * np.abs(d)), dtype=np.float64)


def counterflow_rate(
    displacement: Array,
    *,
    kind: str,
    q_star: float,
    d_c: Array | float,
    atom: float = 0.0,
) -> Array:
    """Aggregate counterflow ``A(D)`` (plus optional zero atom ``pi_0 D``).

    ``kind="exponential"`` is the Eq. 25 closed form; ``kind="quadratic"``
    is the quadratic onset ``omega D|D|`` with ``omega = q*/(2 d_c^2)``.
    ``d_c`` may be scalar or broadcast to ``displacement``.
    """
    d = np.asarray(displacement, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("displacement must be finite")
    _positive_scalar(q_star, "q_star")
    dc = np.asarray(d_c, dtype=float)
    if np.any(dc <= 0.0) or not np.all(np.isfinite(dc)):
        raise ValueError("d_c must be positive and finite")
    a0 = _finite_scalar(atom, "atom")
    if a0 < 0.0:
        raise ValueError("atom must be non-negative")
    u = np.abs(d) / dc
    if kind == "exponential":
        out = q_star * np.sign(d) * (u - 1.0 + np.exp(-u))
    elif kind == "quadratic":
        omega = q_star / (2.0 * dc**2)
        out = omega * d * np.abs(d)
    else:
        raise ValueError(f"kind must be one of {COUNTERFLOW_KINDS}")
    return np.asarray(out + a0 * d, dtype=np.float64)


def counterflow_slope(
    displacement: Array,
    *,
    kind: str,
    q_star: float,
    d_c: Array | float,
    atom: float = 0.0,
) -> Array:
    """Derivative ``A'(D) >= 0`` of the aggregate counterflow."""
    d = np.asarray(displacement, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("displacement must be finite")
    _positive_scalar(q_star, "q_star")
    dc = np.asarray(d_c, dtype=float)
    if np.any(dc <= 0.0) or not np.all(np.isfinite(dc)):
        raise ValueError("d_c must be positive and finite")
    u = np.abs(d) / dc
    a0 = _finite_scalar(atom, "atom")
    if a0 < 0.0:
        raise ValueError("atom must be non-negative")
    if kind == "exponential":
        out = (q_star / dc) * (-np.expm1(-u))
    elif kind == "quadratic":
        omega = q_star / (2.0 * dc**2)
        out = 2.0 * omega * np.abs(d)
    else:
        raise ValueError(f"kind must be one of {COUNTERFLOW_KINDS}")
    return np.asarray(out + a0, dtype=np.float64)


def quadratic_onset_coefficient(q_star: float, d_c: float) -> float:
    """Quadratic-onset coefficient ``omega = q*/(2 d_c^2)`` (Prop. 2.ii)."""
    _positive_scalar(q_star, "q_star")
    _positive_scalar(d_c, "d_c")
    return float(q_star / (2.0 * d_c**2))


def counterflow_time_scale(omega: float, chi_c: float, depth: float) -> float:
    """Counterflow time scale ``tau_cf = L_0 / (omega chi_c)`` (Eq. 47)."""
    return float(
        _positive_scalar(depth, "depth")
        / (_positive_scalar(omega, "omega") * _positive_scalar(chi_c, "chi_c"))
    )


def pool_intensity(y: Array, *, y_rho: float, floor: float, enabled: bool = True) -> Array:
    """Logistic pool map ``rho(y)`` of Eq. 29 (``rho(0)=1``, ``rho+rho(-.)=2``).

    ``enabled=False`` yields the fresh pool ``rho == 1``.
    """
    v = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("y must be finite")
    if not enabled:
        return np.ones_like(v)
    _positive_scalar(y_rho, "y_rho")
    fl = _finite_scalar(floor, "floor")
    if not 0.0 <= fl < 1.0:
        raise ValueError("floor must be in [0, 1)")
    return np.asarray(fl + 2.0 * (1.0 - fl) / (1.0 + np.exp(v / y_rho)), dtype=np.float64)


def threshold_scale_at(
    t_mid: float,
    cfg: LangevinImpactConfig,
    exec_horizon: float,
    t_start: float = 0.0,
) -> float:
    """Noise-units threshold scale ``d_c(t)`` (Eq. 27 / Remark 3)."""
    tm = float(t_mid)
    if cfg.threshold_scaling == "fixed":
        return float(cfg.counterflow_scale)
    if cfg.threshold_scaling == "vol":
        return float(cfg.counterflow_scale * cfg.sigma_x * math.sqrt(cfg.tau_d))
    if cfg.threshold_scaling == "duration":
        return float(cfg.counterflow_scale * cfg.sigma_x * math.sqrt(cfg.c_tau * exec_horizon))
    # "elapsed": noise accumulated since the order began at ``t_start``.
    return float(
        cfg.counterflow_scale * cfg.sigma_x * math.sqrt(cfg.c_tau * max(tm - t_start, 1e-12))
    )


# ---------------------------------------------------------------------------
# Order schedules (Eqs. 36-37, 63, 69-70)
# ---------------------------------------------------------------------------


def constant_rate_schedule(quantity: float, horizon: float) -> RateFn:
    """Constant-rate metaorder ``q_t = Q/T 1_[0,T]`` (Eq. 63)."""
    q = _finite_scalar(quantity, "quantity")
    t_ = _positive_scalar(horizon, "horizon")
    rate = q / t_

    def _rate(t: float) -> float:
        return rate if 0.0 <= t < t_ else 0.0

    return _rate


def shaped_rate_schedule(
    quantity: float, horizon: float, shape: Callable[[float], float]
) -> RateFn:
    """Scaled schedule ``q_t = Q psi_T(t)`` with ``int_0^T psi = 1`` (Eq. 37)."""
    q = _finite_scalar(quantity, "quantity")
    t_ = _positive_scalar(horizon, "horizon")
    grid = np.linspace(0.0, t_, 4001)
    vals = np.array([shape(s) for s in grid], dtype=float)
    if not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
        raise ValueError("shape must be finite and non-negative on [0, T]")
    norm = float(np.trapezoid(vals, grid))
    if norm <= 0.0:
        raise ValueError("shape must have positive integral")

    def _rate(t: float) -> float:
        return q * shape(t) / norm if 0.0 <= t < t_ else 0.0

    return _rate


def flat_shape(s: float) -> float:
    return 1.0


def front_loaded_shape(horizon: float) -> Callable[[float], float]:
    """Front-loaded profile ``psi_T(s) = 2(T-s)/T^2`` (Eq. 69)."""
    t_ = _positive_scalar(horizon, "horizon")
    return lambda s: 2.0 * (t_ - s) / t_**2


def back_loaded_shape(horizon: float) -> Callable[[float], float]:
    """Back-loaded profile ``psi_T(s) = 2s/T^2`` (Eq. 69)."""
    t_ = _positive_scalar(horizon, "horizon")
    return lambda s: 2.0 * s / t_**2


def pause_shape(horizon: float, kappa: float = 0.3) -> Callable[[float], float]:
    """Interrupted profile with symmetric pause fraction ``kappa`` (Eq. 70)."""
    t_ = _positive_scalar(horizon, "horizon")
    k = _finite_scalar(kappa, "kappa")
    if not 0.0 < k < 1.0:
        raise ValueError("kappa must be in (0, 1)")

    def _shape(s: float) -> float:
        return 1.0 if (s <= (1.0 - k) * t_ / 2.0 or s >= (1.0 + k) * t_ / 2.0) else 0.0

    return _shape


def piecewise_rate_schedule(rates: Array, edges: Array) -> RateFn:
    """Piecewise-constant rate: ``rates[i]`` on ``[edges[i], edges[i+1])``."""
    r = np.asarray(rates, dtype=float).reshape(-1)
    e = _monotone_times(edges, "edges")
    if r.size != e.size - 1:
        raise ValueError("rates must have len(edges) - 1 entries")
    if not np.all(np.isfinite(r)):
        raise ValueError("rates must be finite")

    def _rate(t: float) -> float:
        i = int(np.searchsorted(e, t, side="right")) - 1
        return float(r[i]) if 0 <= i < r.size else 0.0

    return _rate


def trajectory_rate_schedule(holdings: Array, horizon: float) -> RateFn:
    """Rate schedule implied by a worked-order holdings trajectory.

    Composes ``execution/almgren_chriss.slice_trades``: child quantities are
    per-cell sells of holdings (sell-positive convention there), negated so
    the rate is signed order flow ``dH/dt`` — buys positive, matching the
    displacement equation ``dD = (q - q^cf)/L0 dt`` where positive ``q``
    raises ``D``.
    """
    h = np.asarray(holdings, dtype=float).reshape(-1)
    if h.size < 2 or not np.all(np.isfinite(h)):
        raise ValueError("holdings must be a finite array of length >= 2")
    t_ = _positive_scalar(horizon, "horizon")
    trades = slice_trades(h)
    edges = np.linspace(0.0, t_, h.size)
    return piecewise_rate_schedule(-trades / (t_ / (h.size - 1)), edges)


# ---------------------------------------------------------------------------
# Lifted GLE simulator (Eqs. 15-17, 28, 30-31, 59)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactPaths:
    """Simulated lifted-state paths of one metaorder experiment.

    All arrays have shape ``(n_paths, n_times)`` except mode arrays which
    add the mode axis. ``sim_internal`` holds per-path simulator-internal
    diagnostics (namespaced ``sim_internal_*``); these are correctness
    material and must never appear in report blobs.
    """

    times: Array
    rates: Array
    displacement: Array
    latent: Array
    pool_intensity: Array
    counterflow: Array
    reference: Array
    log_price: Array
    h_modes: Array | None
    g_modes: Array | None
    sim_internal: dict[str, Array]
    exec_horizon: float
    depth: float
    n_paths: int


def _damped_newton(
    fval_fn: Callable[[Array], Array], fp_fn: Callable[[Array], Array], x0: Array
) -> Array:
    """Vectorized damped Newton on a per-path scalar root.

    ``fval_fn``/``fp_fn`` return the residual and its derivative. Steps are
    damped by ``max(f', 1)`` so the update is a contraction whenever the
    residual slope is positive — the case for every implicit equation in
    this module (counterflow slope ``>= 0``, confining potential ``U''
    bounded below for the baseline even potential).
    """
    x = np.asarray(x0, dtype=float).copy()
    x = np.where(np.isfinite(x), x, 0.0)
    for _ in range(14):
        f = fval_fn(x)
        fp = fp_fn(x)
        step = f / np.maximum(fp, 1.0)
        x = x - step
        if float(np.max(np.abs(step))) < 1e-13 * (1.0 + float(np.max(np.abs(x)))):
            break
    resid = fval_fn(x)
    bad = ~np.isfinite(resid) | (np.abs(resid) > 1e-9 * (1.0 + np.abs(x)))
    if np.any(bad):
        # Bisection fallback: bracket expands until f changes sign. Used only
        # when Newton fails (never for the monotone baseline equations).
        lo = np.minimum(x, x0) - 1.0
        hi = np.maximum(x, x0) + 1.0
        flo = fval_fn(lo)
        fhi = fval_fn(hi)
        grow = np.ones_like(x)
        for _ in range(60):
            need = flo * fhi > 0.0
            if not np.any(need):
                break
            grow = np.where(need & (np.abs(flo) < np.abs(fhi)), grow * 4.0, grow)
            lo = np.where(need & (np.abs(flo) < np.abs(fhi)), lo - grow, lo)
            hi = np.where(need & ~(np.abs(flo) < np.abs(fhi)), hi + grow, hi)
            flo = fval_fn(lo)
            fhi = fval_fn(hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = fval_fn(mid)
            go_right = fm * flo <= 0.0
            hi = np.where(go_right, mid, hi)
            lo = np.where(go_right, lo, mid)
            flo = np.where(go_right, flo, fm)
        x = np.where(bad, 0.5 * (lo + hi), x)
    return np.asarray(x, dtype=np.float64)


def _newton_displacement(
    d_prev: Array,
    y_next: Array,
    q_bar: float,
    d_c: float,
    dt: float,
    cfg: LangevinImpactConfig,
) -> Array:
    """Semi-implicit backward step on the displacement drift.

    Solves ``x = d_prev + dt/L0 * (q_bar - rho(sgn(x) y_next) A(x))`` per
    path. The implicit map ``f(x)`` has slope ``>= 1`` so the root is
    unique; Newton on it converges from the explicit predictor, with a
    bisection fallback (never expected in practice).
    """
    l0 = cfg.depth
    enabled = cfg.counterflow_enabled
    kind = cfg.counterflow_kind
    qs = cfg.counterflow_q_star if enabled else 0.0
    at = cfg.counterflow_atom

    def _rho(x: Array) -> Array:
        return pool_intensity(
            np.sign(x) * y_next,
            y_rho=cfg.pool_y_rho,
            floor=cfg.pool_floor,
            enabled=cfg.pool_enabled,
        )

    if not enabled:
        return np.asarray(d_prev + (dt / l0) * q_bar, dtype=np.float64)
    x0 = d_prev + (dt / l0) * (
        q_bar - _rho(d_prev) * counterflow_rate(d_prev, kind=kind, q_star=qs, d_c=d_c, atom=at)
    )

    def _f(x: Array) -> Array:
        return (
            x
            - d_prev
            - (dt / l0)
            * (q_bar - _rho(x) * counterflow_rate(x, kind=kind, q_star=qs, d_c=d_c, atom=at))
        )

    def _fp(x: Array) -> Array:
        return 1.0 + (dt / l0) * _rho(x) * counterflow_slope(
            x, kind=kind, q_star=qs, d_c=d_c, atom=at
        )

    return _damped_newton(_f, _fp, x0)


def _newton_latent(y_prev: Array, c_const: Array, dt: float, cfg: LangevinImpactConfig) -> Array:
    """Drift-implicit latent update: solve ``y = y_prev + dt (C - U'(y))``.

    Keeps the quartic confinement stable at large forcing (the explicit
    Euler step diverges for ``|y|`` beyond ``sqrt(2/(u4 dt))``). ``C`` is
    the frozen non-potential part of the latent drift, ``-sum a_i h_i +
    sum_j g_j``.
    """
    u2, u3, u4 = cfg.u2, cfg.u3, cfg.u4
    x0 = y_prev + dt * (c_const - (u2 * y_prev + u3 * y_prev**2 + u4 * y_prev**3))
    scale = np.power(np.abs(c_const) / max(u4, 1e-12), 1.0 / 3.0)
    x0 = np.where(np.isfinite(x0), x0, np.sign(c_const) * np.maximum(scale, 1.0))

    def _f(x: Array) -> Array:
        return x - y_prev - dt * (c_const - (u2 * x + u3 * x**2 + u4 * x**3))

    def _fp(x: Array) -> Array:
        return 1.0 + dt * (u2 + 2.0 * u3 * x + 3.0 * u4 * x**2)

    return _damped_newton(_f, _fp, x0)


def simulate_paths(
    cfg: LangevinImpactConfig,
    rate: RateFn,
    times: Array,
    *,
    n_paths: int = 128,
    exec_horizon: float | None = None,
    seed: int | np.random.Generator = 0,
    noise: Array | None = None,
    init_noise: Array | None = None,
    ref_noise: Array | None = None,
    store_modes: bool = False,
) -> ImpactPaths:
    """Monte-Carlo simulation of the lifted system on a time grid.

    ``rate`` is the signed trading rate ``q_t`` (units of volume per time);
    ``times`` is the observation grid (``>= 2`` strictly increasing points,
    starting at the order start). ``exec_horizon`` is the order duration
    ``T`` used by ``threshold_scaling="duration"`` and by diagnostics that
    split execution from post-execution; defaults to ``times[-1]``.
    ``noise``/``init_noise``/``ref_noise`` optionally inject standard-normal
    innovations (shapes ``(P, K, N)``, ``(P, N)``, ``(P, K)``) for common-
    random-number pairing across configurations with equal mode counts.
    """
    if not isinstance(cfg, LangevinImpactConfig):
        raise TypeError("cfg must be a LangevinImpactConfig")
    t = _monotone_times(times)
    if not callable(rate):
        raise TypeError("rate must be callable")
    p = int(n_paths)
    if isinstance(n_paths, bool) or n_paths != p or p < 1:
        raise ValueError("n_paths must be an integer >= 1")
    t0 = float(t[0])
    if t0 < 0.0:
        raise ValueError("times must start at t >= 0")
    horizon = (
        float(t[-1]) if exec_horizon is None else _positive_scalar(exec_horizon, "exec_horizon")
    )
    if horizon > float(t[-1]) + 1e-12:
        raise ValueError("exec_horizon must not exceed times[-1]")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    k_steps = t.size - 1
    n_modes = cfg.n_intrinsic
    m_modes = cfg.n_flow
    gam = np.asarray(cfg.intrinsic_rates, dtype=float)
    a_w = np.asarray(cfg.intrinsic_weights, dtype=float)
    lam = np.asarray(cfg.flow_rates, dtype=float)
    c_a = np.asarray(cfg.flow_amplitudes, dtype=float)
    dt_all = np.diff(t)

    if noise is not None:
        z = np.asarray(noise, dtype=float)
        if z.shape != (p, k_steps, n_modes):
            raise ValueError(f"noise must have shape ({p}, {k_steps}, {n_modes})")
    else:
        z = rng.standard_normal((p, k_steps, n_modes))
    if init_noise is not None:
        z0 = np.asarray(init_noise, dtype=float)
        if z0.shape != (p, n_modes):
            raise ValueError(f"init_noise must have shape ({p}, {n_modes})")
    else:
        z0 = rng.standard_normal((p, n_modes))
    if ref_noise is not None:
        zr = np.asarray(ref_noise, dtype=float)
        if zr.shape != (p, k_steps):
            raise ValueError(f"ref_noise must have shape ({p}, {k_steps})")
    else:
        zr = rng.standard_normal((p, k_steps))

    # Proposition-1 initial law: eta_i0 ~ N(0, sigma_Y^2 / a_i), h0 = -eta0.
    eta_scale = cfg.sigma_y / np.sqrt(a_w)
    h = -z0 * eta_scale[None, :]
    g = np.zeros((p, m_modes))
    d_state = np.zeros(p)
    y_state = np.zeros(p)
    m_ref = np.zeros(p)

    disp = np.empty((p, t.size))
    lat = np.empty((p, t.size))
    pool = np.empty((p, t.size))
    cf = np.empty((p, t.size))
    ref = np.empty((p, t.size))
    price = np.empty((p, t.size))
    hh = np.empty((p, t.size, n_modes)) if store_modes else None
    gg = np.empty((p, t.size, m_modes)) if store_modes else None
    rates = np.empty(k_steps)

    disp[:, 0] = d_state
    lat[:, 0] = y_state
    pool[:, 0] = 1.0
    cf[:, 0] = 0.0
    ref[:, 0] = m_ref
    price[:, 0] = d_state + m_ref
    if hh is not None:
        hh[:, 0, :] = h
    if gg is not None:
        gg[:, 0, :] = g

    for k in range(k_steps):
        dt = float(dt_all[k])
        tm = 0.5 * (t[k] + t[k + 1])
        q_bar = float(rate(tm))
        if not math.isfinite(q_bar):
            raise ValueError(f"rate returned non-finite value at t={tm}")
        rates[k] = q_bar
        dc = threshold_scale_at(tm, cfg, horizon, t_start=t0)

        # Latent drift F = -U'(Y) - sum a_i h_i + sum g_j (Eq. 16). The
        # non-potential part is frozen at its step-open value; the potential
        # part is integrated drift-implicitly so the quartic confinement is
        # stable for arbitrarily large order rates.
        up = cfg.u2 * y_state + cfg.u3 * y_state**2 + cfg.u4 * y_state**3
        c_const = -h @ a_w + g.sum(axis=1)
        drift_f = -up + c_const
        y_new = _newton_latent(y_state, c_const, dt, cfg)

        # Exact mode updates under piecewise-constant drift (exponential
        # Euler): h_i receives the integrated dY increment filtered by its
        # own rate plus the stationary-OU innovation (Eq. 15).
        eg = np.exp(-gam * dt)
        h = (
            eg[None, :] * h
            + drift_f[:, None] * ((1.0 - eg) / gam)[None, :]
            - z[:, k, :] * (cfg.sigma_y * np.sqrt(-np.expm1(-2.0 * gam * dt) / a_w))[None, :]
        )
        el = np.exp(-lam * dt)
        g = el[None, :] * g + (c_a * q_bar)[None, :] * ((1.0 - el) / lam)[None, :]

        # Semi-implicit displacement step on the counterflow feedback.
        d_state = _newton_displacement(d_state, y_new, q_bar, dc, dt, cfg)

        y_state = y_new

        # Reference diffusion (shared with the price, cancels in D; Eq. 3).
        m_ref = m_ref + cfg.sigma_x * math.sqrt(dt) * zr[:, k]

        rho_now = pool_intensity(
            np.sign(d_state) * y_state,
            y_rho=cfg.pool_y_rho,
            floor=cfg.pool_floor,
            enabled=cfg.pool_enabled,
        )
        cf_now = (
            rho_now
            * counterflow_rate(
                d_state,
                kind=cfg.counterflow_kind,
                q_star=cfg.counterflow_q_star,
                d_c=dc,
                atom=cfg.counterflow_atom,
            )
            if cfg.counterflow_enabled
            else np.zeros(p)
        )
        disp[:, k + 1] = d_state
        lat[:, k + 1] = y_state
        pool[:, k + 1] = rho_now
        cf[:, k + 1] = cf_now
        ref[:, k + 1] = m_ref
        price[:, k + 1] = d_state + m_ref
        if hh is not None:
            hh[:, k + 1, :] = h
        if gg is not None:
            gg[:, k + 1, :] = g

    # Simulator-internal diagnostics (namespaced; never in report blobs):
    # per-path execution price ratio int |q| S dt / int |q| dt over the
    # execution window (S = exp(X), X_0 = 0 so S_0 = 1) and net traded volume.
    exec_idx = min(max(int(np.searchsorted(t, horizon, side="left")), 1), k_steps)
    qty = np.abs(rates[:exec_idx]) * dt_all[:exec_idx]
    qty_tot = float(qty.sum())
    if qty_tot > 0.0:
        vwap_paths = np.sum(qty[None, :] * np.exp(price[:, 1 : exec_idx + 1]), axis=1) / qty_tot
    else:
        vwap_paths = np.ones(p)
    sim_internal = {
        "sim_internal_exec_price_ratio": vwap_paths,
        "sim_internal_net_traded": np.full(p, float(np.sum(rates * dt_all))),
    }
    return ImpactPaths(
        times=t,
        rates=rates,
        displacement=disp,
        latent=lat,
        pool_intensity=pool,
        counterflow=cf,
        reference=ref,
        log_price=price,
        h_modes=hh,
        g_modes=gg,
        sim_internal=sim_internal,
        exec_horizon=horizon,
        depth=cfg.depth,
        n_paths=p,
    )


def _exec_index(paths: ImpactPaths) -> int:
    idx = int(np.searchsorted(paths.times, paths.exec_horizon, side="left"))
    return max(idx, 1)


def terminal_impact(paths: ImpactPaths, at: float | None = None) -> tuple[float, float]:
    """Mean and MC standard error of the displacement at the horizon."""
    tt = paths.exec_horizon if at is None else _positive_scalar(at, "at")
    idx = int(np.searchsorted(paths.times, tt, side="left"))
    idx = min(max(idx, 1), paths.times.size - 1)
    d = paths.displacement[:, idx]
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    return mean, se


def cumulative_counterflow(paths: ImpactPaths) -> tuple[float, float]:
    """Mean and SE of ``int_0^T q^cf dt`` over the execution window."""
    idx = _exec_index(paths)
    dt = np.diff(paths.times[: idx + 1])
    cum = np.sum(
        0.5 * (paths.counterflow[:, :idx] + paths.counterflow[:, 1 : idx + 1]) * dt[None, :],
        axis=1,
    )
    mean = float(cum.mean())
    se = float(cum.std(ddof=1) / math.sqrt(cum.size)) if cum.size > 1 else 0.0
    return mean, se


def round_trip_cost(paths: ImpactPaths) -> dict[str, float]:
    """Round-trip cost identity of Proposition 3 (Eq. 35), over the grid.

    Reports the execution-weighted integral ``int q (X - X_0) dt`` (cost),
    its displacement part ``int q D dt``, the terminal-displacement term
    ``(L_0/2) E[D_H^2]``, the counterflow term ``E[int q^cf D dt]``, and
    the residual of the discrete identity (order ``O(dt)``).
    """
    if not isinstance(paths, ImpactPaths):
        raise TypeError("paths must be ImpactPaths")
    q = paths.rates
    dt = np.diff(paths.times)
    d = paths.displacement
    depth = paths.depth
    k = q.size
    dd = d[:, 1 : k + 1]
    int_q_d = float(np.mean(np.sum(q[None, :] * dd * dt[None, :], axis=1)))
    int_q_x = float(
        np.mean(np.sum(q[None, :] * paths.log_price[:, 1 : k + 1] * dt[None, :], axis=1))
    )
    cf_term = float(np.mean(np.sum(paths.counterflow[:, 1 : k + 1] * dd * dt[None, :], axis=1)))
    disp_term = float(0.5 * depth * np.mean(d[:, k] ** 2))
    rhs = disp_term + cf_term
    return {
        "cost": int_q_x,
        "order_integral_displacement": int_q_d,
        "terminal_displacement_term": disp_term,
        "counterflow_term": cf_term,
        "rhs_total": rhs,
        "identity_residual": int_q_d - rhs,
    }


# ---------------------------------------------------------------------------
# Closed forms (Eqs. 41-52)
# ---------------------------------------------------------------------------


def fresh_pool_impact_closed_form(
    quantity: float, horizon: float, omega: float, depth: float
) -> float:
    """Prop. 4 (Eq. 45): ``sqrt(Q/(omega T)) tanh(sqrt(omega Q T)/L_0)``."""
    q = _positive_scalar(quantity, "quantity")
    t_ = _positive_scalar(horizon, "horizon")
    w = _positive_scalar(omega, "omega")
    l0 = _positive_scalar(depth, "depth")
    return float(math.sqrt(q / (w * t_)) * math.tanh(math.sqrt(w * q * t_) / l0))


def small_order_expansion(quantity: float, horizon: float, omega: float, depth: float) -> float:
    """Eq. 41 second-order expansion for a constant-rate order:
    ``Q/L_0 - omega Q|Q| T / (3 L_0^3)``."""
    q = _finite_scalar(quantity, "quantity")
    t_ = _positive_scalar(horizon, "horizon")
    w = _positive_scalar(omega, "omega")
    l0 = _positive_scalar(depth, "depth")
    return float(q / l0 - w * q * abs(q) * t_ / (3.0 * l0**3))


def stationary_displacement(
    rate: float,
    *,
    kind: str,
    q_star: float,
    d_c: float,
    atom: float = 0.0,
    rho: float = 1.0,
) -> float:
    """Stationary displacement ``D_inf`` solving ``rho A(D_inf) = q``."""
    q = _finite_scalar(rate, "rate")
    _positive_scalar(rho, "rho")
    if q == 0.0:
        return 0.0
    target = q / rho
    x = math.copysign(1.0, q)
    for _ in range(60):
        ax = float(counterflow_rate(np.array([x]), kind=kind, q_star=q_star, d_c=d_c, atom=atom)[0])
        axp = float(
            counterflow_slope(np.array([x]), kind=kind, q_star=q_star, d_c=d_c, atom=atom)[0]
        )
        step = (ax - target) / max(axp, 1e-30)
        x -= step
        if abs(step) < 1e-14 * (1.0 + abs(x)):
            return float(x)
    return float(x)


def post_execution_relaxation(
    d_T: float, tau: Array, omega: float, depth: float, rho: float = 1.0
) -> Array:
    """Hyperbolic relaxation after a quadratic-counterflow execution
    (Eq. 50, pool-rescaled Eq. 54): ``D_tau = D_T/(1 + rho omega D_T tau/L_0)``."""
    d0 = _finite_scalar(d_T, "d_T")
    w = _positive_scalar(omega, "omega")
    l0 = _positive_scalar(depth, "depth")
    r = _positive_scalar(rho, "rho")
    tt = np.asarray(tau, dtype=float)
    if np.any(tt < 0.0) or not np.all(np.isfinite(tt)):
        raise ValueError("tau must be a finite non-negative array")
    return np.asarray(d0 / (1.0 + r * w * d0 * tt / l0), dtype=np.float64)


def duration_free_impact(
    quantity: float,
    *,
    sigma: float,
    c_tau: float,
    pi_hat_zero: float,
    depth: float,
    form: str = "duration",
) -> float:
    """Duration-free impact of Eq. 52 (Remark 3, quadratic onset).

    ``form="duration"`` (``tau_d = c_tau T``) gives
    ``sigma sqrt(2 c_tau Q / pi_hat(0)) tanh(x)``; ``form="elapsed"``
    (``s_t = sigma sqrt(c_tau t)``) gives
    ``sigma sqrt(2 c_tau Q / pi_hat(0)) I_1(2x)/I_0(2x)``, with
    ``x = sqrt(pi_hat(0) Q / (2 c_tau)) / (sigma L_0)``.
    """
    q = _positive_scalar(quantity, "quantity")
    s = _positive_scalar(sigma, "sigma")
    ct = _positive_scalar(c_tau, "c_tau")
    p0 = _positive_scalar(pi_hat_zero, "pi_hat_zero")
    l0 = _positive_scalar(depth, "depth")
    x = math.sqrt(p0 * q / (2.0 * ct)) / (s * l0)
    pref = s * math.sqrt(2.0 * ct * q / p0)
    if form == "duration":
        return float(pref * math.tanh(x))
    if form == "elapsed":
        return float(pref * float(_bess_i1(2.0 * x) / _bess_i0(2.0 * x)))
    raise ValueError("form must be 'duration' or 'elapsed'")


def sqrt_regime_bounds(
    omega: float, chi_c: float, horizon: float, depth: float
) -> tuple[float, float]:
    """Asymptotic square-root range of Eq. 46:
    ``L_0^2/(omega T) << Q << omega chi_c^2 T``."""
    w = _positive_scalar(omega, "omega")
    cc = _positive_scalar(chi_c, "chi_c")
    t_ = _positive_scalar(horizon, "horizon")
    l0 = _positive_scalar(depth, "depth")
    return (l0**2 / (w * t_), w * cc**2 * t_)


def pathwise_upper_bound(
    quantity: float,
    horizon: float,
    omega: float,
    depth: float,
    rho_min: float,
) -> float:
    """Prop. 5 (Eq. 53) upper bound ``D_T^{(rho_min)}`` for quadratic onset."""
    return fresh_pool_impact_closed_form(
        quantity, horizon, omega * _positive_scalar(rho_min, "rho_min"), depth
    )


# ---------------------------------------------------------------------------
# Regime diagnostics (Eqs. 64-68, Sec. 7)
# ---------------------------------------------------------------------------


def impact_curve(
    cfg: LangevinImpactConfig,
    sizes: Array,
    horizon: float,
    *,
    n_paths: int = 128,
    dt: float = 0.01,
    seed: int = 0,
) -> dict[str, Array]:
    """Terminal expected impact over a size grid (Eq. 64).

    Returns ``sizes``, ``impact`` (MC mean terminal displacement),
    ``impact_se``, centered local exponent ``exponent`` (Eq. 65, NaN at the
    endpoints), and marginal slopes (Eq. 66).
    """
    v = np.asarray(sizes, dtype=float).reshape(-1)
    if v.size < 3 or np.any(v <= 0.0) or not np.all(np.isfinite(v)):
        raise ValueError("sizes must be a finite positive array of length >= 3")
    t_ = _positive_scalar(horizon, "horizon")
    dtp = _positive_scalar(dt, "dt")
    k_steps = max(int(math.ceil(t_ / dtp)), 8)
    times = np.linspace(0.0, t_, k_steps + 1)
    n = v.size
    imp = np.empty(n)
    se = np.empty(n)
    for i, q in enumerate(v):
        rate = constant_rate_schedule(float(q), t_)
        paths = simulate_paths(
            cfg, rate, times, n_paths=n_paths, exec_horizon=t_, seed=int(seed) + i
        )
        imp[i], se[i] = terminal_impact(paths)
    exp_local = local_impact_exponents(v, imp)
    slopes = marginal_slopes(v, imp)
    return {
        "sizes": v,
        "impact": imp,
        "impact_se": se,
        "exponent": exp_local,
        "marginal_slopes": slopes,
    }


def local_impact_exponents(sizes: Array, impacts: Array) -> Array:
    """Centered local exponent ``d log I / d log Q`` (Eq. 65), NaN at ends."""
    v = np.asarray(sizes, dtype=float).reshape(-1)
    i_ = np.asarray(impacts, dtype=float).reshape(-1)
    if v.size != i_.size or v.size < 2:
        raise ValueError("sizes and impacts must match length >= 2")
    if np.any(v <= 0.0) or np.any(i_ <= 0.0):
        raise ValueError("sizes and impacts must be positive for log-log slopes")
    lv = np.log(v)
    li = np.log(i_)
    out = np.full(v.size, np.nan)
    if v.size >= 3:
        out[1:-1] = (li[2:] - li[:-2]) / (lv[2:] - lv[:-2])
    return out


def marginal_slopes(sizes: Array, impacts: Array) -> Array:
    """Marginal impact slopes ``s_k = Delta I / Delta V`` (Eq. 66)."""
    v = np.asarray(sizes, dtype=float).reshape(-1)
    i_ = np.asarray(impacts, dtype=float).reshape(-1)
    if v.size != i_.size or v.size < 2:
        raise ValueError("sizes and impacts must match length >= 2")
    if np.any(np.diff(v) <= 0.0):
        raise ValueError("sizes must be strictly increasing")
    return np.diff(i_) / np.diff(v)


def sqrt_band(
    sizes: Array,
    impacts: Array,
    *,
    tol: float = 0.1,
    min_decades: float = 1.0,
) -> dict[str, float]:
    """Operational square-root band of Eqs. 67-68.

    The band is the longest connected run of interior grid points where the
    centered exponent satisfies ``|delta - 1/2| <= tol`` while impact is
    positive increasing with nonincreasing marginal slopes (within a small
    slack). ``width`` is ``log10(V_hi/V_lo)`` in decades; ``found`` records
    the one-decade minimum-width criterion.
    """
    v = np.asarray(sizes, dtype=float).reshape(-1)
    i_ = np.asarray(impacts, dtype=float).reshape(-1)
    if v.size != i_.size or v.size < 5:
        raise ValueError("sizes and impacts must match length >= 5")
    t = _finite_scalar(tol, "tol")
    if t <= 0.0:
        raise ValueError("tol must be positive")
    md = _finite_scalar(min_decades, "min_decades")
    if md < 0.0:
        raise ValueError("min_decades must be non-negative")
    slopes = marginal_slopes(v, i_)
    exps = local_impact_exponents(v, i_)
    # Concavity slack absorbs Monte-Carlo jitter in the mean curve.
    slack = 0.02 * float(np.max(np.abs(slopes))) if slopes.size else 0.0

    def _run_ok(lo_i: int, hi_i: int) -> bool:
        seg = i_[lo_i : hi_i + 1]
        seg_slopes = np.diff(seg)
        if np.any(seg_slopes <= 0.0):
            return False
        inner = slopes[lo_i:hi_i]
        return bool(inner.size == 0 or np.all(np.diff(inner) <= slack))

    ok = np.isfinite(exps) & (np.abs(exps - 0.5) <= t)
    # Longest connected run of qualifying interior points that also satisfies
    # the concavity/monotonicity criterion.
    best = (0, -1)
    start: int | None = None
    for k in range(v.size):
        if ok[k] and start is None:
            start = k
        if (not ok[k] or k == v.size - 1) and start is not None:
            end = k - 1 if not ok[k] else k
            if end - start > best[1] - best[0] and _run_ok(start, end):
                best = (start, end)
            start = None
    lo_i, hi_i = best
    if hi_i <= lo_i:
        return {
            "found": 0.0,
            "v_lo": float("nan"),
            "v_hi": float("nan"),
            "width_decades": 0.0,
            "mean_exponent": float("nan"),
            "n_points": 0.0,
        }
    width = math.log10(v[hi_i] / v[lo_i])
    mean_exp = float(np.nanmean(exps[lo_i : hi_i + 1]))
    return {
        "found": float(width >= md),
        "v_lo": float(v[lo_i]),
        "v_hi": float(v[hi_i]),
        "width_decades": float(width),
        "mean_exponent": mean_exp,
        "n_points": float(hi_i - lo_i + 1),
    }


def regime_scan(
    cfg: LangevinImpactConfig,
    sizes: Array,
    horizons: Array,
    *,
    n_paths: int = 128,
    dt: float = 0.01,
    seed: int = 0,
) -> dict[str, Array | dict[float, dict[str, float]]]:
    """Impact curves and square-root bands across durations (Sec. 7.1)."""
    hs = np.asarray(horizons, dtype=float).reshape(-1)
    if hs.size == 0 or np.any(hs <= 0.0) or not np.all(np.isfinite(hs)):
        raise ValueError("horizons must be a finite positive array")
    impacts = np.empty((hs.size, np.asarray(sizes).size))
    bands: dict[float, dict[str, float]] = {}
    for j, t_ in enumerate(hs):
        curve = impact_curve(
            cfg, sizes, float(t_), n_paths=n_paths, dt=dt, seed=int(seed) + 1000 * j
        )
        impacts[j] = curve["impact"]
        bands[float(t_)] = sqrt_band(curve["sizes"], curve["impact"])
    return {"sizes": np.asarray(sizes), "horizons": hs, "impacts": impacts, "bands": bands}


def history_effect(
    cfg: LangevinImpactConfig,
    *,
    prior_qty: float,
    probe_qty: float,
    exec_dur: float,
    gap: float,
    obs_after: float = 0.0,
    n_paths: int = 256,
    dt: float = 0.01,
    seed: int = 0,
) -> dict[str, float]:
    """Prior-order history effect of Sec. 7.3 (Eqs. 77-79).

    Four histories share common random numbers: prior+probe, prior-only,
    probe-only, neither. The probe starts at ``t_G = T + gap``; histories
    end at ``t_G + T + obs_after``. Returns ``j_prior_probe`` (J_{2|1}),
    ``j_probe_only`` (J_{2|0}), the relative effect ``h = J_{2|1}/J_{2|0}-1``
    (Eq. 79), and paired standard errors.
    """
    qp = _positive_scalar(prior_qty, "prior_qty")
    qr = _positive_scalar(probe_qty, "probe_qty")
    t_ = _positive_scalar(exec_dur, "exec_dur")
    g_gap = _finite_scalar(gap, "gap")
    if g_gap < 0.0:
        raise ValueError("gap must be non-negative")
    obs = _finite_scalar(obs_after, "obs_after")
    if obs < 0.0:
        raise ValueError("obs_after must be non-negative")
    dtp = _positive_scalar(dt, "dt")
    t_g = t_ + g_gap
    t_end = t_g + t_ + obs
    times = np.arange(0.0, t_end + 0.5 * dtp, dtp)
    p = int(n_paths)
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((p, times.size - 1, cfg.n_intrinsic))
    z0 = rng.standard_normal((p, cfg.n_intrinsic))
    zr = rng.standard_normal((p, times.size - 1))

    def _rate(prior: bool, probe: bool) -> RateFn:
        def _r(t: float) -> float:
            v = 0.0
            if prior and 0.0 <= t < t_:
                v += qp / t_
            if probe and t_g <= t < t_g + t_:
                v += qr / t_
            return v

        return _r

    def _run(prior: bool, probe: bool) -> ImpactPaths:
        return simulate_paths(
            cfg,
            _rate(prior, probe),
            times,
            n_paths=p,
            exec_horizon=t_end,
            noise=z,
            init_noise=z0,
            ref_noise=zr,
        )

    p11 = _run(True, True)
    p10 = _run(True, False)
    p01 = _run(False, True)
    p00 = _run(False, False)
    d_end = p11.times.size - 1
    diff1 = p11.displacement[:, d_end] - p10.displacement[:, d_end]
    diff0 = p01.displacement[:, d_end] - p00.displacement[:, d_end]
    j1 = float(diff1.mean())
    j0 = float(diff0.mean())
    se1 = float(diff1.std(ddof=1) / math.sqrt(p)) if p > 1 else 0.0
    se0 = float(diff0.std(ddof=1) / math.sqrt(p)) if p > 1 else 0.0
    h_val = j1 / j0 - 1.0 if abs(j0) > 1e-12 else float("nan")
    # Paired delta-method SE for the ratio statistic.
    if abs(j0) > 1e-12 and p > 1:
        grad = np.cov(diff1, diff0)
        var = (grad[0, 0] / j0**2 - 2.0 * j1 * grad[0, 1] / j0**3 + j1**2 * grad[1, 1] / j0**4) / p
        h_se = math.sqrt(max(var, 0.0))
    else:
        h_se = float("nan")
    return {
        "j_prior_probe": j1,
        "j_prior_probe_se": se1,
        "j_probe_only": j0,
        "j_probe_only_se": se0,
        "h": h_val,
        "h_se": h_se,
        "residual_displacement": float(
            (p10.displacement[:, d_end] - p00.displacement[:, d_end]).mean()
        ),
    }


def ewma_sigma(returns: Array, lam: float = 0.94) -> float:
    """EWMA sigma estimate composing ``models/volatility.ewma_variance``."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    var = ewma_variance(r, lam=lam)
    out = float(math.sqrt(float(var[-1])))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("ewma_sigma produced a non-positive estimate")
    return out


def almgren_sqrt_reference(quantity: float, daily_volume: float, sigma: float) -> float:
    """Power-law impact reference delta via ``execution/impact`` (composition)."""
    return pow_law_total_impact(quantity, daily_volume, sigma, exponent=0.5)


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def bench_langevin_impact(
    *,
    seed: int = 20260930,
    n_paths: int = 384,
    dt: float = 0.01,
) -> dict[str, float]:
    """Seeded SYNTHETIC bench returning a flat ``dict[str, float]``.

    Exercises: fresh-pool closed-form agreement, Table-3-scale GLE impact,
    mid-regime local exponent, vol proportionality, duration invariance,
    depletion band narrowing, round-trip nonnegativity, and the Q_0 latent
    displacement targeting of Table 1. All outputs are model-correctness
    diagnostics, never market evidence.
    """
    t0 = time.perf_counter()
    out: dict[str, float] = {}
    cfg = baseline_config()

    # 1) Fresh-pool deterministic impact vs Table 3 and the Eq. 45 form.
    times1 = np.arange(0.0, 1.0 + 0.5 * dt, dt)
    fresh = fresh_pool_config()
    omega = quadratic_onset_coefficient(cfg.counterflow_q_star, cfg.counterflow_scale)
    rate = constant_rate_schedule(1.0, 1.0)
    pf = simulate_paths(
        replace(fresh, counterflow_kind="quadratic"),
        rate,
        times1,
        n_paths=1,
        exec_horizon=1.0,
        seed=seed,
    )
    fresh_imp = terminal_impact(pf)[0]
    cf_form = fresh_pool_impact_closed_form(1.0, 1.0, omega, cfg.depth)
    out["synthetic_fresh_pool_impact_t1"] = fresh_imp
    out["synthetic_fresh_pool_closed_form_t1"] = cf_form
    out["synthetic_fresh_pool_abs_err_t1"] = abs(fresh_imp - cf_form)

    # 2) GLE-pool stochastic impact at Q0, T=tau0 (Table 3 pin ~0.158).
    pg = simulate_paths(cfg, rate, times1, n_paths=n_paths, exec_horizon=1.0, seed=seed + 1)
    gle_imp, gle_se = terminal_impact(pg)
    out["synthetic_gle_impact_t1"] = gle_imp
    out["synthetic_gle_impact_se_t1"] = gle_se

    # 3) Mid-regime local exponent from a coarse size grid (fresh pool).
    sizes = np.logspace(-3, 3, 25)
    curve = impact_curve(
        replace(fresh, counterflow_kind="quadratic"),
        sizes,
        1.0,
        n_paths=1,
        dt=dt,
        seed=seed + 2,
    )
    exps = curve["exponent"]
    mid = int(np.argmin(np.abs(np.log10(sizes))))
    out["synthetic_local_exponent_at_q0"] = float(exps[mid])
    out["synthetic_min_exponent"] = float(np.nanmin(exps))

    # 4) Vol proportionality in the sqrt regime (scaled thresholds).
    cfg_vol = fresh_pool_config(
        counterflow_kind="quadratic", threshold_scaling="vol", counterflow_scale=1.0
    )
    imps = []
    for s_x in (1.0, 2.0):
        pv = simulate_paths(
            replace(cfg_vol, sigma_x=s_x),
            rate,
            times1,
            n_paths=1,
            exec_horizon=1.0,
            seed=seed + 3,
        )
        imps.append(terminal_impact(pv)[0])
    out["synthetic_vol_scaling_ratio"] = imps[1] / imps[0]

    # 5) Duration invariance under duration-scaled thresholds (Eq. 52).
    cfg_dur = replace(cfg_vol, threshold_scaling="duration")
    dur_imps = []
    for t_exec in (1.0, 4.0):
        times_d = np.arange(0.0, t_exec + 0.5 * dt, dt)
        pv = simulate_paths(
            cfg_dur,
            constant_rate_schedule(1.0, t_exec),
            times_d,
            n_paths=1,
            exec_horizon=t_exec,
            seed=seed + 4,
        )
        dur_imps.append(terminal_impact(pv)[0])
    out["synthetic_duration_invariance_ratio"] = dur_imps[1] / dur_imps[0]

    # 6) Depletion narrowing: sqrt-band widths at T = tau0 (Table 4 scale).
    sizes_band = np.logspace(-4, 4, 33)
    band_fresh = sqrt_band(
        sizes_band,
        impact_curve(fresh, sizes_band, 1.0, n_paths=1, dt=dt, seed=seed + 5)["impact"],
    )
    band_gle = sqrt_band(
        sizes_band,
        impact_curve(cfg, sizes_band, 1.0, n_paths=n_paths, dt=dt, seed=seed + 6)["impact"],
    )
    out["synthetic_fresh_band_width_decades"] = band_fresh["width_decades"]
    out["synthetic_gle_band_width_decades"] = band_gle["width_decades"]
    out["synthetic_depletion_narrows_band"] = float(
        band_gle["width_decades"] < band_fresh["width_decades"]
    )

    # 7) Round-trip nonnegativity across kernel spectra (Prop. 3).
    rt_min = float("inf")
    for cfg_k in (cfg, single_mode_config(), broad_spectrum_config()):
        times_rt = np.arange(0.0, 2.0 + 0.5 * dt, dt)
        # Buy Q over [0,1], sell Q over [1,2]: net zero position.
        rate_rt = piecewise_rate_schedule(np.array([1.0, -1.0]), np.array([0.0, 1.0, 2.0]))
        prt = simulate_paths(
            cfg_k, rate_rt, times_rt, n_paths=n_paths, exec_horizon=2.0, seed=seed + 7
        )
        rt_min = min(rt_min, round_trip_cost(prt)["rhs_total"])
    out["synthetic_round_trip_min_cost"] = rt_min

    # 8) Table-1 targeting: noise-free unit order moves Y by ~1.01.
    quiet = replace(cfg, sigma_y=0.0)
    pq = simulate_paths(quiet, rate, times1, n_paths=1, exec_horizon=1.0, seed=seed + 8)
    out["synthetic_latent_displacement_q0"] = float(pq.latent[0, -1])

    out["synthetic_almgren_sqrt_reference_q1"] = almgren_sqrt_reference(1.0, 1.0, 1.0)
    out["runtime_seconds"] = time.perf_counter() - t0
    return out
