"""Passive-impact optimal execution (limit orders with exponential fills).

Implements the mesoscopic passive-impact execution model of Barzykin, Boyce,
Neuman & Tuschmann (2026): a limit order posted at distance ``delta`` ticks
from the midprice fills with Poisson intensity ``Lambda(delta) = lam *
exp(-k * delta)`` while passive quoting pressure permanently moves the
midprice at rate ``eta * exp(-m * delta)``.  The price-response ingredient is
the Cont-Kukanov-Stoikov order-flow-imbalance (OFI) linear relation
``dS = beta * OFI + eps``; combining it with exponential fill decay yields
the reduced-form passive impact rate ``eta e^{-m delta}`` with
``eta = xi * lam``, ``m = k + ell`` (paper Sec. 2).  The trader liquidates
``q0`` inventory units over ``[0, T]`` by choosing the quote distance
``delta_t`` — aggressive quotes fill faster but accumulate impact; passive
quotes earn the spread but risk non-execution.

Solved problems (paper Secs. 3 and 6):

- ``m == k``: closed form.  Theorem 3.1 gives
  ``delta*(t, q) = 1/k + (1/k) log(omega(t,q) / omega(t,q-1)) +
  (eta/lam) q`` where ``omega(t, .) = exp(-(T-t) A) b`` for the upper
  bidiagonal generator ``A`` (paper eq. 3.12) with diagonal
  ``q^2 k phi`` and superdiagonal ``-e^{-1} lam exp(-k (eta/lam) q)``, and
  terminal vector ``b_q = exp(-k alpha q^2)``.  We evaluate ``omega``
  literally via ``scipy.linalg.expm`` on the bidiagonal generator — the
  naive exponential-sum expansion of the Volterra recursion (Remark 3.14,
  ``W[q,j] = c_q W[q-1,j] / (a_q - a_j)``) is exponentially ill-
  conditioned for small ``k phi`` (its coefficients grow like
  ``prod 1/(a_q - a_j)`` while the sum stays O(1)), so the scaled-and-
  squared matrix exponential is the numerically honest evaluation.
- ``m != k``: semi-explicit form.  Theorem 6.1 gives
  ``delta*(t,q) = 1/k + dtheta + (1/(m-k)) Phi_q(dtheta)`` with
  ``dtheta = theta(t,q) - theta(t,q-1)``,
  ``Phi_q(z) = W_0(((m-k)/k)(eta/lam) q m exp(-((m-k)/k)(1 + k z)))``,
  and ``theta`` solving the triangular backward ODE (6.6), which we
  integrate on a uniform grid with fixed-step RK4 after the time reversal
  ``tau = T - t``.  The paper assumes ``m > k`` or ``k - eps < m < k``;
  we fail closed when the Lambert-W0 argument leaves ``[-e^{-1}, inf)``.
- Aggressive fallback: residual inventory is crossed as a market order
  priced through ``execution.impact`` (sqrt temporary + linear permanent
  impact) — we do not reimplement those primitives.

Honesty: this is a *model* of execution costs, not market evidence.  All
Monte-Carlo diagnostics are SYNTHETIC correctness checks over seeded paths
and every returned metric is namespaced ``sim_internal_*``; nothing here is
a Sharpe/P&L headline and no live-trading capability exists.  Deterministic:
all randomness flows through ``numpy.random.Generator`` and identical seeds
produce bit-identical paths.

References:
- Barzykin, Boyce, Neuman & Tuschmann (2026). "Optimal Execution with
  Passive Market Impact". arXiv:2607.28323 [q-fin.TR].  (Spec note: the lane
  brief cited "Barzykin, Bergault & coauthors"; the actual author list is
  Barzykin, Boyce, Neuman, Tuschmann — verified against the arXiv page.)
- Avellaneda & Stoikov (2008). "High-frequency trading in a limit order
  book". Quantitative Finance 8(3) — exponential fill intensity.
- Cont, Kukanov & Stoikov (2014). "The price impact of order book events".
  J. Financial Econometrics 12(1) — linear OFI price response.
- Xu, Gould & Howison (2019). "Multi-level order-flow imbalance".
  Market Microstructure and Liquidity — MLOFI decay behind beta(delta).
- Guéant, Lehalle & Fernandez-Tapia (2012). "Optimal portfolio liquidation
  with limit orders". — the eta = 0 reduction (paper Remark 3.3).
- Almgren & Chriss (2001). "Optimal execution of portfolio transactions".
  J. Risk — the aggressive-only benchmark via ``almgren_chriss.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.special import lambertw

from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory
from quant_fund.execution.impact import permanent_impact, sqrt_impact_bps

Array = NDArray[np.float64]

_LAMBERT_DOMAIN_MIN = -math.exp(-1.0)
_W_SERIES_TOL = 1e-12


def _finite_scalar(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _pos_scalar(x: float, name: str) -> float:
    v = _finite_scalar(x, name)
    if v <= 0.0:
        raise ValueError(f"{name} must be positive")
    return v


def _nonneg_scalar(x: float, name: str) -> float:
    v = _finite_scalar(x, name)
    if v < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return v


def _finite_vector(x: Array, name: str, min_len: int = 1) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < min_len or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite vector of length >= {min_len}")
    return v


def _check_int(value: int, name: str, lo: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(value)


@dataclass(frozen=True)
class PassiveImpactSpec:
    """Calibrated parameters of the Barzykin et al. (2026) execution model.

    Units: ``delta`` (quote distance) is measured in ticks throughout; ``k``
    and ``m`` are per-tick exponential decays; ``lam`` is the base fill
    intensity in inventory units per second at ``delta = 0``; ``eta`` is the
    passive impact rate in price units per second at ``delta = 0``;
    ``sigma`` is midprice volatility in price units per sqrt-second;
    ``phi``/``alpha`` are the running/terminal inventory penalties of the
    objective (3.7); ``beta_ofi`` is the Cont et al. linear OFI price
    response in price units per imbalance unit; ``tick_size`` converts
    quoted ticks to price units.
    """

    lam: float
    k: float
    eta: float = 0.0
    m: float | None = None
    sigma: float = 0.0
    phi: float = 0.0
    alpha: float = 0.0
    beta_ofi: float = 0.0
    tick_size: float = 0.01
    resilience: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lam", _pos_scalar(self.lam, "lam"))
        object.__setattr__(self, "k", _pos_scalar(self.k, "k"))
        object.__setattr__(self, "eta", _nonneg_scalar(self.eta, "eta"))
        m = self.k if self.m is None else _nonneg_scalar(self.m, "m")
        object.__setattr__(self, "m", m)
        if m != self.k and self.eta > 0.0 and m <= 0.0:
            raise ValueError("m must be positive when it differs from k and eta > 0")
        object.__setattr__(self, "sigma", _nonneg_scalar(self.sigma, "sigma"))
        object.__setattr__(self, "phi", _nonneg_scalar(self.phi, "phi"))
        object.__setattr__(self, "alpha", _nonneg_scalar(self.alpha, "alpha"))
        object.__setattr__(self, "beta_ofi", _nonneg_scalar(self.beta_ofi, "beta_ofi"))
        object.__setattr__(self, "tick_size", _pos_scalar(self.tick_size, "tick_size"))
        object.__setattr__(self, "resilience", _nonneg_scalar(self.resilience, "resilience"))

    @property
    def m_eff(self) -> float:
        """Impact decay per tick; defaults to ``k`` when ``m`` is unset."""
        return self.k if self.m is None else float(self.m)

    @property
    def equal_decay(self) -> bool:
        """True iff the fill-intensity and impact decays coincide (m == k)."""
        return self.m_eff == self.k


# ---------------------------------------------------------------------------
# Fill model (paper Sec. 2): Poisson fills, exponential depth decay.
# ---------------------------------------------------------------------------


def fill_intensity(delta_ticks: Array | float, lam: float, k: float) -> Array:
    """Poisson fill rate ``lam * exp(-k * delta)`` in fills per second.

    ``delta`` may be negative (marketable/aggressive limit orders, paper
    Sec. 3 admissible controls).  Vectorized over ``delta``.
    """
    lam = _pos_scalar(lam, "lam")
    k = _pos_scalar(k, "k")
    d = np.asarray(delta_ticks, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("delta_ticks must be finite")
    return lam * np.exp(-k * d)


def fill_probability(delta_ticks: Array | float, dt: float, lam: float, k: float) -> Array:
    """Probability of >= 1 fill in ``[t, t+dt]``: ``1 - exp(-Lambda dt)``."""
    dt = _pos_scalar(dt, "dt")
    lam_eff = fill_intensity(delta_ticks, lam, k)
    return -np.expm1(-lam_eff * dt)


def expected_fill_time(delta_ticks: Array | float, lam: float, k: float) -> Array:
    """Mean waiting time to fill ``lam^{-1} exp(k delta)`` seconds (eq. 2.5)."""
    lam = _pos_scalar(lam, "lam")
    k = _pos_scalar(k, "k")
    d = np.asarray(delta_ticks, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("delta_ticks must be finite")
    return np.exp(k * d) / lam


def repost_attempts(delta_ticks: Array | float, tau: float, lam: float, k: float) -> Array:
    """Expected quote submissions per fill ``1 / p_delta`` (paper eq. 2.9).

    ``tau`` is the active lifetime of each posted child order; the
    continuous-time limit ``tau -> 0`` recovers ``expected_fill_time``.
    """
    tau = _pos_scalar(tau, "tau")
    p = fill_probability(delta_ticks, tau, lam, k)
    if np.any(p <= 0.0):
        raise ValueError("fill probability underflowed to zero; deepen check on delta")
    return 1.0 / p


def passive_impact_rate(delta_ticks: Array | float, eta: float, m: float) -> Array:
    """Passive impact drift ``eta * exp(-m * delta)`` in price units / second."""
    eta = _nonneg_scalar(eta, "eta")
    m = _nonneg_scalar(m, "m")
    d = np.asarray(delta_ticks, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("delta_ticks must be finite")
    return eta * np.exp(-m * d)


# ---------------------------------------------------------------------------
# Order flow imbalance (paper eq. 2.1-2.3, Cont et al. 2014).
# ---------------------------------------------------------------------------


def order_flow_imbalance(
    limit_bid: Array | float,
    cancel_bid: Array | float,
    market_ask: Array | float,
    limit_ask: Array | float,
    cancel_ask: Array | float,
    market_bid: Array | float,
) -> Array:
    """OFI = L^b - C^b - M^a - L^a + C^a + M^b over an interval.

    Positive OFI (net bid-side pressure) pushes the midprice up.  All six
    inputs are non-negative sizes on the best quotes; broadcasting allowed.
    """
    names = ("limit_bid", "cancel_bid", "market_ask", "limit_ask", "cancel_ask", "market_bid")
    vals = [
        np.asarray(v, dtype=float)
        for v in (limit_bid, cancel_bid, market_ask, limit_ask, cancel_ask, market_bid)
    ]
    for name, v in zip(names, vals, strict=True):
        if not np.all(np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    return np.asarray(vals[0] - vals[1] - vals[2] - vals[3] + vals[4] + vals[5])


def ofi_price_response(ofi: Array | float, beta: float) -> Array:
    """Linear OFI price response ``beta * OFI`` (paper eq. 2.2, eps exogenous)."""
    beta = _nonneg_scalar(beta, "beta")
    o = np.asarray(ofi, dtype=float)
    if not np.all(np.isfinite(o)):
        raise ValueError("ofi must be finite")
    return beta * o


# ---------------------------------------------------------------------------
# omega weights for the m == k closed form (paper Thm. 3.1, eqs. 3.11-3.12).
# ---------------------------------------------------------------------------


def _omega_generator(spec: PassiveImpactSpec, q0: int) -> Array:
    """Bidiagonal generator ``A`` of eq. 3.12, rows ordered q0, ..., 1, 0."""
    q0 = _check_int(q0, "q0", 1)
    q_desc = np.arange(q0, -1, -1, dtype=np.float64)
    a = np.zeros((q0 + 1, q0 + 1))
    a[np.arange(q0 + 1), np.arange(q0 + 1)] = spec.k * spec.phi * q_desc**2
    sup = -(math.exp(-1.0) * spec.lam) * np.exp(-spec.k * (spec.eta / spec.lam) * q_desc[:-1])
    a[np.arange(q0), np.arange(1, q0 + 1)] = sup
    return a


def _omega_terminal(spec: PassiveImpactSpec, q0: int) -> Array:
    """Terminal vector ``b_q = exp(-k alpha q^2)``, ordered q0, ..., 1, 0."""
    q_desc = np.arange(q0, -1, -1, dtype=np.float64)
    return np.exp(-spec.k * spec.alpha * q_desc**2)


def _omega_weights(spec: PassiveImpactSpec, q0: int, tau: float) -> Array:
    """``omega(t, q)`` for ``q = 0..q0`` at remaining time ``tau = T - t``."""
    tau = _nonneg_scalar(tau, "tau")
    w_desc = expm(-tau * _omega_generator(spec, q0)) @ _omega_terminal(spec, q0)
    w = w_desc[::-1]
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("omega weights non-positive; check spec parameters")
    return np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# theta ODE for the m != k case (paper Thm. 6.1, eqs. 6.4-6.7).
# ---------------------------------------------------------------------------


def _phi_q(z: float, spec: PassiveImpactSpec, q: int) -> float:
    """Phi_q(z) = W_0(((m-k)/k)(eta/lam) q m exp(-((m-k)/k)(1 + k z)))."""
    m, k, eta, lam = spec.m_eff, spec.k, spec.eta, spec.lam
    dmk = m - k
    arg = (dmk / k) * (eta / lam) * q * m * math.exp(-(dmk / k) * (1.0 + k * z))
    if arg < _LAMBERT_DOMAIN_MIN - 1e-12:
        raise ValueError(
            f"Lambert W0 argument {arg} below -1/e at q={q}, z={z}; "
            "paper Thm. 6.1 requires m > k or k - eps < m < k"
        )
    arg = max(arg, _LAMBERT_DOMAIN_MIN)
    return float(np.real(lambertw(arg, 0)))


def _phi_over_dmk(z: float, spec: PassiveImpactSpec, q: int) -> float:
    """Phi_q(z) / (m - k) with the first-order series fallback near m == k."""
    m, k, eta, lam = spec.m_eff, spec.k, spec.eta, spec.lam
    dmk = m - k
    arg = (dmk / k) * (eta / lam) * q * m * math.exp(-(dmk / k) * (1.0 + k * z))
    if abs(arg) < _W_SERIES_TOL:
        # W_0(arg) ~ arg; arg/(m-k) is finite even as m -> k.
        return (eta / lam) * q * (m / k) * math.exp(-(dmk / k) * (1.0 + k * z))
    return _phi_q(z, spec, q) / dmk


def _prev_at(prev: Array, frac_index: float) -> float:
    """Linear interpolation of the solved theta_{q-1} row on its grid."""
    lo = int(math.floor(frac_index))
    hi = min(lo + 1, prev.size - 1)
    w = frac_index - lo
    return float(prev[lo] * (1.0 - w) + prev[hi] * w)


def _theta_rhs(theta_q: float, theta_prev: float, spec: PassiveImpactSpec, q: int) -> float:
    """dTheta/dtau for reversed time tau = T - t (paper eq. 6.6)."""
    m, k, lam = spec.m_eff, spec.k, spec.lam
    dz = theta_q - theta_prev
    phi_term = _phi_q(dz, spec, q)
    phi_over = _phi_over_dmk(dz, spec, q)
    drift = -spec.phi * q * q
    jump = lam * (1.0 / k + phi_term / m) * math.exp(-1.0 - k * dz - k * phi_over)
    return drift + jump


def _theta_grid(
    spec: PassiveImpactSpec, q0: int, horizon: float, n_grid: int
) -> tuple[Array, Array]:
    """Backward ODE (6.6)-(6.7): uniform-grid RK4 in reversed time."""
    q0 = _check_int(q0, "q0", 1)
    horizon = _pos_scalar(horizon, "horizon")
    n_grid = _check_int(n_grid, "n_grid", 2)
    taus = np.linspace(0.0, horizon, n_grid + 1)
    h = float(taus[1] - taus[0])
    theta = np.empty((q0 + 1, n_grid + 1))
    theta[0, :] = 0.0
    idx = np.arange(q0 + 1, dtype=np.float64)
    theta[:, 0] = -spec.alpha * idx**2
    for q in range(1, q0 + 1):
        prev = theta[q - 1]
        for i in range(n_grid):
            y0 = float(theta[q, i])
            k1 = _theta_rhs(y0, _prev_at(prev, float(i)), spec, q)
            k2 = _theta_rhs(y0 + 0.5 * h * k1, _prev_at(prev, i + 0.5), spec, q)
            k3 = _theta_rhs(y0 + 0.5 * h * k2, _prev_at(prev, i + 0.5), spec, q)
            k4 = _theta_rhs(y0 + h * k3, _prev_at(prev, float(i + 1)), spec, q)
            theta[q, i + 1] = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(theta[q])):
            raise ValueError(f"theta ODE diverged at q={q}; check spec parameters")
    return taus, theta


# ---------------------------------------------------------------------------
# Optimal plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassiveExecutionPlan:
    """Optimal posting schedule ``delta*(t, q)`` on a uniform time grid.

    ``times`` has ``n_grid + 1`` points covering ``[0, horizon]``;
    ``quotes[i, q]`` is the optimal distance (ticks, mid-relative: a sell is
    posted at ``S_t + delta * tick_size``) when ``q`` inventory units remain.
    ``quotes[:, 0]`` is NaN — quoting stops at full liquidation (paper
    eq. 3.3).  ``theta_log_weights`` carries ``(1/k) log omega`` for the
    equal-decay closed form or the integrated ``theta`` for ``m != k``.
    """

    times: Array
    quotes: Array
    theta_log_weights: Array
    n_units: int
    horizon: float
    method: str
    spec: PassiveImpactSpec

    def quote_at(self, t: float, q: int) -> float:
        """Nearest-grid optimal quote at time ``t`` with ``q`` units left."""
        q = _check_int(q, "q", 1)
        if q > self.n_units:
            raise ValueError("q exceeds plan inventory")
        tt = _nonneg_scalar(t, "t")
        i = int(round(min(tt, self.horizon) / self.horizon * (self.times.size - 1)))
        return float(self.quotes[i, q])


def optimal_execution_plan(
    spec: PassiveImpactSpec,
    n_units: int,
    horizon: float,
    n_grid: int = 200,
) -> PassiveExecutionPlan:
    """Optimal quote schedule ``delta*(t, q)`` for ``q0`` inventory units.

    ``m == k`` uses the exact closed form (Thm. 3.1); ``m != k`` integrates
    the triangular Lambert-W ODE (Thm. 6.1) on the same grid.  ``n_grid`` is
    both the plan's time resolution and the ODE step count (RK4: global
    error O(h^4), deterministic).
    """
    q0 = _check_int(n_units, "n_units", 1)
    horizon = _pos_scalar(horizon, "horizon")
    n_grid = _check_int(n_grid, "n_grid", 8)
    times = np.linspace(0.0, horizon, n_grid + 1)
    quotes = np.full((n_grid + 1, q0 + 1), np.nan)
    if spec.equal_decay:
        logw = np.empty((n_grid + 1, q0 + 1))
        for i, t in enumerate(times):
            logw[i] = np.log(_omega_weights(spec, q0, horizon - float(t)))
        theta = logw / spec.k
        for q in range(1, q0 + 1):
            quotes[:, q] = (
                1.0 / spec.k + (theta[:, q] - theta[:, q - 1]) + (spec.eta / spec.lam) * q
            )
        return PassiveExecutionPlan(
            times=times,
            quotes=quotes,
            theta_log_weights=theta,
            n_units=q0,
            horizon=horizon,
            method="closed_form_m_eq_k",
            spec=spec,
        )
    _taus, theta_rev = _theta_grid(spec, q0, horizon, n_grid)
    # theta(t, q) = Theta(T - t, q): the tau-grid is uniform and symmetric,
    # so the reversal is an exact axis flip.
    theta = theta_rev[:, ::-1]
    for q in range(1, q0 + 1):
        for i in range(n_grid + 1):
            dz = float(theta[q, i] - theta[q - 1, i])
            quotes[i, q] = 1.0 / spec.k + dz + _phi_over_dmk(dz, spec, q)
    return PassiveExecutionPlan(
        times=times,
        quotes=quotes,
        theta_log_weights=theta,
        n_units=q0,
        horizon=horizon,
        method="lambert_ode_m_neq_k",
        spec=spec,
    )


def value_function_quote(
    t: float, q: int, spec: PassiveImpactSpec, n_units: int, horizon: float
) -> float:
    """Single-point optimal quote ``delta*(t, q)`` (equal-decay path only).

    Convenience wrapper for the closed form; for ``m != k`` build a plan
    with ``optimal_execution_plan`` instead.
    """
    if not spec.equal_decay:
        raise ValueError("value_function_quote requires m == k; use optimal_execution_plan")
    q0 = _check_int(n_units, "n_units", 1)
    q = _check_int(q, "q", 1)
    if q > q0:
        raise ValueError("q exceeds n_units")
    horizon = _pos_scalar(horizon, "horizon")
    t = _nonneg_scalar(t, "t")
    w = _omega_weights(spec, q0, horizon - min(t, horizon))
    dlog = math.log(w[q] / w[q - 1])
    return 1.0 / spec.k + dlog / spec.k + (spec.eta / spec.lam) * q


# ---------------------------------------------------------------------------
# Fluid inventory relaxation + diagnostics.
# ---------------------------------------------------------------------------


def expected_inventory_path(plan: PassiveExecutionPlan) -> Array:
    """Fluid relaxation of the fill process under the optimal schedule.

    Deterministic ODE ``dQ/dt = -Lambda(delta*(t, ceil(Q)))`` integrated by
    explicit Euler on the plan grid — the deterministic skeleton of the
    counting-process dynamics (3.3); the discrete simulator below adds the
    Poisson jumps.  Returns expected units remaining per grid point.
    """
    spec = plan.spec
    times = plan.times
    q = np.empty(times.size)
    q[0] = float(plan.n_units)
    for i in range(times.size - 1):
        dt = float(times[i + 1] - times[i])
        qi = int(math.ceil(q[i] - 1e-12))
        if qi <= 0:
            q[i + 1] = 0.0
            continue
        d = float(plan.quotes[i, qi])
        rate = float(fill_intensity(d, spec.lam, spec.k))
        q[i + 1] = max(0.0, q[i] - rate * dt)
    return q


def fill_probability_curve(depths: Array, dt: float, spec: PassiveImpactSpec) -> Array:
    """Expected fill probability per posted depth over window ``dt``.

    The diagnostic depth-vs-fill trade-off curve: deeper quotes earn more
    spread conditional on fill but fill exponentially less often.
    """
    d = _finite_vector(np.asarray(depths, dtype=float), "depths")
    return fill_probability(d, dt, spec.lam, spec.k)


# ---------------------------------------------------------------------------
# Aggressive fallback — priced via execution.impact (composed, not dup'd).
# ---------------------------------------------------------------------------


def urgency_kappa(spec: PassiveImpactSpec, eta_temp: float) -> float:
    """Almgren-Chriss urgency ``kappa = sqrt(lambda_risk sigma^2 / eta)``.

    Maps the running inventory penalty ``phi`` onto AC risk aversion, so the
    aggressive-only benchmark shares the spec's risk dial.  ``eta_temp`` is
    the AC temporary-impact coefficient in matching units.
    """
    eta_temp = _pos_scalar(eta_temp, "eta_temp")
    return math.sqrt(spec.phi * spec.sigma**2 / eta_temp)


def aggressive_schedule(
    quantity: float,
    n_slices: int,
    spec: PassiveImpactSpec,
    *,
    eta_temp: float,
    gamma_perm: float,
    tau: float = 1.0,
    risk_aversion: float | None = None,
) -> Array:
    """Aggressive-only holdings benchmark — delegates to ``almgren_chriss``.

    The ``kappa -> infinity`` reduction: as the mapped risk aversion
    ``phi -> inf`` the AC trajectory degenerates to immediate liquidation,
    which is exactly the passive model's ``delta* -> -inf`` (fill intensity
    ``-> inf``) regime — verified in the test-suite to ~1e-8.
    """
    ra = spec.phi if risk_aversion is None else _nonneg_scalar(risk_aversion, "risk_aversion")
    return almgren_chriss_trajectory(
        quantity,
        n_slices,
        sigma=spec.sigma,
        eta=eta_temp,
        gamma=gamma_perm,
        risk_aversion=ra,
        tau=tau,
    )


def aggressive_fill_price(
    mid_price: float,
    quantity: float,
    daily_volume: float,
    sigma_daily: float,
    spec: PassiveImpactSpec,
    *,
    upsilon: float = 0.6,
    gamma_perm: float = 0.5,
    half_spread_bps: float = 0.0,
    side: int = -1,
) -> float:
    """Expected execution price of crossing the spread (market order).

    Sell side (``side = -1``) by default, matching the liquidation
    direction of the paper; a buy flips the sign.  Price impact components
    come from ``execution.impact`` — sqrt temporary + linear permanent —
    plus an optional half-spread haircut, all as return fractions.
    """
    mid = _pos_scalar(mid_price, "mid_price")
    qty = _pos_scalar(quantity, "quantity")
    dv = _pos_scalar(daily_volume, "daily_volume")
    sd = _nonneg_scalar(sigma_daily, "sigma_daily")
    hs = _nonneg_scalar(half_spread_bps, "half_spread_bps")
    if side not in (-1, 1):
        raise ValueError("side must be -1 (sell) or +1 (buy)")
    temp = sqrt_impact_bps(qty / dv, sd, upsilon) / 1e4
    perm = permanent_impact(qty, dv, sd, gamma_perm)
    haircut = temp + perm + hs / 1e4
    return float(mid * (1.0 + side * haircut))


# ---------------------------------------------------------------------------
# Seeded Monte-Carlo simulator (SYNTHETIC diagnostics only).
# ---------------------------------------------------------------------------


def _metric(report: dict[str, object], key: str) -> float:
    v = report.get(key)
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        raise ValueError(f"{key} missing or non-numeric in simulation report")
    return float(v)


def _as_rng(seed: np.random.Generator | int) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        return np.random.default_rng(int(seed))
    raise ValueError("seed must be an int or numpy Generator")


def simulate_passive_execution(
    plan: PassiveExecutionPlan,
    *,
    mid_price: float,
    unit_size: float,
    n_paths: int,
    seed: np.random.Generator | int,
    ofi_shocks: Array | None = None,
    liquidate_residual: bool = True,
    daily_volume: float = 1.0e9,
    sigma_daily: float = 0.02,
    half_spread_bps: float = 0.5,
) -> dict[str, object]:
    """Monte-Carlo execution under the optimal schedule. SYNTHETIC only.

    Path dynamics (paper eqs. 3.2-3.6, forward Euler-Maruyama on the plan
    grid): the impacted midprice follows ``dS = sigma dW - eta e^{-m delta}
    1{Q>0} dt + beta * OFI`` (with ``resilience > 0`` the impact state
    decays at rate ``rho`` per eq. 6.8); fills arrive as ``Poisson(Lambda
    dt)`` truncated at remaining inventory; each fill sells ``unit_size``
    shares at ``S_t + delta * tick_size``.  Residual inventory at ``T`` is
    marked at the impacted midprice plus optional aggressive-liquidation
    pricing via ``aggressive_fill_price``.  All randomness is seeded —
    identical ``seed`` gives bit-identical output.  Metrics are namespaced
    ``sim_internal_*`` and labelled SYNTHETIC: correctness diagnostics,
    never market evidence.
    """
    spec = plan.spec
    s0 = _pos_scalar(mid_price, "mid_price")
    usz = _pos_scalar(unit_size, "unit_size")
    n_paths = _check_int(n_paths, "n_paths", 1)
    rng = _as_rng(seed)
    n = plan.times.size - 1
    dt = float(plan.horizon) / n
    shocks = np.zeros(n)
    if ofi_shocks is not None:
        shocks = _finite_vector(ofi_shocks, "ofi_shocks", min_len=n)
        if shocks.size != n:
            raise ValueError("ofi_shocks must match the plan grid length")
    z = rng.standard_normal((n_paths, n))
    s = np.empty((n_paths, n + 1))
    impact_state = np.zeros((n_paths, n + 1))
    s[:, 0] = s0
    cash = np.zeros(n_paths)
    qty_units = np.full(n_paths, float(plan.n_units))
    fills_units = np.zeros((n_paths, n + 1))
    trading_time = np.full(n_paths, np.inf)
    for i in range(n):
        qi = np.ceil(qty_units - 1e-12).astype(int)
        active = qi > 0
        delta = np.full(n_paths, np.nan)
        if np.any(active):
            delta[active] = plan.quotes[i, qi[active]]
        rate = np.where(active, spec.eta * np.exp(-spec.m_eff * np.nan_to_num(delta, nan=0.0)), 0.0)
        if spec.resilience > 0.0:
            impact_state[:, i + 1] = impact_state[:, i] * (1.0 - spec.resilience * dt) + rate * dt
            s[:, i + 1] = (
                s[:, i]
                + spec.sigma * z[:, i] * math.sqrt(dt)
                - (impact_state[:, i + 1] - impact_state[:, i])
                + spec.beta_ofi * shocks[i]
            )
        else:
            s[:, i + 1] = (
                s[:, i]
                + spec.sigma * z[:, i] * math.sqrt(dt)
                - rate * dt
                + spec.beta_ofi * shocks[i]
            )
        lam_eff = np.where(active, spec.lam * np.exp(-spec.k * np.nan_to_num(delta, nan=0.0)), 0.0)
        jumps = np.where(active, rng.poisson(lam_eff * dt), 0)
        jumps = np.minimum(jumps, np.maximum(qi, 0))
        fills_units[:, i] = jumps
        exec_price = s[:, i] + np.nan_to_num(delta, nan=0.0) * spec.tick_size
        cash += jumps * usz * exec_price
        qty_units -= jumps
        done = (qty_units <= 0.0) & np.isinf(trading_time)
        trading_time[done] = plan.times[i + 1]
    residual_units = np.maximum(qty_units, 0.0)
    resid_price = s[:, -1].copy()
    if liquidate_residual:
        resid_price = np.array(
            [
                aggressive_fill_price(
                    float(s[p, -1]),
                    float(residual_units[p] * usz),
                    daily_volume,
                    sigma_daily,
                    spec,
                    half_spread_bps=half_spread_bps,
                )
                if residual_units[p] > 0.0
                else float(s[p, -1])
                for p in range(n_paths)
            ]
        )
    terminal_value = cash + residual_units * usz * resid_price
    pnl = terminal_value - float(plan.n_units) * usz * s0
    is_cost = -pnl  # shortfall vs arrival mark for a sell program
    fill_frac = 1.0 - residual_units / float(plan.n_units)
    tt = np.where(np.isinf(trading_time), plan.horizon, trading_time)
    return {
        "label": "SYNTHETIC",
        "n_paths": n_paths,
        "sim_internal_fill_fraction_mean": float(np.mean(fill_frac)),
        "sim_internal_fill_fraction_p10": float(np.quantile(fill_frac, 0.1)),
        "sim_internal_fill_fraction_p90": float(np.quantile(fill_frac, 0.9)),
        "sim_internal_trading_time_mean": float(np.mean(tt)),
        "sim_internal_trading_time_p90": float(np.quantile(tt, 0.9)),
        "sim_internal_pnl_mean": float(np.mean(pnl)),
        "sim_internal_pnl_p10": float(np.quantile(pnl, 0.1)),
        "sim_internal_pnl_p90": float(np.quantile(pnl, 0.9)),
        "sim_internal_shortfall_mean": float(np.mean(is_cost)),
        "sim_internal_terminal_price_mean": float(np.mean(s[:, -1])),
        "sim_internal_total_fills_mean": float(np.mean(fills_units.sum(axis=1))),
    }


def compare_vs_aggressive_baseline(
    spec: PassiveImpactSpec,
    n_units: int,
    horizon: float,
    *,
    mid_price: float,
    unit_size: float,
    n_paths: int,
    seed: int,
    n_grid: int = 200,
    eta_temp: float = 1e-6,
    gamma_perm: float = 0.5,
    risk_aversion: float = 1e-4,
    daily_volume: float = 1.0e9,
    sigma_daily: float = 0.02,
    half_spread_bps: float = 0.5,
) -> dict[str, object]:
    """Passive plan vs aggressive AC schedule on identical seeded paths.

    SYNTHETIC diagnostic: identical Brownian paths and identical residual
    pricing isolate the schedule effect.  The aggressive baseline liquidates
    on the AC trajectory (mapped risk aversion) paying
    ``aggressive_fill_price`` on each slice; the passive arm runs
    ``simulate_passive_execution``.  Returned keys are ``sim_internal_*``.
    """
    q0 = _check_int(n_units, "n_units", 1)
    horizon = _pos_scalar(horizon, "horizon")
    n_grid = _check_int(n_grid, "n_grid", 8)
    n_paths = _check_int(n_paths, "n_paths", 1)
    s0 = _pos_scalar(mid_price, "mid_price")
    usz = _pos_scalar(unit_size, "unit_size")
    rng = np.random.default_rng(seed)
    plan = optimal_execution_plan(spec, q0, horizon, n_grid)
    passive = simulate_passive_execution(
        plan,
        mid_price=s0,
        unit_size=usz,
        n_paths=n_paths,
        seed=rng,
        daily_volume=daily_volume,
        sigma_daily=sigma_daily,
        half_spread_bps=half_spread_bps,
    )
    n = n_grid
    dt = horizon / n
    holdings = aggressive_schedule(
        q0 * usz,
        n,
        spec,
        eta_temp=eta_temp,
        gamma_perm=gamma_perm,
        tau=dt,
        risk_aversion=risk_aversion,
    )
    slice_qty = np.maximum(-np.diff(holdings), 0.0)
    # Identical-path baseline: same seeds -> same Brownian skeleton.
    rng_base = np.random.default_rng(seed)
    z = np.empty((n_paths, n))
    for p in range(n_paths):
        z[p] = rng_base.standard_normal(n)
    pnl_agg = np.zeros(n_paths)
    for p in range(n_paths):
        s = s0
        cash = 0.0
        for i in range(n):
            if slice_qty[i] <= 0.0:
                s += spec.sigma * z[p, i] * math.sqrt(dt)
                continue
            px = aggressive_fill_price(
                s,
                float(slice_qty[i]),
                daily_volume,
                sigma_daily,
                spec,
                half_spread_bps=half_spread_bps,
            )
            cash += float(slice_qty[i]) * px
            s += spec.sigma * z[p, i] * math.sqrt(dt)
        cash += holdings[-1] * s
        pnl_agg[p] = cash - q0 * usz * s0
    passive_pnl = _metric(passive, "sim_internal_pnl_mean")
    return {
        "label": "SYNTHETIC",
        "n_paths": n_paths,
        "method": plan.method,
        "sim_internal_passive_pnl_mean": passive_pnl,
        "sim_internal_aggressive_pnl_mean": float(np.mean(pnl_agg)),
        "sim_internal_passive_minus_aggressive_mean": passive_pnl - float(np.mean(pnl_agg)),
        "sim_internal_passive_fill_fraction_mean": _metric(
            passive, "sim_internal_fill_fraction_mean"
        ),
        "sim_internal_passive_trading_time_mean": _metric(
            passive, "sim_internal_trading_time_mean"
        ),
    }
