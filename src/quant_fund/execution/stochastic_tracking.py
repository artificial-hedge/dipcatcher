"""Stochastic tracking and regularized Obizhaeva--Wang execution. No Sharpe.

Implements the quadratic stochastic-tracking construction and the generalized
Obizhaeva--Wang (OW) optimal-execution model of

    Nutz, M., & Voss, M. (2026). "The Convergence Rate of Stochastic Tracking
    with Application to Optimal Execution." arXiv:2608.29468 [q-fin.TR]
    (29 Aug 2026). Citation verified against https://arxiv.org/abs/2608.29468
    and the full HTML text (fetched 2026-09-30).

Model (paper Sec. 4). On a uniform grid ``t_i = i h`` over ``[0, T]`` an
execution strategy is a cadlag cumulative-position path ``Q`` with
``Q_{0-} = 0`` and terminal constraint ``Q_T = Xi_T`` (possibly random). The
transient-impact process ``Y = Y^Q`` solves ``dY_t = -beta_t Y_t dt +
lam_t dQ_t``, ``Y_{0-} = y`` (eq. 4.3): exponential resilience ``beta``,
depth/push factor ``lam``, with ``gamma_dot_t = dot(lam)_t / lam_t``,
``eta_t = (beta_t + gamma_dot_t)/(2 beta_t + gamma_dot_t)`` and
``theta_t = int_0^t (beta_s + gamma_dot_s)^2 / (lam_s (2 beta_s + gamma_dot_s))
ds`` (eq. 4.1-4.2). Assumption 4.1 is enforced fail-closed: ``lam > 0`` and
``2 beta + gamma_dot`` bounded away from zero (the no-price-manipulation
condition). The execution objective is the Hilbert-space form of Lemma 4.5,

    J_0(Q) = E[ (1/2)(Y_T^2/lam_T - y^2/lam_0)
               + int_0^T (2 beta_t + gamma_dot_t)/(2 lam_t) Y_t^2 dt ]
           = <Y, Y>_H - y^2/(2 lam_0),                                  (4.15)

which is exact for any cadlag semimartingale ``Q``; on the grid the running
integral uses the closed-form per-cell integral ``int Y^2`` of the
exponentially decaying path (no quadrature error for piecewise-constant
impact between grid trades).

Unregularized optimum (Theorem 4.2). With terminal-target martingale
``Xi_t = E[Xi_T | F_t]``, ``alpha_t = eta_t/lam_t`` and

    M_0  = lam_T/(1 + lam_T theta_T) * (Xi_0 + y/lam_0),
    M_t  = M_0 + int_0^t lam_T/(1 + lam_T (theta_T - theta_s)) dXi_s,   (4.6-4.8)

the optimal impact is ``Y*_t = eta_t M_t`` on ``[0, T)``, ``Y*_T = M_T``
(eq. 4.9), and the optimal strategy is

    Q^0_t = alpha_t M_t - y/lam_0 + int_0^t theta_dot_s M_s ds,  t < T, (5.17)

completed by the terminal block ``Q^0_T = Xi_T`` (eq. 4.10-4.11). The optimal
cost is ``V(0) = (1/2)(E[int theta_dot_t M_t^2 dt] + E[M_T^2]/lam_T -
y^2/lam_0)`` (eq. 4.12). For constant ``beta, lam`` and deterministic
``Xi_T`` with ``y = 0`` this reduces to the classical OW schedule
``dQ^0 = Xi_T/D (delta_0 + delta_T) + beta Xi_T/D dt`` with ``D = beta T + 2``
and ``V(0) = lam Xi_T^2 / D`` (Prop. 5.9, eq. 5.35) -- verified here against
the repo's ``almgren_chriss`` TWAP limit (``beta -> inf`` recovers TWAP).

Regularization (Sec. 5). Restricting to absolutely continuous strategies
``Q_t = int_0^t u_s ds`` adds a quadratic trading-rate penalty
``J_eps(Q) = J_0(Q) + eps E int u_t^2 dt`` (eq. 5.2). The paper's implementable
near-optimizer ``Q^eps`` (Lemma 5.5, eqs. 5.18-5.20) exponentially filters
``Q^0`` at relaxation scale ``a = sqrt(eps)`` up to ``s = T - a``, then runs a
terminal bridge ``dot Q_t = (Xi_t - Q_t)/(T - t)`` enforcing ``Q_T = Xi_T``:

    Q^eps_t = (1/a) int_0^t e^{-(t-r)/a} Q^0_r dr,            t <= T - a,
    Q^eps_t = (T-t) [ Qbar_s/a + int_s^t Xi_r/(T-r)^2 dr ],   T - a < t < T.

``Q^eps`` needs no solver; Theorem 5.7 bounds both the excess impact cost
``J_0(Q^eps) - J_0(Q^0)`` and the value gap ``V(eps) - V(0)`` by
``O(sqrt(eps))``, and Proposition 5.9 proves the rate sharp in the constant
benchmark ``y = 0``, ``Xi_T`` deterministic:

    V(eps) = lam Xi_T^2/D + 2 sqrt(beta lam) Xi_T^2/D^2 * sqrt(eps) + O(eps).

The regularized optimizer in that benchmark is the closed form of Chen, Horst
& Tran (2024, ref. [12], Thm 2.1; reproduced at eqs. 5.36-5.40):
``k_eps = sqrt(beta^2 + beta lam/eps)``, ``coth``-stabilized here for
``eps`` down to ~1e-9,

    Q^{*,eps}_t = Xi_T (a_e + b_e t + c_e sinh(k_e (t - T/2))) / (2 a_e + b_e T),
    V(eps)  = eps k_e^2 / beta^2 * d_e * Xi_T,  d_e = Xi_T b_e/(2 a_e + b_e T).

Stochastic-target extension: the paper's ``Xi_t`` is the conditional-
expectation martingale of the terminal target; this module accepts arbitrary
cadlag martingale paths of ``Xi`` (the input contract -- the martingale
property itself cannot be validated from one path family) and evaluates
``M_t``, ``Q^0``, ``Q^eps`` pathwise, so expectations are path means. This is
the documented simplification where the paper is PDE/Hilbert-heavy: the grid
discretizes Ito-weight ``lam_T/(1 + lam_T (theta_T - theta_s))`` at the
increment's landing point and the drift ``int theta_dot M`` at cell
resolution; both are exact for piecewise-constant paths. A Besov-side
diagnostic, ``translation_modulus``/``tr_seminorm``, implements the squared
L2 time-translation modulus ``omega(h)`` and seminorm ``[.]_tr`` of
eqs. (2.6)-(2.7)/(5.14)-(5.15) used by the tracking bounds.

Honesty: every quantity produced here is an execution-model cost functional
of the paper's own objective (impact cost ``J_0``, rate penalty, tracking
L2, time-translation modulus) or a seeded SYNTHETIC Monte-Carlo statistic
(keys prefixed ``synthetic_``); they are correctness evidence for the
implemented mathematics, never market evidence, never a P&L or Sharpe-family
headline, and they make no live-trading claim. Fail-closed throughout:
degenerate grids, non-finite or non-positive coefficients, violated
Assumption 4.1, ``eps`` outside ``(0, T^2/4]``, or malformed target paths all
raise ``ValueError``.

Composition notes: ``almgren_chriss.almgren_chriss_trajectory`` owns the
static mean-variance AC schedule and its TWAP limit -- reused in the test
lane for the ``beta -> inf`` reduction, not reimplemented. ``impact.py``
owns the Bouchaud power-law propagator family ``G(tau) = (1+tau)^{-beta}``;
the OW exponential resilience kernel of eq. (4.3) is a different impact
family (its convolution is defined here, not imported, because the kernels
are not interchangeable). ``impact._finite_scalar``-style validators are
rewritten locally to keep this module self-contained per house style.

References:
- Nutz, M., & Voss, M. (2026). The convergence rate of stochastic tracking
  with application to optimal execution. arXiv:2608.29468 [q-fin.TR] --
  Thm 4.2 (optimal strategy), Lemma 4.5 (Hilbert form), Lemma 5.5 (explicit
  strategy), Thm 5.7 (bounds), Prop. 5.9 (sharp sqrt(eps) rate).
- Obizhaeva, A., & Wang, J. (2013). Optimal trading strategy and
  supply/demand dynamics. *Journal of Financial Markets* 16(1), 1-32.
- Fruth, A., Schoneborn, T., & Urusov, M. (2019). Optimal trade execution
  in order books with stochastic liquidity. *Mathematical Finance* 29(2).
- Chen, L., Horst, U., & Tran, T. N. (2024). Portfolio liquidation under
  transient price impact -- closed-form benchmark cited at eqs. 5.36-5.40.
- Bank, P., Soner, H. M., & Voss, M. (2017). Hedging with temporary price
  impact. *Mathematics and Financial Economics* 11(2) -- tracking setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
type RNG = np.random.Generator | int

#: Uniform bound-from-zero floor for the Assumption 4.1(iii) no-price-
#: manipulation condition 2 beta + gamma_dot > c > 0 on the grid.
_RESILIENCE_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Validation helpers (house style: fail-closed, no silent returns)
# ---------------------------------------------------------------------------


def _finite_scalar(x: float, name: str, *, positive: bool = False) -> float:
    v = float(x)
    if not np.isfinite(v) or (positive and v <= 0.0):
        kind = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {kind}")
    return v


def _as_1d(x: Array | float, name: str) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0 or not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return v


def _coeff_vector(x: float | Array, name: str, n_cols: int) -> Array:
    """Scalar -> constant vector; vector of length n_cols -> itself."""
    v = _as_1d(x, name)
    if v.size == 1:
        return np.full(n_cols, v[0], dtype=np.float64)
    if v.size != n_cols:
        raise ValueError(f"{name} must be scalar or length {n_cols}")
    return np.asarray(v, dtype=np.float64)


def _as_paths(x: Array, name: str, n_cols: int) -> Array:
    """Normalize to (P, n_cols) finite paths; 1-D input becomes one path."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != n_cols or a.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_paths, {n_cols})")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    return np.asarray(a, dtype=np.float64)


def _resolve_rng(rng: RNG | None) -> np.random.Generator:
    if rng is None:
        raise ValueError("pass rng (seed int or Generator); simulators are seeded-only")
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# ---------------------------------------------------------------------------
# Grid and coefficients
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingGrid:
    """Uniform grid ``t_i = i h`` on ``[0, horizon]`` with ``n_steps >= 2``.

    ``n_steps >= 2`` guarantees at least one full grid cell inside the
    terminal bridge region ``(T - sqrt(eps), T]`` whenever
    ``eps <= T^2/4`` (the paper's admissible range, Lemma 5.5).
    """

    horizon: float
    n_steps: int

    def __post_init__(self) -> None:
        t = _finite_scalar(self.horizon, "horizon", positive=True)
        object.__setattr__(self, "horizon", t)
        n = self.n_steps
        if isinstance(n, bool) or int(n) != n or int(n) < 2:
            raise ValueError("n_steps must be an integer >= 2")
        object.__setattr__(self, "n_steps", int(n))

    @property
    def step(self) -> float:
        return self.horizon / self.n_steps

    @property
    def times(self) -> Array:
        return np.arange(self.n_steps + 1, dtype=np.float64) * self.step


@dataclass(frozen=True)
class OWCoefficients:
    """Generalized OW coefficients sampled on a :class:`TrackingGrid`.

    ``beta`` resilience, ``lam`` depth (both length ``n_steps + 1``);
    ``gamma_dot = dot(lam)/lam`` discretized as ``(lam_i - lam_{i-1})/(h lam_i)``
    for ``i >= 1`` with ``gamma_dot_0 = gamma_dot_1``; ``eta``, ``theta_dot``,
    cumulative ``theta`` (trapezoid, ``theta_0 = 0``) and ``alpha = eta/lam``
    per paper eq. (4.1)-(4.2); ``weight = (2 beta + gamma_dot)/(2 lam)`` is the
    integrand of the Hilbert form (4.16); ``c_h_sq``, ``l_y``, ``c_j`` are the
    bound constants of eqs. (5.6), (5.9), (5.10).
    """

    beta: Array
    lam: Array
    gamma_dot: Array
    eta: Array
    theta_dot: Array
    theta: Array
    alpha: Array
    weight: Array
    c_h_sq: float
    l_y: float
    c_j: float


def ow_coefficients(
    beta: float | Array,
    lam: float | Array,
    grid: TrackingGrid,
) -> OWCoefficients:
    """Build validated OW coefficients from scalars or ``(n_steps+1,)`` arrays.

    Fail-closed on Assumption 4.1: ``lam`` strictly positive, all entries
    finite, and ``2 beta + gamma_dot >= _RESILIENCE_FLOOR > 0`` uniformly
    (no price manipulation). ``beta >= 0`` (resilience rate).
    """
    n = grid.n_steps
    h = grid.step
    b = _coeff_vector(beta, "beta", n + 1)
    lam_arr = _coeff_vector(lam, "lam", n + 1)
    if bool(np.any(b < 0.0)):
        raise ValueError("beta must be >= 0")
    if bool(np.any(lam_arr <= 0.0)):
        raise ValueError("lam must be > 0 (depth bounded away from zero)")
    gamma_dot = np.empty(n + 1, dtype=np.float64)
    gamma_dot[1:] = (lam_arr[1:] - lam_arr[:-1]) / (h * lam_arr[1:])
    gamma_dot[0] = gamma_dot[1]
    denom = 2.0 * b + gamma_dot
    if bool(np.any(denom < _RESILIENCE_FLOOR)):
        raise ValueError(
            "2*beta + gamma_dot must be bounded away from zero "
            "(Assumption 4.1(iii): no price manipulation)"
        )
    eta = (b + gamma_dot) / denom
    theta_dot = (b + gamma_dot) ** 2 / (lam_arr * denom)
    theta = np.zeros(n + 1, dtype=np.float64)
    theta[1:] = np.cumsum(0.5 * h * (theta_dot[:-1] + theta_dot[1:]))
    alpha = eta / lam_arr
    weight = denom / (2.0 * lam_arr)
    c_h_sq = 0.5 * float(max(np.max(denom / lam_arr), 1.0 / lam_arr[-1]))
    l_y = float(
        np.max(lam_arr)
        * math.sqrt(2.0 + grid.horizon * (grid.horizon + 2.0) * np.max(b + gamma_dot) ** 2)
    )
    return OWCoefficients(
        beta=b,
        lam=lam_arr,
        gamma_dot=gamma_dot,
        eta=eta,
        theta_dot=theta_dot,
        theta=theta,
        alpha=alpha,
        weight=weight,
        c_h_sq=c_h_sq,
        l_y=l_y,
        c_j=c_h_sq * l_y * l_y,
    )


# ---------------------------------------------------------------------------
# Impact process and cost functionals
# ---------------------------------------------------------------------------


def position_trades(positions: Array) -> Array:
    """Trade vector ``dQ_i = Q_i - Q_{i-1}`` (``dQ_0 = Q_0 - Q_{0-} = Q_0``)."""
    q = np.asarray(positions, dtype=float)
    return np.diff(q, axis=-1, prepend=np.zeros(q.shape[:-1] + (1,)))


def impact_process(
    trades: Array,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
    y0: float = 0.0,
) -> Array:
    """Impact path ``Y`` of a trade schedule under the exponential OW kernel.

    ``trades[i]`` is the (possibly block) trade landing at grid time ``t_i``;
    ``Y_i = y e^{-B_i} + sum_{j <= i} e^{-(B_i - B_j)} lam_j trades_j`` with
    ``B_i = sum_{k <= i} beta_k h`` -- exact for piecewise-constant-in-cell
    impact between grid trades. Accepts ``(n_steps+1,)`` or ``(P, n_steps+1)``.
    """
    n = grid.n_steps
    h = grid.step
    d = _as_paths(trades, "trades", n + 1)
    y_init = _finite_scalar(y0, "y0")
    decay = np.exp(-coeffs.beta * h)  # decay_i over cell (t_{i-1}, t_i]
    decay[0] = 1.0
    y = np.empty_like(d)
    prev = np.full(d.shape[0], y_init) + coeffs.lam[0] * d[:, 0]
    y[:, 0] = prev
    for i in range(1, n + 1):
        prev = decay[i] * prev + coeffs.lam[i] * d[:, i]
        y[:, i] = prev
    return y[0] if np.asarray(trades).ndim == 1 else y


def _cell_impact_sq(y_paths: Array, coeffs: OWCoefficients, grid: TrackingGrid) -> Array:
    """Per-path ``int_0^T weight_t Y_t^2 dt`` via exact cell integrals.

    On cell ``(t_{i-1}, t_i]`` the path decays as ``Y_{i-1} e^{-beta_i s}`` so
    ``int_cell Y^2 = Y_{i-1}^2 (1 - e^{-2 beta_i h})/(2 beta_i)`` (``h Y^2`` if
    ``beta_i = 0``); ``weight`` is cell-averaged. Exact for the
    piecewise-decaying grid path.
    """
    n = grid.n_steps
    h = grid.step
    b = coeffs.beta
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_int = np.where(b > 0.0, -0.5 * np.expm1(-2.0 * b * h) / b, h)
    # cell j+1 spans (t_j, t_{j+1}]: path decays from left state y_j; weight
    # is cell-averaged. terms[j] multiplies y_j^2 (exact for the grid path).
    w_cell = 0.5 * (coeffs.weight[:-1] + coeffs.weight[1:])
    terms = w_cell * cell_int[1:]
    seg = (y_paths[:, :n] ** 2) * terms[np.newaxis, :]
    return np.asarray(np.sum(seg, axis=1), dtype=np.float64)


def hilbert_sq_norm(
    y_paths: Array,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
) -> Array:
    """Per-path Hilbert energy ``<Y, Y>_H`` of paper eq. (4.16).

    ``int_0^T weight_t Y_t^2 dt + Y_T^2/(2 lam_T)``. Takes impact paths
    ``(P, n_steps+1)`` (or ``(n_steps+1,)``), returns ``(P,)``.
    """
    y = _as_paths(y_paths, "y_paths", grid.n_steps + 1)
    out = _cell_impact_sq(y, coeffs, grid) + y[:, -1] ** 2 / (2.0 * coeffs.lam[-1])
    return np.asarray(out, dtype=np.float64)


def impact_cost(
    positions: Array,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
    y0: float = 0.0,
) -> Array:
    """Per-path unregularized execution cost ``J_0(Q)`` (eq. 4.5 = 4.15).

    Valid for arbitrary cadlag grid strategies, including initial and
    terminal block trades; the terminal position ``Q_T`` is whatever the path
    says (strategies in ``Q_cal`` satisfy ``Q_T = Xi_T``).
    """
    n = grid.n_steps
    q = _as_paths(positions, "positions", n + 1)
    y_init = _finite_scalar(y0, "y0")
    y = impact_process(position_trades(q), coeffs, grid, y_init)
    out = hilbert_sq_norm(y, coeffs, grid) - y_init**2 / (2.0 * coeffs.lam[0])
    return np.asarray(out, dtype=np.float64)


def regularized_cost(
    positions: Array,
    eps: float,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
    y0: float = 0.0,
) -> Array:
    """Per-path regularized objective ``J_eps(Q)`` of eq. (5.2).

    ``J_0(Q) + eps * int (dQ/dt)^2 dt``; the trading rate of a grid path is
    ``u_i = dQ_i/h`` on cell ``(t_{i-1}, t_i]`` (a block trade is a finite
    rate on the grid -- the penalty explodes as ``h -> 0``, as it should for
    singular strategies).
    """
    e = _finite_scalar(eps, "eps")
    if e < 0.0:
        raise ValueError("eps must be >= 0")
    q = _as_paths(positions, "positions", grid.n_steps + 1)
    j0 = impact_cost(q, coeffs, grid, y0)
    rates = position_trades(q) / grid.step
    return j0 + e * grid.step * np.sum(rates**2, axis=1)


# ---------------------------------------------------------------------------
# Unregularized optimum (Theorem 4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnregularizedSolution:
    """Optimal unregularized strategy of Theorem 4.2 on the grid.

    ``positions`` = ``Q^0`` (``(P, n_steps+1)``, ``Q^0_T = Xi_T``);
    ``martingale`` = auxiliary martingale ``M`` of eqs. (4.6)-(4.8);
    ``impact`` = ``Y^{Q^0}`` (equals ``eta_t M_t`` in the interior and
    ``M_T`` at ``T``, eq. 4.9, up to grid resolution at jump cells);
    ``xi`` = target-martingale paths ``Xi_t``; ``block_initial`` /
    ``block_terminal`` are the eq. (4.11) trading spikes; ``value`` =
    ``V(0)`` of eq. (4.12) as a path mean.
    """

    positions: Array
    martingale: Array
    impact: Array
    xi: Array
    block_initial: Array
    block_terminal: Array
    value: float


def _resolve_xi(xi: float | Array, n_cols: int) -> Array:
    """Deterministic scalar -> one constant path; arrays -> (P, n_cols)."""
    if np.isscalar(xi) or np.asarray(xi).ndim == 0:
        v = _finite_scalar(float(xi), "xi")
        return np.full((1, n_cols), v, dtype=np.float64)
    return _as_paths(np.asarray(xi), "xi", n_cols)


def optimal_unregularized(
    xi: float | Array,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
    y0: float = 0.0,
) -> UnregularizedSolution:
    """Optimal singular strategy ``Q^0`` of Theorem 4.2, pathwise.

    ``xi`` is the terminal-target martingale: a scalar deterministic target,
    a ``(n_steps+1,)`` single path, or ``(P, n_steps+1)`` paths whose last
    column is ``Xi_T`` (input contract: rows must be martingales on the
    grid -- a path family property the module cannot verify). The Ito weight
    ``lam_T/(1 + lam_T (theta_T - theta_s))`` is evaluated at each
    increment's landing grid point.
    """
    n = grid.n_steps
    h = grid.step
    x = _resolve_xi(xi, n + 1)
    y_init = _finite_scalar(y0, "y0")
    lam_t = coeffs.lam[-1]
    theta_t = coeffs.theta[-1]
    m0 = (lam_t / (1.0 + lam_t * theta_t)) * (x[:, 0] + y_init / coeffs.lam[0])
    d_xi = np.diff(x, axis=1)  # d_xi[:, j-1] = increment landing at t_j
    w = lam_t / (1.0 + lam_t * (theta_t - coeffs.theta[1:]))
    m = m0[:, np.newaxis] + np.concatenate(
        [np.zeros((x.shape[0], 1)), np.cumsum(w[np.newaxis, :] * d_xi, axis=1)], axis=1
    )
    # drift_i = int_0^{t_i} theta_dot_s M_s ds: cell-mean theta_dot times the
    # cadlag left value M_{j-1} -- exact for piecewise-constant M.
    td_cell = 0.5 * (coeffs.theta_dot[:-1] + coeffs.theta_dot[1:])
    drift = h * np.concatenate(
        [np.zeros((x.shape[0], 1)), np.cumsum(td_cell[np.newaxis, :] * m[:, :-1], axis=1)],
        axis=1,
    )
    q_inner = coeffs.alpha[np.newaxis, :] * m - y_init / coeffs.lam[0] + drift
    q = q_inner.copy()
    q[:, -1] = x[:, -1]  # terminal constraint Q_T = Xi_T
    y = impact_process(position_trades(q), coeffs, grid, y_init)
    m_sq_cell = 0.5 * (
        coeffs.theta_dot[:-1][np.newaxis, :] * m[:, :-1] ** 2
        + coeffs.theta_dot[1:][np.newaxis, :] * m[:, 1:] ** 2
    )
    m_sq_int = h * np.sum(m_sq_cell, axis=1)
    value = float(
        0.5 * (np.mean(m_sq_int) + np.mean(m[:, -1] ** 2) / lam_t - y_init**2 / coeffs.lam[0])
    )
    return UnregularizedSolution(
        positions=q,
        martingale=m,
        impact=y,
        xi=x,
        block_initial=q[:, 0].copy(),
        block_terminal=(x[:, -1] - q_inner[:, -1]),
        value=value,
    )


# ---------------------------------------------------------------------------
# Explicit regularized strategy (Lemma 5.5, eqs. 5.18-5.20)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplicitStrategy:
    """Absolutely continuous near-optimal strategy ``Q^eps`` (eq. 5.19).

    ``positions`` = ``Q^eps`` with ``Q^eps_0 = 0`` and ``Q^eps_T = Xi_T``;
    ``rates`` = per-cell trading rate ``u_i = dQ_i/h`` (column ``i`` is the
    rate on ``(t_{i-1}, t_i]``); ``switch_index`` = last grid index
    ``<= T - sqrt(eps)``; ``tracking_l2`` = path mean of
    ``int (Q^eps_t - Q^0_t)^2 dt``; ``rate_integral`` = path mean of
    ``int u_t^2 dt``.
    """

    positions: Array
    rates: Array
    switch_index: int
    eps: float
    tracking_l2: float
    rate_integral: float


def explicit_strategy(
    solution: UnregularizedSolution,
    eps: float,
    coeffs: OWCoefficients,
    grid: TrackingGrid,
) -> ExplicitStrategy:
    """Implementable ``Q^eps`` of Lemma 5.5 tracking ``Q^0``, then bridging to ``Xi_T``.

    The exponential filter ``Qbar_t = (1/a) int_0^t e^{-(t-r)/a} Q^0_r dr``
    (``a = sqrt(eps)``) is applied to the cadlag step path ``Q^0`` -- exact
    recurrence ``Qbar_i = e^{-h/a} Qbar_{i-1} + Q^0_{i-1} (1 - e^{-h/a})``.
    On ``(T-a, T)`` the terminal bridge integrates ``Xi_r/(T-r)^2`` cellwise
    against the cadlag left value ``Xi_{j-1}`` -- exact per cell; for a
    deterministic target it reduces to a linear bridge. Fail-closed unless
    ``0 < eps <= T^2/4`` (the paper's admissible range).
    """
    n = grid.n_steps
    h = grid.step
    t_end = grid.horizon
    e = _finite_scalar(eps, "eps", positive=True)
    if e > 0.25 * t_end * t_end:
        raise ValueError("eps must be <= T^2/4 (Lemma 5.5 admissible range)")
    a = float(np.sqrt(e))
    times = grid.times
    q0 = solution.positions
    x = solution.xi
    k_s = int(np.floor((t_end - a) / h))
    k_s = min(max(k_s, 1), n - 1)
    q = np.zeros_like(q0)
    f = float(np.exp(-h / a))
    for i in range(1, k_s + 1):
        q[:, i] = f * q[:, i - 1] + q0[:, i - 1] * (1.0 - f)
    # bridge: acc_i = Qbar_ks/a + sum_{j=ks+1..i} Xi_{j-1} (1/(T-t_j) - 1/(T-t_{j-1}))
    acc = q[:, k_s] / a
    for i in range(k_s + 1, n):
        acc = acc + x[:, i - 1] * (1.0 / (t_end - times[i]) - 1.0 / (t_end - times[i - 1]))
        q[:, i] = (t_end - times[i]) * acc
    q[:, -1] = x[:, -1]
    rates = position_trades(q) / h
    tracking = h * np.sum(
        0.5 * ((q[:, :-1] - q0[:, :-1]) ** 2 + (q[:, 1:] - q0[:, 1:]) ** 2), axis=1
    )
    return ExplicitStrategy(
        positions=q,
        rates=rates,
        switch_index=k_s,
        eps=e,
        tracking_l2=float(np.mean(tracking)),
        rate_integral=float(h * np.mean(np.sum(rates**2, axis=1))),
    )


# ---------------------------------------------------------------------------
# Chen--Horst--Tran closed form (constant benchmark; paper eqs. 5.36-5.40)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegularizedSolution:
    """Exact regularized optimizer for constant ``beta, lam``, deterministic target.

    ``positions`` = ``Q^{*,eps}`` on the grid; ``rates`` = ``u^{*,eps}`` at
    grid points; ``value`` = ``V(eps)`` via eq. (5.37); ``k_eps`` = the
    ``k_epsilon`` of eq. (5.36).
    """

    positions: Array
    rates: Array
    value: float
    k_eps: float


def _cht_pieces(
    xi_terminal: float,
    beta: float,
    lam: float,
    eps: float,
    horizon: float,
) -> tuple[float, float, float, float]:
    """Stable ``(k_eps, a_over_sinhK, b_over_sinhK, denom)`` of eq. (5.36).

    Both ``sinh(K_eps)`` and ``cosh(K_eps)`` overflow for ``eps ~ 1e-9``, so
    numerator and denominator are divided by ``sinh(K)``: ``a/sinhK = lam/eps``,
    ``b/sinhK = beta k coth K + k^2``, ``c = lam/eps``, and
    ``coth K = (1 + e^{-2K})/(1 - e^{-2K})``.
    """
    _finite_scalar(xi_terminal, "xi_terminal")
    b_ = _finite_scalar(beta, "beta", positive=True)
    l_ = _finite_scalar(lam, "lam", positive=True)
    e = _finite_scalar(eps, "eps", positive=True)
    t_end = _finite_scalar(horizon, "horizon", positive=True)
    k = float(np.sqrt(b_ * b_ + b_ * l_ / e))
    big_k = k * t_end / 2.0
    e2k = float(np.exp(-2.0 * big_k))
    coth_k = (1.0 + e2k) / (1.0 - e2k)
    a_over_sinh = l_ / e
    b_over_sinh = b_ * k * coth_k + k * k
    denom = 2.0 * a_over_sinh + b_over_sinh * t_end
    return k, a_over_sinh, b_over_sinh, denom


def cht_value(
    eps: float,
    xi_terminal: float,
    beta: float,
    lam: float,
    horizon: float,
) -> float:
    """Regularized value ``V(eps)`` of eq. (5.37) in the constant benchmark."""
    xi = _finite_scalar(xi_terminal, "xi_terminal")
    b_ = _finite_scalar(beta, "beta", positive=True)
    e = _finite_scalar(eps, "eps", positive=True)
    k, _a, b_over_sinh, denom = _cht_pieces(xi, b_, lam, e, horizon)
    d_eps = xi * b_over_sinh / denom
    return float(e * k * k / (b_ * b_) * d_eps * xi)


def cht_regularized(
    xi_terminal: float,
    beta: float,
    lam: float,
    eps: float,
    grid: TrackingGrid,
) -> RegularizedSolution:
    """Exact regularized optimizer ``Q^{*,eps}``, ``u^{*,eps}`` (eq. 5.36).

    Constant ``beta, lam``, deterministic ``xi_terminal``, ``y = 0`` only --
    this is the benchmark used by Proposition 5.9. Evaluated in the
    ``sinh(K)``-rescaled (overflow-free) form; the ``cosh`` rate uses the
    same rescaling ``cosh(k(t-T/2))/sinh(K) = e^{|u|-K} (1+e^{-2|u|})/(1-e^{-2K})``.
    """
    xi = _finite_scalar(xi_terminal, "xi_terminal")
    b_ = _finite_scalar(beta, "beta", positive=True)
    l_ = _finite_scalar(lam, "lam", positive=True)
    e = _finite_scalar(eps, "eps", positive=True)
    t_end = grid.horizon
    k, a_s, b_s, den = _cht_pieces(xi, b_, l_, e, t_end)
    c = l_ / e
    big_k = k * t_end / 2.0
    times = grid.times
    arg = k * (times - t_end / 2.0)
    abs_arg = np.abs(arg)
    e2k = np.exp(-2.0 * big_k)
    sinh_ratio = np.sign(arg) * np.exp(abs_arg - big_k) * (-np.expm1(-2.0 * abs_arg)) / (1.0 - e2k)
    cosh_ratio = np.exp(abs_arg - big_k) * (1.0 + np.exp(-2.0 * abs_arg)) / (1.0 - e2k)
    q = xi * (a_s + b_s * times + c * sinh_ratio) / den
    d_eps = xi * b_s / den
    u = d_eps + xi * c * k / den * cosh_ratio
    q[-1] = xi  # exact terminal constraint
    value = e * k * k / (b_ * b_) * d_eps * xi
    return RegularizedSolution(
        positions=np.asarray(q, dtype=np.float64),
        rates=np.asarray(u, dtype=np.float64),
        value=float(value),
        k_eps=k,
    )


def ow_benchmark_value(xi_terminal: float, beta: float, lam: float, horizon: float) -> float:
    """``V(0) = lam Xi^2/(beta T + 2)`` of eq. (5.35) (constant benchmark)."""
    xi = _finite_scalar(xi_terminal, "xi_terminal")
    b_ = _finite_scalar(beta, "beta", positive=True)
    l_ = _finite_scalar(lam, "lam", positive=True)
    t_end = _finite_scalar(horizon, "horizon", positive=True)
    return float(l_ * xi * xi / (b_ * t_end + 2.0))


def sharp_rate_constant(xi_terminal: float, beta: float, lam: float, horizon: float) -> float:
    """Leading ``sqrt(eps)`` coefficient ``2 sqrt(beta lam) Xi^2/(beta T+2)^2`` (Prop. 5.9)."""
    xi = _finite_scalar(xi_terminal, "xi_terminal")
    b_ = _finite_scalar(beta, "beta", positive=True)
    l_ = _finite_scalar(lam, "lam", positive=True)
    t_end = _finite_scalar(horizon, "horizon", positive=True)
    d = b_ * t_end + 2.0
    return float(2.0 * np.sqrt(b_ * l_) * xi * xi / (d * d))


# ---------------------------------------------------------------------------
# Time-translation modulus (tracking bounds, eqs. 2.6-2.7, 5.14-5.15)
# ---------------------------------------------------------------------------


def translation_modulus(paths: Array, lag_steps: Array, grid: TrackingGrid) -> Array:
    """Squared L2 time-translation modulus ``omega(h_k)`` of eq. (2.6).

    ``omega(k h) = E int_0^T |X_t - X_{(t - k h)+}|^2 dt`` discretized on the
    grid (left-continuous lag); ``lag_steps`` are integer cell lags ``>= 1``.
    Returns one value per lag.
    """
    p = _as_paths(paths, "paths", grid.n_steps + 1)
    lags = np.asarray(lag_steps, dtype=int).reshape(-1)
    if lags.size == 0 or bool(np.any(lags < 1)) or bool(np.any(lags > grid.n_steps)):
        raise ValueError(f"lag_steps must be integers in [1, {grid.n_steps}]")
    if not bool(np.all(np.asarray(lag_steps) == lags)):
        raise ValueError("lag_steps must be integers")
    h = grid.step
    out = np.empty(lags.size, dtype=np.float64)
    for i, k in enumerate(lags):
        lagged = p[:, np.clip(np.arange(grid.n_steps + 1) - int(k), 0, None)]
        out[i] = float(h * np.mean(np.sum((p - lagged) ** 2, axis=1)))
    return out


def tr_seminorm(paths: Array, grid: TrackingGrid) -> float:
    """Time-translation seminorm ``[X]_tr^2 = sup_h omega(h)/h`` (eq. 2.7)."""
    n = grid.n_steps
    omega = translation_modulus(paths, np.arange(1, n + 1, dtype=np.float64), grid)
    h = grid.step
    return float(np.max(omega / (np.arange(1, n + 1) * h)))


# ---------------------------------------------------------------------------
# SYNTHETIC martingale-target simulators (correctness material only)
# ---------------------------------------------------------------------------


def sim_instant_revelation(
    n_paths: int,
    grid: TrackingGrid,
    xi_mean: float,
    xi_sd: float,
    rng: RNG | None,
) -> Array:
    """SYNTHETIC ``Xi_t = Xi_T`` paths: all terminal information arrives at ``t = 0``.

    ``Xi_T ~ N(xi_mean, xi_sd^2)`` per path -- the simplest martingale
    satisfying the reachability condition (5.4); the optimal strategy reduces
    to the deterministic OW schedule applied to a random terminal size.
    """
    g = _resolve_rng(rng)
    mu = _finite_scalar(xi_mean, "xi_mean")
    sd = _finite_scalar(xi_sd, "xi_sd")
    if sd < 0.0:
        raise ValueError("xi_sd must be >= 0")
    if int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError("n_paths must be an integer >= 1")
    draws = mu + sd * g.standard_normal(int(n_paths))
    return np.repeat(draws[:, np.newaxis], grid.n_steps + 1, axis=1)


def sim_delayed_revelation(
    n_paths: int,
    grid: TrackingGrid,
    xi_mean: float,
    xi_sd: float,
    reveal_index: int,
    rng: RNG | None,
) -> Array:
    """SYNTHETIC single-jump target martingale: ``Xi`` reveals ``Xi_T`` at ``t_{reveal}``.

    ``Xi_i = xi_mean`` for ``i < reveal_index`` else ``Xi_T`` -- a jump
    martingale with all quadratic variation at one grid time, satisfying
    the reachability condition (5.4) whenever ``reveal_index < n_steps``.
    ``reveal_index = 0`` degenerates to instant revelation.
    """
    g = _resolve_rng(rng)
    mu = _finite_scalar(xi_mean, "xi_mean")
    sd = _finite_scalar(xi_sd, "xi_sd")
    if sd < 0.0:
        raise ValueError("xi_sd must be >= 0")
    r = int(reveal_index)
    if r < 0 or r > grid.n_steps:
        raise ValueError(f"reveal_index must be in [0, {grid.n_steps}]")
    if int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError("n_paths must be an integer >= 1")
    x = np.full((int(n_paths), grid.n_steps + 1), mu, dtype=np.float64)
    x[:, r:] = (mu + sd * g.standard_normal(int(n_paths)))[:, np.newaxis]
    return x


def sim_gaussian_revelation(
    n_paths: int,
    grid: TrackingGrid,
    xi_mean: float,
    xi_sd: float,
    last_info_index: int,
    rng: RNG | None,
) -> Array:
    """SYNTHETIC continuous-revelation martingale, information done by ``t_{last}``.

    ``Xi_i = xi_mean + xi_sd * W_{min(i, last)}/sqrt(last)`` with
    ``W_i = sum_{j<=i} Z_j`` Gaussian increments: a discrete Brownian
    terminal-target martingale that stops learning after ``last_info_index``
    so the reachability integral (5.4) stays finite on the grid.
    """
    g = _resolve_rng(rng)
    mu = _finite_scalar(xi_mean, "xi_mean")
    sd = _finite_scalar(xi_sd, "xi_sd")
    if sd < 0.0:
        raise ValueError("xi_sd must be >= 0")
    last = int(last_info_index)
    if last < 1 or last > grid.n_steps:
        raise ValueError(f"last_info_index must be in [1, {grid.n_steps}]")
    if int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError("n_paths must be an integer >= 1")
    z = g.standard_normal((int(n_paths), last))
    w = np.cumsum(z, axis=1) / np.sqrt(last)
    x = np.empty((int(n_paths), grid.n_steps + 1), dtype=np.float64)
    x[:, 0] = mu
    x[:, 1 : last + 1] = mu + sd * w
    x[:, last + 1 :] = (mu + sd * w[:, -1])[:, np.newaxis]
    return x


# ---------------------------------------------------------------------------
# sqrt(eps) rate verification sweep (Prop. 5.9 + Thm 5.7 diagnostics)
# ---------------------------------------------------------------------------


def _loglog_slope(x: Array, y: Array) -> float:
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


def regularization_gap_sweep(
    epsilons: Array,
    xi: float | Array,
    beta: float,
    lam: float,
    horizon: float,
    n_steps: int,
    y0: float = 0.0,
) -> dict[str, float | Array]:
    """Empirical ``O(sqrt(eps))`` verification: value gaps and explicit-strategy excess.

    For each ``eps``: ``V(eps) - V(0)`` via the Chen--Horst--Tran closed form
    (deterministic scalar ``xi`` only -- returns NaN otherwise),
    ``J_0(Q^eps) - J_0(Q^0)`` via :func:`explicit_strategy` (path mean for
    stochastic ``xi``), the tracking L2 ``E int (Q^eps - Q^0)^2`` and the
    Lemma-5.5 combined quantity ``E int (Q^eps - Q^0)^2 + eps E int u^2`` of
    eq. (5.23). Fits log-log slopes of each gap. All outputs are SYNTHETIC
    correctness diagnostics, never market evidence. On a discrete grid the
    smallest ``eps`` with ``sqrt(eps)`` below a few cell widths hit a
    resolution floor and drag fitted slopes below 1/2 -- restrict sweeps to
    ``sqrt(eps) >= ~5 h``.
    """
    eps_arr = np.sort(_as_1d(epsilons, "epsilons"))
    if eps_arr.size < 2:
        raise ValueError("epsilons must contain >= 2 points for a slope fit")
    if bool(np.any(eps_arr <= 0.0)) or bool(np.any(eps_arr > 0.25 * horizon * horizon)):
        raise ValueError("epsilons must lie in (0, T^2/4]")
    grid = TrackingGrid(horizon, n_steps)
    coeffs = ow_coefficients(beta, lam, grid)
    sol = optimal_unregularized(xi, coeffs, grid, y0)
    v0 = sol.value
    deterministic = sol.xi.shape[0] == 1
    gaps_cht = np.full(eps_arr.size, np.nan)
    gaps_explicit = np.empty(eps_arr.size, dtype=np.float64)
    tracking = np.empty(eps_arr.size, dtype=np.float64)
    combined = np.empty(eps_arr.size, dtype=np.float64)
    mean_q0_cost = float(np.mean(impact_cost(sol.positions, coeffs, grid, y0)))
    for i, e in enumerate(eps_arr):
        if deterministic:
            gaps_cht[i] = cht_value(float(e), float(sol.xi[0, -1]), beta, lam, horizon) - v0
        strat = explicit_strategy(sol, float(e), coeffs, grid)
        gaps_explicit[i] = (
            float(np.mean(impact_cost(strat.positions, coeffs, grid, y0))) - mean_q0_cost
        )
        tracking[i] = strat.tracking_l2
        combined[i] = strat.tracking_l2 + float(e) * strat.rate_integral
    return {
        "eps": eps_arr,
        "synthetic_value_gap_cht": gaps_cht,
        "synthetic_excess_explicit": gaps_explicit,
        "synthetic_tracking_l2": tracking,
        "synthetic_combined_bound": combined,
        "synthetic_loglog_slope_cht": _loglog_slope(eps_arr, gaps_cht),
        "synthetic_loglog_slope_explicit": _loglog_slope(eps_arr, gaps_explicit),
        "synthetic_loglog_slope_tracking": _loglog_slope(eps_arr, tracking),
        "synthetic_loglog_slope_combined": _loglog_slope(eps_arr, combined),
        "value_unregularized": v0,
        "n_steps": float(n_steps),
    }
