"""Cash-constrained multi-asset optimal execution (QCQP formulation).

Extends the discrete Almgren-Chriss framework to a multi-asset portfolio with
linear permanent/temporary *cross*-impact and intertemporal expected-cash
constraints, following Hashimoto & Stillman (2026, arXiv:2609.27786).  The
trader must execute signed orders ``Q^(n)`` over ``K`` periods of length
``tau`` while keeping the *expected cumulative net cash spent* under a budget
at every period (paper Eq. 6):

    min_{Q_k}  E[IS] + lam * Var(IS)
    s.t.       sum_k Q_k^(n) = Q^(n)          (inventory, all n)
               E[M_k] <= c_k                  (cash, all k = 1..K)

with the paper's price convention (Eq. 4): trade ``k`` bears its own period's
permanent impact, ``P_k = P_{k-1} + tau * g(Q_k/tau) + sigma * sqrt(tau) Z_k``
and execution price ``P~_k = P_k + h(Q_k/tau)``, with linear impact
``g(x) = Gamma x``, ``h(x) = Eta x``.  (This timing convention differs from
the Almgren-Chriss (2001) midpoint convention by a diagonal ``+Gamma/2``
shift; the paper's Proposition 1 derivation is reproduced exactly.)

Under linear impact the problem is a QCQP (paper Proposition 1): with the
asset-major stacking ``x = (Q^(1); ...; Q^(N))`` in R^{NK}, upper-triangular
ones ``U`` (``q = U Q`` collects remaining orders) and ``u_k``/``L_k``/``D_k``
the period-k cash selectors,

    objective   x' H x,   H = A + lam * B
    A           = kron(Eta, I_K)/tau + kron(Gamma, U')          (E[IS], Eq. 16)
    B           = tau * kron(Sigma, U'U)                        (Var IS, Eq. 17)
    inventory   C x = b,  C = blkdiag(1_K'),  b = orders        (Eq. 19)
    cash k      a_k' x + x' D_k x <= c_k                        (Eq. 25-27)
    a_k         = kron(p0, u_k)
    D_k         = kron(Gamma, (L_k + L_k')/2) + kron(Eta, D_k^u)/tau

Convexity conditions (paper Proposition 2, generalized to cross-impact and
asserted fail-closed here): ``Gamma``, ``Eta``, ``Sigma`` symmetric positive
semidefinite, ``lam >= 0``, ``tau > 0``.  Then ``A``, ``B`` and every ``D_k``
are PSD (kron of PSD factors; ``(U+U')/2`` and ``(L_k+L_k')/2`` are PSD by the
paper's Eqs. 28-29), so the QCQP is convex and solvable to global optimality.
Symmetric PSD permanent cross-impact is also the standard multi-asset
no-price-manipulation condition (Huberman & Stanzl 2004); the paper's scalar
coefficients ``gamma^(n), eta^(n) >= 0`` are the diagonal special case.

Paper corrections found while reproducing (fail-closed behaviour here):
- The Table-1 budgets ``c_k = 5 + beta*k`` with ``beta in {0.1, 0.2}`` are
  strictly infeasible at the terminal period: with ``Q = +-20``, ``K = 20``,
  ``gamma = 0.02``, ``eta = 0.05`` the minimum possible ``E[M_K]`` is
  ``2 * (gamma/2 * (Q^2 + Q^2/K) + eta * Q^2/K) = 10.4 > c_K = 7`` (resp. 9),
  whatever the schedule.  This module raises ``RuntimeError`` on infeasible
  budgets instead of returning a constraint-violating schedule (the paper
  used SCS, which can return inaccurate points on infeasible inputs).
- With perfectly symmetric legs (equal prices, equal impact/vol) no feasible
  budget produces the sell-first asymmetry of the paper's Figure 2, since the
  linear cash term cancels; the sell-first mechanism requires imbalanced cash
  legs (as in the paper's Experiment 2, where buy/sell notionals differ).

Honesty: all numbers this module produces are execution-cost quantities
(expected/realized implementation shortfall in the Perold 1988 sense, expected
cash paths, peak cash drawdown of the *funding ledger* normalised by buy
notional).  SYNTHETIC outputs from :func:`simulate_execution` are correctness
tests of the closed-form expectations, never market evidence.  No Sharpe /
Sortino / Calmar / P&L / NAV content.

References:
- Hashimoto, R., & Stillman, N. R. (2026). Feasible multi-asset optimal
  execution under cash constraints. arXiv:2609.27786 — Eqs. 4-7 (setup and
  QCQP), Propositions 1-2 (reformulation and convexity), Tables 1-2.
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio
  transactions. *Journal of Risk* 3, 5-39.
- Perold, A. (1988). The implementation shortfall: paper versus reality.
  *Journal of Portfolio Management* 14.
- Huberman, G., & Stanzl, W. (2004). Price manipulation and quasi-arbitrage.
  *Econometrica* 72(4), 1247-1275 — PSD permanent cross-impact condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "OeResult",
    "QcqpForm",
    "build_qcqp",
    "is_components",
    "peak_cash_drawdown",
    "schedule_shape",
    "simulate_execution",
    "solve_multiasset_oe",
    "unconstrained_oe",
]

# Dense NK x NK blocks are built for the objective and every cash constraint,
# so memory scales like K * (N*K)^2; refuse oversized problems fail-closed
# instead of exhausting memory (lab use is N, K in the tens).
_MAX_STACKED_DIM = 600
_PSD_TOL = 1e-10
_FEASIBILITY_TOL = 1e-6


def _as_symmetric_psd(m: Array, name: str, n: int) -> Array:
    """Validate a finite symmetric positive-semidefinite (n, n) matrix."""
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape != (n, n) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite ({n}, {n}) matrix")
    scale = max(1.0, float(np.abs(a).max()))
    if np.max(np.abs(a - a.T)) > 1e-6 * scale:
        raise ValueError(f"{name} must be symmetric (cross-impact arbitrage condition)")
    if float(np.linalg.eigvalsh(0.5 * (a + a.T)).min()) < -_PSD_TOL * scale:
        raise ValueError(f"{name} must be positive semidefinite (convexity condition)")
    return np.asarray(0.5 * (a + a.T), dtype=np.float64)


def _as_finite_vector(v: Array, name: str, n: int) -> Array:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return a


def _as_positive_vector(v: Array, name: str, n: int) -> Array:
    a = _as_finite_vector(v, name, n)
    if np.any(a <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return a


def _check_scalar(v: float, name: str, *, positive: bool = False, nonneg: bool = False) -> float:
    x = float(v)
    if positive:
        kind, ok = "positive and finite", np.isfinite(x) and x > 0.0
    elif nonneg:
        kind, ok = "non-negative and finite", np.isfinite(x) and x >= 0.0
    else:
        kind, ok = "finite", np.isfinite(x)
    if not ok:
        raise ValueError(f"{name} must be {kind}")
    return x


def _check_budgets(budgets: Array | None, n_periods: int) -> Array | None:
    if budgets is None:
        return None
    c = np.asarray(budgets, dtype=float).reshape(-1)
    if c.size != n_periods:
        raise ValueError(f"cash_budgets must have length n_periods={n_periods}")
    if np.any(np.isnan(c)) or np.any(c == -np.inf):
        raise ValueError("cash_budgets entries must be finite or +inf (NaN/-inf rejected)")
    return c


def _upper_ones(n_periods: int) -> Array:
    """``U[k, i] = 1`` iff ``i >= k``; maps trades to remaining orders q = U Q."""
    return np.triu(np.ones((n_periods, n_periods), dtype=np.float64))


@dataclass(frozen=True)
class QcqpForm:
    """QCQP data of paper Proposition 1 (objective + inventory blocks).

    ``objective_matrix`` is ``H = A + risk_aversion * B`` (PSD under the
    documented convexity conditions); ``inventory_matrix``/``inventory_rhs``
    are ``C``/``b`` with ``C x = b`` enforcing full execution per asset.
    Cash-constraint blocks depend on ``p0`` and are built inside the solver.
    """

    objective_matrix: Array
    inventory_matrix: Array
    inventory_rhs: Array
    n_assets: int
    n_periods: int


@dataclass(frozen=True)
class OeResult:
    """Solution of the (cash-constrained) multi-asset Almgren-Chriss QCQP.

    ``trades`` is the (K, N) signed schedule (positive = buy, consumes cash).
    ``expected_cash_path[k]`` is ``E[M_{k+1}]``, the expected cumulative net
    cash spent after period k (paper Eq. 25); ``peak_cash_drawdown`` is
    ``max(0, max_k E[M_k])`` and ``peak_cash_drawdown_frac`` normalises it by
    the target buy notional ``sum_n p0_n * max(Q_n, 0)`` (NaN when the program
    has no buy leg).  ``expected_is``/``variance_is`` are the Perold shortfall
    moments against arrival prices; ``objective = expected_is + lam * var``.
    """

    trades: Array
    expected_is: float
    variance_is: float
    objective: float
    expected_cash_path: Array
    peak_cash_drawdown: float
    peak_cash_drawdown_frac: float
    risk_aversion: float
    cash_budgets: Array | None
    status: str


def build_qcqp(
    orders: Array,
    *,
    perm_impact: Array,
    temp_impact: Array,
    cov: Array,
    n_periods: int,
    tau: float = 1.0,
    risk_aversion: float = 0.0,
) -> QcqpForm:
    """Build the convex QCQP objective/inventory blocks (paper Proposition 1).

    ``perm_impact`` (Gamma), ``temp_impact`` (Eta) and ``cov`` (Sigma) are
    symmetric PSD (N, N) matrices; the paper's diagonal coefficients are the
    special case ``Gamma = diag(gamma^n)`` etc.  Raises ``ValueError`` on any
    violated convexity condition (fail-closed).
    """
    b = _as_finite_vector(orders, "orders", int(np.size(orders)))
    n = b.size
    if n < 1:
        raise ValueError("orders must contain at least one asset")
    if isinstance(n_periods, bool) or int(n_periods) != n_periods or int(n_periods) < 1:
        raise ValueError("n_periods must be an integer >= 1")
    k = int(n_periods)
    t = _check_scalar(tau, "tau", positive=True)
    lam = _check_scalar(risk_aversion, "risk_aversion", nonneg=True)
    gamma = _as_symmetric_psd(np.asarray(perm_impact, dtype=float), "perm_impact", n)
    eta = _as_symmetric_psd(np.asarray(temp_impact, dtype=float), "temp_impact", n)
    sigma = _as_symmetric_psd(np.asarray(cov, dtype=float), "cov", n)
    if n * k > _MAX_STACKED_DIM:
        raise ValueError(f"problem size n_assets * n_periods = {n * k} exceeds {_MAX_STACKED_DIM}")

    u = _upper_ones(k)
    eye_k = np.eye(k, dtype=np.float64)
    # E[IS] block (paper Eq. 16 generalised): temporary + permanent impact.
    a_mat = np.kron(eta, eye_k) / t + np.kron(gamma, u.T)
    a_mat = 0.5 * (a_mat + a_mat.T)
    # Var(IS) block (paper Eq. 17 generalised to a return covariance).
    b_mat = t * np.kron(sigma, u.T @ u)
    h_mat = 0.5 * ((a_mat + lam * b_mat) + (a_mat + lam * b_mat).T)
    scale = max(1.0, float(np.abs(h_mat).max()))
    if float(np.linalg.eigvalsh(h_mat).min()) < -_PSD_TOL * scale:
        raise ValueError("objective Hessian is not PSD; convexity conditions violated")

    c_mat = np.zeros((n, n * k), dtype=np.float64)
    for i in range(n):
        c_mat[i, i * k : (i + 1) * k] = 1.0
    return QcqpForm(
        objective_matrix=h_mat,
        inventory_matrix=c_mat,
        inventory_rhs=b,
        n_assets=n,
        n_periods=k,
    )


def _cash_blocks(
    k_idx: int,
    p0: Array,
    gamma: Array,
    eta: Array,
    n_periods: int,
    tau: float,
) -> tuple[Array, Array]:
    """Linear part ``a_k`` and PSD quadratic part ``D_k`` of ``E[M_k]``.

    Paper Eq. 25-27 generalised to cross-impact: ``E[M_k] = a_k'x + x'D_k x``
    with ``D_k = kron(Gamma, (L_k + L_k')/2) + kron(Eta, diag(u_k))/tau``.
    ``(L_k + L_k')/2`` is PSD by the paper's Eq. 29 argument, so ``D_k`` is a
    kron sum of PSD blocks (convex constraint) whenever Gamma, Eta are PSD.
    """
    u_vec = np.zeros(n_periods, dtype=np.float64)
    u_vec[: k_idx + 1] = 1.0
    a_k = np.asarray(np.kron(p0, u_vec), dtype=np.float64)
    l_k = np.tril(np.ones((n_periods, n_periods), dtype=np.float64))
    l_k[k_idx + 1 :, :] = 0.0
    d_k = np.kron(gamma, 0.5 * (l_k + l_k.T)) + np.kron(eta, np.diag(u_vec)) / tau
    return a_k, np.asarray(0.5 * (d_k + d_k.T), dtype=np.float64)


def solve_multiasset_oe(
    orders: Array,
    p0: Array,
    *,
    perm_impact: Array,
    temp_impact: Array,
    cov: Array,
    n_periods: int,
    tau: float = 1.0,
    risk_aversion: float = 0.0,
    cash_budgets: Array | None = None,
) -> OeResult:
    """Solve the safe-OE QCQP (paper Eq. 6-7) with per-period cash budgets.

    ``orders`` are signed target quantities per asset (positive = buy);
    ``p0`` are arrival reference prices; ``cash_budgets[k]`` caps the expected
    cumulative net cash spent after period k (entries may be ``+inf``;
    ``None`` removes the cash constraints entirely).  Convexity conditions
    (symmetric PSD impact/covariance matrices, ``risk_aversion >= 0``,
    ``tau > 0``) are validated up front and the assembled problem is checked
    for DCP convexity before solving — fail-closed with ``ValueError`` /
    ``RuntimeError``.  Infeasible budgets raise ``RuntimeError`` (see module
    docstring: the paper's tightest Table-1 budgets are infeasible).
    """
    import cvxpy as cp

    form = build_qcqp(
        orders,
        perm_impact=perm_impact,
        temp_impact=temp_impact,
        cov=cov,
        n_periods=n_periods,
        tau=tau,
        risk_aversion=risk_aversion,
    )
    n, k = form.n_assets, form.n_periods
    prices = _as_positive_vector(p0, "p0", n)
    budgets = _check_budgets(cash_budgets, k)
    t = float(tau)
    gamma = _as_symmetric_psd(np.asarray(perm_impact, dtype=float), "perm_impact", n)
    eta = _as_symmetric_psd(np.asarray(temp_impact, dtype=float), "temp_impact", n)

    x = cp.Variable(n * k)
    constraints = [form.inventory_matrix @ x == form.inventory_rhs]
    cash_lin: list[Array] = []
    cash_quad: list[Array] = []
    for idx in range(k):
        a_k, d_k = _cash_blocks(idx, prices, gamma, eta, k, t)
        cash_lin.append(a_k)
        cash_quad.append(d_k)
        if budgets is not None and np.isfinite(budgets[idx]):
            constraints.append(a_k @ x + cp.quad_form(x, cp.psd_wrap(d_k)) <= budgets[idx])

    prob = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(form.objective_matrix))), constraints)
    if not prob.is_dcp():
        raise RuntimeError("cash-constrained OE problem is not DCP-convex; refusing to solve")
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except (cp.error.SolverError, ArithmeticError, ValueError) as exc:
        # Narrowed from `except Exception` (quality ratchet, ceiling 72):
        # cvxpy raises SolverError on solver failure; numpy raises
        # LinAlgError (an ArithmeticError) on factorization breakdowns;
        # solvers may raise ValueError on malformed problem data. Exotic
        # exceptions propagate unwrapped — still fail-closed.
        raise RuntimeError(f"cash-constrained OE solve failed: solver error {exc}") from exc
    status = str(prob.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        raise RuntimeError(f"cash-constrained OE solve failed: solver status {status}")
    xv = np.asarray(x.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(xv)):
        raise RuntimeError("cash-constrained OE solve failed: non-finite solver output")

    # Post-hoc feasibility audit (fail-closed even on OPTIMAL_INACCURATE).
    inv_res = float(np.abs(form.inventory_matrix @ xv - form.inventory_rhs).max())
    if inv_res > _FEASIBILITY_TOL * max(1.0, float(np.abs(form.inventory_rhs).max())):
        raise RuntimeError(f"cash-constrained OE solve failed: inventory residual {inv_res:g}")
    cash_path = np.array(
        [float(a @ xv + xv @ d @ xv) for a, d in zip(cash_lin, cash_quad, strict=True)]
    )
    if budgets is not None:
        finite = np.isfinite(budgets)
        if np.any(finite):
            viol = float((cash_path[finite] - budgets[finite]).max())
            if viol > _FEASIBILITY_TOL * max(1.0, float(np.abs(budgets[finite]).max())):
                raise RuntimeError(f"cash-constrained OE solve failed: budget violation {viol:g}")

    trades = xv.reshape(n, k).T.copy()
    u = _upper_ones(k)
    eye_k = np.eye(k, dtype=np.float64)
    a_only = np.kron(eta, eye_k) / t + np.kron(gamma, u.T)
    a_only = 0.5 * (a_only + a_only.T)
    b_only = t * np.kron(_as_symmetric_psd(np.asarray(cov, dtype=float), "cov", n), u.T @ u)
    expected_is = float(xv @ a_only @ xv)
    variance_is = float(xv @ b_only @ xv)
    lam = float(risk_aversion)
    buy_notional = float(np.sum(prices * np.clip(form.inventory_rhs, 0.0, None)))
    peak = peak_cash_drawdown(cash_path)
    return OeResult(
        trades=trades,
        expected_is=expected_is,
        variance_is=variance_is,
        objective=float(expected_is + lam * variance_is),
        expected_cash_path=cash_path,
        peak_cash_drawdown=peak,
        peak_cash_drawdown_frac=(peak / buy_notional) if buy_notional > 0.0 else float("nan"),
        risk_aversion=lam,
        cash_budgets=None if budgets is None else budgets.copy(),
        status=status,
    )


def unconstrained_oe(
    orders: Array,
    p0: Array,
    *,
    perm_impact: Array,
    temp_impact: Array,
    cov: Array,
    n_periods: int,
    tau: float = 1.0,
    risk_aversion: float = 0.0,
) -> OeResult:
    """Classical multi-asset Almgren-Chriss baseline: same QCQP, no cash caps.

    Equivalent to ``solve_multiasset_oe(..., cash_budgets=None)`` and to any
    all-``+inf`` budget vector; since ``H`` is PSD (PD when ``Eta`` has a
    positive diagonal) the solution also equals the KKT closed form
    ``x* = H^{-1} C' (C H^{-1} C')^{-1} b`` of the equality-constrained QP.
    """
    return solve_multiasset_oe(
        orders,
        p0,
        perm_impact=perm_impact,
        temp_impact=temp_impact,
        cov=cov,
        n_periods=n_periods,
        tau=tau,
        risk_aversion=risk_aversion,
        cash_budgets=None,
    )


def is_components(
    trades: Array,
    *,
    perm_impact: Array,
    temp_impact: Array,
    cov: Array,
    tau: float = 1.0,
) -> dict[str, float]:
    """Perold-style expected implementation-shortfall decomposition.

    ``trades`` is the (K, N) signed schedule.  Returns the temporary-impact
    cost ``(1/tau) sum_k Q_k' Eta Q_k``, the permanent-impact cost
    ``sum_k q_k' Gamma Q_k`` on remaining orders ``q_k``, their sum
    ``expected_is`` (paper Eq. 9) and ``variance_is = tau sum_k q_k' Sigma q_k``
    (paper Eq. 10).  All are cost moments against arrival prices — no
    Sharpe/P&L content.
    """
    q = np.asarray(trades, dtype=float)
    if q.ndim != 2 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("trades must be a finite non-empty (K, N) matrix")
    k, n = q.shape
    t = _check_scalar(tau, "tau", positive=True)
    gamma = _as_symmetric_psd(np.asarray(perm_impact, dtype=float), "perm_impact", n)
    eta = _as_symmetric_psd(np.asarray(temp_impact, dtype=float), "temp_impact", n)
    sigma = _as_symmetric_psd(np.asarray(cov, dtype=float), "cov", n)
    remaining = np.cumsum(q[::-1], axis=0)[::-1]  # q_k = sum_{i>=k} Q_i
    temporary = float(np.einsum("kn,nm,km->", q, eta, q) / t)
    permanent = float(np.einsum("kn,nm,km->", remaining, gamma, q))
    variance = float(t * np.einsum("kn,nm,km->", remaining, sigma, remaining))
    return {
        "temporary_impact_cost": temporary,
        "permanent_impact_cost": permanent,
        "expected_is": temporary + permanent,
        "variance_is": variance,
    }


def peak_cash_drawdown(expected_cash_path: Array) -> float:
    """``max(0, max_k E[M_k])``: worst expected cumulative net cash outlay.

    This is a funding-ledger feasibility metric (paper's "peak cash
    drawdown"), not a strategy-return drawdown; a path that never spends more
    than it raises yields 0.
    """
    path = np.asarray(expected_cash_path, dtype=float).reshape(-1)
    if path.size == 0 or not np.all(np.isfinite(path)):
        raise ValueError("expected_cash_path must be a finite non-empty vector")
    return float(max(0.0, float(path.max())))


def schedule_shape(trades: Array) -> dict[str, float | Array | int]:
    """Schedule-shape metrics: first-half execution fractions and centroids.

    ``buy_first_half_fraction`` / ``sell_first_half_fraction`` are the traded
    volume fractions executed in the first ``ceil(K/2)`` periods (0 when that
    side is absent); ``timing_centroid[n]`` is the volume-weighted mean period
    (1-based) of asset n's trades, ``(K+1)/2`` for a flat asset.  Tighter cash
    budgets raise the sell fraction and lower the buy fraction (sell-first).
    """
    q = np.asarray(trades, dtype=float)
    if q.ndim != 2 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("trades must be a finite non-empty (K, N) matrix")
    k, n = q.shape
    if float(np.abs(q).sum()) <= 0.0:
        raise ValueError("schedule_shape is undefined for an all-zero schedule")
    half = (k + 1) // 2
    buy_total = float(np.clip(q, 0.0, None).sum())
    sell_total = float(np.clip(-q, 0.0, None).sum())
    buy_half = float(np.clip(q[:half], 0.0, None).sum())
    sell_half = float(np.clip(-q[:half], 0.0, None).sum())
    periods = np.arange(1, k + 1, dtype=np.float64)
    vol = np.abs(q)
    col_vol = vol.sum(axis=0)
    centroid = np.where(
        col_vol > 0.0,
        (periods @ vol) / np.where(col_vol > 0.0, col_vol, 1.0),
        (k + 1) / 2.0,
    )
    return {
        "buy_first_half_fraction": buy_half / buy_total if buy_total > 0.0 else 0.0,
        "sell_first_half_fraction": sell_half / sell_total if sell_total > 0.0 else 0.0,
        "timing_centroid": np.asarray(centroid, dtype=np.float64),
        "n_periods": k,
        "n_assets": n,
    }


def simulate_execution(
    trades: Array,
    p0: Array,
    *,
    perm_impact: Array,
    temp_impact: Array,
    cov: Array,
    tau: float = 1.0,
    n_paths: int = 1000,
    seed: int = 0,
) -> dict[str, Array]:
    """SYNTHETIC Monte-Carlo of the paper's price process (Eq. 4).

    Seeded correctness harness — never market evidence.  Simulates
    ``P_k = P_{k-1} + Gamma Q_k + sqrt(tau) L z_k`` (with ``L`` the symmetric
    PSD square root of ``cov``) and execution prices ``P~_k = P_k + Eta Q_k /
    tau``, returning per-path ``realized_is`` (against arrival prices),
    ``cash_paths`` (cumulative net cash ``M_k``) and ``peak_drawdowns``.
    Sample means must match the closed-form ``expected_is`` /
    ``expected_cash_path`` of :func:`solve_multiasset_oe`.
    """
    q = np.asarray(trades, dtype=float)
    if q.ndim != 2 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("trades must be a finite non-empty (K, N) matrix")
    k, n = q.shape
    prices = _as_positive_vector(p0, "p0", n)
    t = _check_scalar(tau, "tau", positive=True)
    gamma = _as_symmetric_psd(np.asarray(perm_impact, dtype=float), "perm_impact", n)
    eta = _as_symmetric_psd(np.asarray(temp_impact, dtype=float), "temp_impact", n)
    sigma = _as_symmetric_psd(np.asarray(cov, dtype=float), "cov", n)
    if isinstance(n_paths, bool) or int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError("n_paths must be an integer >= 1")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    paths = int(n_paths)

    eigvals, eigvecs = np.linalg.eigh(sigma)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((paths, k, n))
    noise = np.sqrt(t) * (z @ sqrt_cov)

    level = np.tile(prices, (paths, 1))
    realized_is = np.zeros(paths, dtype=np.float64)
    cash = np.zeros((paths, k), dtype=np.float64)
    arrival = float(prices @ q.sum(axis=0))
    for idx in range(k):
        level = level + q[idx] @ gamma + noise[:, idx, :]
        exec_price = level + q[idx] @ eta / t
        spend = exec_price * q[idx]
        realized_is += spend.sum(axis=1)
        cash[:, idx] = spend.sum(axis=1)
    realized_is -= arrival
    cash = np.cumsum(cash, axis=1)
    peaks = np.maximum(0.0, cash.max(axis=1))
    return {
        "realized_is": realized_is,
        "cash_paths": cash,
        "peak_drawdowns": peaks,
    }
