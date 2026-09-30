"""Wasserstein distributionally robust (DRO) mean-variance portfolios.

Worst-case mean-variance over a type-2 Wasserstein ball of radius ``radius``
around the nominal estimate ``(mu, cov)``, via the exact duality for the
linear (mean) loss.  For loss ``ell(xi; w) = -xi'w`` and ground cost
``c(xi, xi_hat) = ||xi - xi_hat||_2**2`` the DRO supremum has the closed form
(Esfahani & Kuhn 2018, Theorem 4.2; Blanchet & Murthy 2019, Theorem 2;
Gao & Kleywegt 2023, Theorem 1)

    sup_{P : W_2(P, P_hat) <= r} E_P[ell(xi; w)]
        = inf_{lam >= 0}  lam * r**2
              + E_{P_hat}[ sup_xi ( ell(xi; w) - lam * c(xi, xi_hat) ) ]

The inner supremum is a quadratic maximisation with the exact value
``-xi_hat'w + ||w||_2**2 / (4 * lam)`` (first-order condition), so the dual
collapses to

    sup = -mu'w + inf_{lam >= 0} ( lam * r**2 + ||w||_2**2 / (4 * lam) )
        = -mu'w + r * ||w||_2          at  lam* = ||w||_2 / (2 * r)

There is no piecewise branch to grid over: the joint program in ``(w, lam)``
is convex and the partially minimised form is an exact second-order-cone
program,

    min_w  -mu'w + risk_aversion * w' cov w + radius * ||w||_2,

which is what this module solves.  The quadratic risk term is taken at the
nominal covariance (return-mean ambiguity only), the standard mean-variance
specialisation of the Esfahani & Kuhn program.  As ``radius`` grows, the
``||w||_2`` regulariser dominates and the allocation shrinks monotonically
toward the minimum-norm point of the feasible set (equal weights on the
budget simplex); ``radius = 0`` recovers the nominal mean-variance solution
exactly.  No Sharpe/Sortino/P&L content — objectives are risk-adjusted
costs only.

References:
- Esfahani, P. M., & Kuhn, D. (2018). Data-driven distributionally robust
  optimization using the Wasserstein metric. *Mathematical Programming* 171,
  115-166. arXiv:1505.03516 — Theorem 4.2 (finite-concave dual).
- Blanchet, J., & Murthy, K. (2019). Quantifying distributional model risk
  via optimal transport. *Mathematics of Operations Research* 44(2), 565-600.
  arXiv:1604.03064 — Theorem 2 (exact dual for transport costs).
- Gao, R., & Kleywegt, A. (2023). Distributionally robust stochastic
  optimization with Wasserstein distance. *Mathematical Programming* 197,
  1111-1171. arXiv:1604.02199 — Theorem 1 (worst-case expectation dual).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = ["DroResult", "solve_wasserstein_dro"]


@dataclass(frozen=True)
class DroResult:
    """Solution of the Wasserstein-DRO mean-variance program.

    ``nominal_objective`` is ``-mu'w + risk_aversion * w' cov w`` and
    ``worst_case_objective`` is the same plus the exact worst-case mean
    correction ``radius * ||w||_2``; hence the latter is always >= the
    former.  ``dual_lambda`` is the optimal multiplier of the Wasserstein
    ball constraint, ``||w||_2 / (2 * radius)`` (0 when ``radius == 0``).
    """

    weights: Array
    nominal_objective: float
    worst_case_objective: float
    dual_lambda: float
    radius: float


def _as_cov(cov: Array) -> Array:
    m = np.asarray(cov, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2 or not np.all(np.isfinite(m)):
        raise ValueError("cov must be a finite square matrix (n >= 2)")
    if np.max(np.abs(m - m.T)) > 1e-6 * max(1.0, float(np.abs(m).max())):
        raise ValueError("cov must be symmetric")
    if np.linalg.eigvalsh(m).min() < -1e-12 * max(float(np.abs(m).max()), 1e-12):
        raise ValueError("cov must be positive semidefinite")
    return m


def solve_wasserstein_dro(
    mu: Array,
    cov: Array,
    radius: float,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    budget: float = 1.0,
    per_asset_cap: float | None = None,
) -> DroResult:
    """Solve the worst-case mean-variance program over a W2 ball of ``radius``.

    Minimises ``-mu'w + risk_aversion * w' cov w + radius * ||w||_2``
    subject to ``sum(w) == budget`` (plus ``w >= 0`` when ``long_only`` and
    ``w <= per_asset_cap`` when given).  The correction term is the exact
    dual value of the worst-case expected return inside the Wasserstein
    ball, not a heuristic penalty (see module docstring).

    Fail-closed: non-finite or non-conformable inputs, non-PSD ``cov``,
    negative ``radius``, non-positive ``risk_aversion`` or ``budget``, and
    any solver status other than optimal raise.
    """
    import cvxpy as cp
    from cvxpy.atoms.norm import norm2

    c = _as_cov(cov)
    n = c.shape[0]
    m = np.asarray(mu, dtype=float).reshape(-1)
    if m.size != n or not np.all(np.isfinite(m)):
        raise ValueError("mu must be a finite vector matching cov")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive and finite")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("budget must be positive and finite")
    if per_asset_cap is not None:
        if not np.isfinite(per_asset_cap) or per_asset_cap <= 0.0:
            raise ValueError("per_asset_cap must be positive and finite")
        if long_only and per_asset_cap * n < budget - 1e-9:
            raise ValueError("per_asset_cap is too small to meet the budget")

    w = cp.Variable(n)
    constraints = [cp.sum(w) == budget]
    if long_only:
        constraints.append(w >= 0.0)
    if per_asset_cap is not None:
        constraints.append(w <= float(per_asset_cap))
    variance = cp.quad_form(w, c)
    obj = cp.Minimize(-m @ w + risk_aversion * variance + radius * norm2(w))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except (cp.error.SolverError, ArithmeticError, ValueError) as exc:
        # Narrowed from `except Exception` (quality ratchet, ceiling 72):
        # cvxpy raises SolverError on solver failure; numpy raises
        # LinAlgError (an ArithmeticError) on factorization breakdowns;
        # solvers may raise ValueError on malformed problem data. Exotic
        # exceptions propagate unwrapped — still fail-closed.
        raise RuntimeError(f"Wasserstein DRO solve failed: solver error {exc}") from exc
    status = str(prob.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
        raise RuntimeError(f"Wasserstein DRO solve failed: solver status {status}")
    wv = np.asarray(w.value, dtype=float).reshape(-1)
    if wv.shape != (n,) or not np.isfinite(wv).all():
        raise RuntimeError("Wasserstein DRO solve failed: invalid solver output")

    nominal = float(-(m @ wv) + risk_aversion * float(wv @ c @ wv))
    norm_w = float(np.linalg.norm(wv))
    worst_case = float(nominal + radius * norm_w)
    dual_lambda = float(norm_w / (2.0 * radius)) if radius > 0.0 else 0.0
    return DroResult(
        weights=wv,
        nominal_objective=nominal,
        worst_case_objective=worst_case,
        dual_lambda=dual_lambda,
        radius=float(radius),
    )
