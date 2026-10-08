"""Conformal risk-averse decision making with OCE risk control.

Implements the single-level framework of Farzaneh & Simeone (2026),
arXiv:2608.28179 (abstract + full HTML fetched and verified 2026-09-30).  An
agent observes features ``X``, selects an action ``a(X)``, and incurs a loss
``ell(a(X), Y)`` under an unknown state ``Y``.  Risk is measured via the
optimized certainty equivalent (OCE),

    OCE_rho(L) = min_t { t + E[rho(L - t)] },               (paper Eq. 1)

for a convex, increasing penalty ``rho``: a deterministic reserve ``t`` plus
an expected penalty on the residual.  Sign convention: the classical
Ben-Tal & Teboulle OCE is a SUP over reserves under a concave utility; the
paper (and this module) uses the equivalent loss-side MIN under a convex
penalty, ``u(z) = -rho(-z)``.  The family includes (paper Sec. II):

* CVaR at tail level ``alpha``: ``rho(z) = z^+ / alpha``, giving
  ``OCE = CVaR_alpha(L)`` (Rockafellar-Uryasev).  ``alpha`` is the mass of
  the WORST tail (paper convention) — the complement of the confidence
  ``alpha`` used by ``quant_fund.metrics.spectral_risk.expected_shortfall_srm``.
* Smooth CVaR (sCVaR): ``rho(z) = (tau/alpha) * ln(1 + e^{z/tau})``,
  ``tau > 0``; recovers CVaR as ``tau -> 0+``.
* Mean-variance: ``rho(z) = z + beta z^2``, giving
  ``OCE = E[L] + beta Var(L)``.
* Entropic: the paper's intro cites entropic risk as an OCE special case
  (via Ben-Tal & Teboulle) without spelling out the penalty; this module pins
  the standard exponential penalty ``rho(z) = (e^{theta z} - 1) / theta``,
  for which ``OCE = (1/theta) log E[e^{theta L}]`` — exactly
  ``quant_fund.metrics.entropic_risk.entropic_risk_measure`` (test-pinned).

Known distribution (paper Sec. III).  For a fixed reserve the problem
separates across observations (Eq. 7),

    a*(x, t) = argmin_a E_{Y|X=x}[rho(ell(a, Y) - t)],

and the reserve solves a scalar program (Eq. 8) with stationarity condition
``E[rho'(ell(a*, Y) - t*)] = 1`` (Eq. 9).  For CVaR the solution reduces to
a prediction-set route (Sec. III-B): as the sCVaR temperature vanishes the
stationarity condition hardens to ``Pr(ell(a*, Y) > t*) = alpha`` (Eq. 14),
so ``t* = VaR_alpha`` of the deployed loss, the per-observation action
minimises the expected excess loss above ``t*`` (Eq. 15), and
``C*(x) = {y : ell(a*(x), y) <= t*}`` is a prediction set with marginal
coverage ``1 - alpha`` — the operational reading of conformal-type sets:
unlike the VaR/max-min rule of Kiyani et al. (2026), which minimises the
worst loss INSIDE the set, the CVaR rule minimises the average loss OUTSIDE
it.  ``known_distribution_policy`` (reserve scan + polish) and
``cvar_prediction_set_policy`` (exact enumeration of loss atoms) implement
both routes; tests assert they agree on seeded discrete distributions.

Unknown distribution (paper Sec. IV, Algorithm 1).  Given a synthetic
likelihood model ``P_hat_{Y|X}`` and held-out i.i.d. calibration data, each
reserve ``t`` on a finite grid ``T`` is scored by

    UCB_n(t) = t + (1/n) sum_i rho(ell(a_hat(X_i, t), Y_i) - t)
               + (B(t) - lo(t)) * sqrt(log(|T| / delta) / (2 n)),  (paper Eq. 19)

a Hoeffding + union-bound upper bound, valid simultaneously for all t in T
with probability >= 1 - delta (penalties bounded in a known range; the
paper states one uniform ``[0, B]``, this module uses the per-reserve range
implied by the loss bounds and monotonicity of ``rho``, which is the same
argument tightened).  Any ``t_hat`` with ``UCB_n(t_hat) <= epsilon``
certifies ``OCE_rho(ell(a_hat(X, t_hat), Y)) <= epsilon`` with probability
>= 1 - delta (Eq. 20; for the CVaR penalty this is ``CVaR_alpha <=
epsilon``).  This is a learn-then-test style concentration argument
(Angelopoulos et al.), NOT a conformal quantile construction — the conformal
content of the paper is the prediction-set interpretation above.  When no
grid point certifies, the result is ``certified=False`` (fail-closed: no
policy is deployed without a certificate).

Multi-level connection (same authors, arXiv:2609.11524, abstract fetched and
verified 2026-09-30): the extension maximises a weighted average of loss
certificates at several outage levels, shown equivalent to optimising over
NESTED prediction sets with a dual that decouples across inputs ``x``.  The
single-level set ``C*(x)`` here is one member (level ``1 - alpha``) of such a
nested family, and the per-x action rule (Eq. 7 / Eq. 15) is the decoupled
inner solve of that dual.  Not implemented (stretch goal); documented per
the fetched abstract only.

Honesty: every number here is a proper risk score (CVaR, entropic,
mean-variance OCE) on losses — no Sharpe/Sortino/Calmar/P&L content.  The
bench helper runs a SYNTHETIC, seeded Monte-Carlo correctness experiment;
it is never market evidence.

References:
- Farzaneh, A., & Simeone, O. (2026). Conformal Risk-Averse Decision Making
  with Optimized Certainty Equivalent Risk Control. arXiv:2608.28179 —
  Eqs. (1), (7)-(9), (14)-(15), (18)-(20), Algorithm 1.
- Farzaneh, A., & Simeone, O. (2026). Risk-Averse Decision Making with
  Multi-Level Reliability Guarantees. arXiv:2609.11524 — nested prediction
  sets (documented connection only).
- Ben-Tal, A., & Teboulle, M. (2007). An old-new concept of convex risk
  measures: the optimized certainty equivalent. *Mathematical Finance*
  17(3), 449-476.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional
  value-at-risk. *Journal of Risk* 2(3), 21-41.
- Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I., & Lei, L.
  (2025). Learn then test: calibrating predictive algorithms to achieve
  risk control. *Annals of Applied Statistics* 19(2), 1641-1662.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit

Array = NDArray[np.float64]
# Action / state index arrays. Kept distinct from ``Array`` because they are
# integer label arrays used for fancy-indexing, never as float operands.
IntArray = NDArray[np.integer[Any]]

__all__ = [
    "OCE",
    "CalibrationResult",
    "KnownPolicyResult",
    "PredictionSetResult",
    "bench_oce_calibration",
    "conformal_oce_calibration",
    "cvar_prediction_set_policy",
    "known_distribution_policy",
    "optimal_policy_lp",
    "weighted_cvar",
]

PENALTY_KINDS = ("cvar", "scvar", "entropic", "mean_variance")


# --------------------------------------------------------------------------
# OCE risk measure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OCE:
    """Optimized certainty equivalent risk for a fixed penalty family.

    ``alpha`` is the CVaR/sCVaR TAIL level (mass of the worst tail, paper
    convention), ``tau`` the sCVaR temperature, ``theta`` the entropic risk
    aversion, and ``beta`` the mean-variance coefficient.  All parameters are
    validated on construction (fail-closed).
    """

    kind: str = "cvar"
    alpha: float = 0.2
    tau: float = 1e-2
    theta: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in PENALTY_KINDS:
            raise ValueError(f"kind must be one of {PENALTY_KINDS}, got {self.kind!r}")
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be a finite tail level in (0, 1)")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not np.isfinite(self.theta) or self.theta <= 0.0:
            raise ValueError("theta must be finite and positive")
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")

    def rho(self, z: Array) -> Array:
        """Penalty ``rho(z)`` — convex, increasing (paper Sec. II)."""
        x = np.asarray(z, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("penalty input must be finite")
        if self.kind == "cvar":
            out = np.maximum(x, 0.0) / self.alpha
        elif self.kind == "scvar":
            # (tau/alpha) * ln(1 + e^{z/tau}), overflow-free via logaddexp.
            out = (self.tau / self.alpha) * np.logaddexp(0.0, x / self.tau)
        elif self.kind == "entropic":
            with np.errstate(over="ignore"):
                out = np.expm1(self.theta * x) / self.theta
        else:  # mean_variance
            with np.errstate(over="ignore"):
                out = x + self.beta * x * x
        if not np.isfinite(out).all():
            raise ValueError(
                f"{self.kind} penalty overflowed for the given inputs; "
                "rescale losses/parameters (fail-closed)"
            )
        return np.asarray(out, dtype=float)

    def rho_prime(self, z: Array) -> Array:
        """Derivative (or CVaR subgradient selection) of ``rho`` — Eq. (9)."""
        x = np.asarray(z, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("penalty derivative input must be finite")
        if self.kind == "cvar":
            out = (x > 0.0).astype(float) / self.alpha
        elif self.kind == "scvar":
            out = expit(x / self.tau) / self.alpha
        elif self.kind == "entropic":
            with np.errstate(over="ignore"):
                out = np.exp(self.theta * x)
        else:  # mean_variance
            out = 1.0 + 2.0 * self.beta * x
        if not np.isfinite(out).all():
            raise ValueError(
                f"{self.kind} penalty derivative overflowed for the given inputs (fail-closed)"
            )
        return np.asarray(out, dtype=float)

    def oce(self, losses: Array, min_obs: int = 5) -> float:
        """``OCE_rho(L)`` on an empirical sample via the pinned closed form.

        CVaR uses the Rockafellar-Uryasev minimiser at the empirical
        ``(1 - alpha)``-quantile (equal to ``expected_shortfall_srm`` at
        confidence ``1 - alpha``); entropic uses the log-mean-exp certainty
        equivalent (equal to ``entropic_risk_measure``); mean-variance uses
        ``mean + beta * var``; sCVaR has no closed form and falls back to
        :meth:`oce_numerical`.
        """
        arr = np.asarray(losses, dtype=float).ravel()
        if arr.size < min_obs or not np.isfinite(arr).all():
            raise ValueError(f"losses must be finite with >= {min_obs} observations")
        if self.kind == "cvar":
            return _empirical_cvar(arr, self.alpha)
        if self.kind == "entropic":
            return _log_mean_exp(arr, self.theta) / self.theta
        if self.kind == "mean_variance":
            return float(arr.mean() + self.beta * arr.var())
        return self.oce_numerical(arr)

    def oce_numerical(self, losses: Array) -> float:
        """Generic ``min_t { t + mean(rho(L - t)) }`` by bounded scalar search.

        The objective is convex in ``t`` for a fixed sample, so the bounded
        golden-section search on a padded bracket around the loss support is
        reliable.  Used for sCVaR and as an independent cross-check of the
        closed forms.
        """
        arr = np.asarray(losses, dtype=float).ravel()
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("losses must be finite and non-empty")
        lo, hi = float(arr.min()), float(arr.max())
        pad = max(1.0, 0.5 * (hi - lo))

        def obj(t: float) -> float:
            return float(t + float(np.mean(self.rho(arr - t))))

        res = minimize_scalar(
            obj, bounds=(lo - pad, hi + pad), method="bounded", options={"xatol": 1e-10}
        )
        value = float(res.fun)
        if not np.isfinite(value):
            raise ValueError("numerical OCE minimisation returned a non-finite value")
        return value


def _empirical_cvar(losses_sorted_agnostic: Array, alpha: float) -> float:
    """Rockafellar-Uryasev CVaR of the empirical distribution.

    ``t*`` is the ``k``-th order statistic with ``k = ceil(n (1 - alpha))``
    (the VaR of Eq. 3 on the empirical CDF) and the value is
    ``t* + mean((L - t*)^+) / alpha``.  Matches
    ``spectral_risk.expected_shortfall_srm(losses, alpha=1 - alpha)``.
    """
    ordered = np.sort(losses_sorted_agnostic)
    n = int(ordered.size)
    k = int(np.ceil(n * (1.0 - alpha)))
    k = min(max(k, 1), n)
    t = float(ordered[k - 1])
    return float(t + float(np.mean(np.maximum(ordered - t, 0.0))) / alpha)


def _log_mean_exp(losses: Array, theta: float) -> float:
    """Numerically stable ``log E[exp(theta L)]`` (log-sum-exp shift)."""
    a = theta * losses
    amax = float(a.max())
    return amax + float(np.log(np.mean(np.exp(a - amax))))


def weighted_cvar(values: Array, probs: Array, alpha: float) -> tuple[float, float]:
    """Exact ``(VaR_alpha, CVaR_alpha)`` of a discrete distribution.

    ``VaR_alpha = min{t : Pr(L <= t) >= 1 - alpha}`` (paper Eq. 3) and the
    CVaR is the Rockafellar-Uryasev value at that reserve.  Used for exact
    population-level theorem checks and Monte-Carlo violation tests.
    """
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    if v.size == 0 or v.size != p.size or not np.isfinite(v).all():
        raise ValueError("values must be finite, non-empty, and aligned with probs")
    if not np.isfinite(p).all() or (p < 0.0).any():
        raise ValueError("probs must be finite and non-negative")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a finite tail level in (0, 1)")
    if abs(float(p.sum()) - 1.0) > 1e-6:
        raise ValueError("probs must sum to one")
    order = np.argsort(v, kind="stable")
    vs, ps = v[order], p[order]
    cum = np.cumsum(ps)
    idx = int(np.searchsorted(cum, 1.0 - alpha - 1e-12, side="left"))
    idx = min(max(idx, 0), vs.size - 1)
    var = float(vs[idx])
    cvar = var + float(np.sum(ps * np.maximum(vs - var, 0.0))) / alpha
    return var, float(cvar)


# --------------------------------------------------------------------------
# Known-distribution optimal policy (paper Sec. III)
# --------------------------------------------------------------------------


def _check_distribution(
    loss_matrix: Array, cond_probs: Array, marginal: Array
) -> tuple[Array, Array, Array, int, int]:
    """Validate ``(loss (A,K), cond (m,K), marginal (m,))``; fail-closed."""
    loss = np.asarray(loss_matrix, dtype=float)
    cond = np.asarray(cond_probs, dtype=float)
    marg = np.asarray(marginal, dtype=float).ravel()
    if loss.ndim != 2 or not np.isfinite(loss).all() or loss.shape[0] < 1 or loss.shape[1] < 1:
        raise ValueError("loss_matrix must be a finite non-empty 2-D array (A, K)")
    n_act, n_states = loss.shape
    if cond.ndim != 2 or cond.shape[1] != n_states or not np.isfinite(cond).all():
        raise ValueError("cond_probs must be a finite (m, K) array matching loss_matrix")
    if (cond < 0.0).any() or not np.allclose(cond.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("cond_probs rows must be non-negative and sum to one")
    m = cond.shape[0]
    if marg.size != m or not np.isfinite(marg).all():
        raise ValueError("marginal must be a finite vector matching cond_probs rows")
    if (marg < 0.0).any() or abs(float(marg.sum()) - 1.0) > 1e-8:
        raise ValueError("marginal must be non-negative and sum to one")
    return loss, cond, marg, m, n_act


@dataclass(frozen=True)
class KnownPolicyResult:
    """Optimal policy under a known joint distribution (paper Eqs. 7-9).

    ``risk`` is ``g(t*) = t* + E[rho(ell(a*(X, t*), Y) - t*)]``, which for
    the optimal reserve equals ``OCE_rho`` of the deployed loss.
    ``stationarity`` is the Eq. (9) residual ``E[rho'(L* - t*)] - 1``
    (exact target 0 for differentiable penalties; informational for the
    non-smooth CVaR penalty, whose hard-coverage limit is ``exceedance``
    vs ``alpha``, Eq. 14).
    """

    actions: IntArray  # (m,) action index per observation
    reserve: float
    risk: float
    exceedance: float  # Pr(ell(a*, Y) > t*)
    stationarity: float  # E[rho'(L* - t*)] - 1
    loss_values: Array  # (m*K,) deployed loss atoms
    loss_probs: Array  # (m*K,) deployed loss probabilities


def known_distribution_policy(
    loss_matrix: Array,
    cond_probs: Array,
    marginal: Array,
    oce: OCE,
    grid: int = 1500,
) -> KnownPolicyResult:
    """Optimal OCE policy under a known ``P_{XY}`` (paper Sec. III-A).

    Scans the reserve over a fine grid unioned with the loss atoms (exact
    for CVaR, whose piecewise-linear optimum sits at an atom; grid-accurate
    for smooth penalties), applies the separated action rule of Eq. (7) at
    each candidate, and polishes the best bracket with a bounded scalar
    minimisation.  Deterministic; fail-closed on malformed inputs.
    """
    loss, cond, marg, m, _n_act = _check_distribution(loss_matrix, cond_probs, marginal)
    if int(grid) < 16:
        raise ValueError("grid must be at least 16")
    lo, hi = float(loss.min()), float(loss.max())
    pad = max(1.0, 0.5 * (hi - lo))
    cand = np.unique(np.concatenate([np.unique(loss), np.linspace(lo - pad, hi + pad, int(grid))]))

    def g_of_t(t: float) -> tuple[float, IntArray, Array]:
        pen = oce.rho(loss - t)  # (A, K)
        cost = np.asarray(cond @ pen.T, dtype=float)  # (m, A) — Eq. (7) inner expectation
        act = np.argmin(cost, axis=1)
        picked = cost[np.arange(m), act]
        return float(t + float(picked @ marg)), act, picked

    penal = oce.rho(loss[None, :, :] - cand[:, None, None])  # (T, A, K)
    cost_all = np.einsum("tak,ik->tia", penal, cond)  # (T, m, A)
    act_all = np.argmin(cost_all, axis=2)  # (T, m)
    picked_all = np.take_along_axis(cost_all, act_all[:, :, None], axis=2)[:, :, 0]
    g_all = cand + np.asarray(picked_all @ marg, dtype=float)  # (T,)
    j = int(np.argmin(g_all))  # ties resolve to the smallest reserve (cand is sorted)

    # Local polish for smooth penalties: the bracket minimum can only lower g.
    lo_b = float(cand[max(j - 1, 0)])
    hi_b = float(cand[min(j + 1, cand.size - 1)])
    best_t, best_g = float(cand[j]), float(g_all[j])
    if hi_b > lo_b:
        res = minimize_scalar(
            lambda t: g_of_t(float(t))[0],
            bounds=(lo_b, hi_b),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if np.isfinite(res.fun) and float(res.fun) < best_g:
            best_t, best_g = float(res.x), float(res.fun)

    risk, actions, _ = g_of_t(best_t)
    dep_vals = np.asarray(loss[actions, :], dtype=float).ravel()  # (m*K,)
    dep_probs = np.asarray(marg[:, None] * cond, dtype=float).ravel()
    resid = dep_vals - best_t
    exceedance = float(np.sum(dep_probs * (resid > 0.0).astype(float)))
    stationarity = float(np.sum(dep_probs * oce.rho_prime(resid)) - 1.0)
    return KnownPolicyResult(
        actions=np.asarray(actions, dtype=np.int64),
        reserve=best_t,
        risk=risk,
        exceedance=exceedance,
        stationarity=stationarity,
        loss_values=dep_vals,
        loss_probs=dep_probs,
    )


@dataclass(frozen=True)
class PredictionSetResult:
    """CVaR prediction-set route (paper Sec. III-B, Eqs. 14-15).

    ``var_alpha`` is the VaR (Eq. 3) of the deployed loss and equals
    ``reserve`` at the optimum; ``coverage = Pr(L* <= t*)`` is the marginal
    coverage of the prediction sets ``C*(x) = {y : ell(a*(x), y) <= t*}``;
    ``prediction_sets`` is the (m, K) membership mask.
    """

    actions: IntArray
    reserve: float
    risk: float
    var_alpha: float
    cvar_weighted: float
    coverage: float
    exceedance: float
    prediction_sets: NDArray[np.bool_]
    loss_values: Array
    loss_probs: Array


def cvar_prediction_set_policy(
    loss_matrix: Array,
    cond_probs: Array,
    marginal: Array,
    alpha: float,
) -> PredictionSetResult:
    """Set-based CVaR policy: exact enumeration over loss atoms.

    For CVaR the reserve optimum of ``g(t) = min_{a(.)} { t +
    E[(ell - t)^+]/alpha }`` is always attained at an atom of the loss
    matrix (for any fixed policy the Rockafellar-Uryasev minimiser is a loss
    atom, and interchanging the minima keeps the value).  This route
    therefore enumerates atoms exactly, applies the excess-loss action rule
    of Eq. (15), and returns the prediction-set quantities of Eqs. (12)-(14).
    """
    loss, cond, marg, m, _n_act = _check_distribution(loss_matrix, cond_probs, marginal)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a finite tail level in (0, 1)")
    atoms = np.unique(loss)
    excess = np.maximum(loss[None, :, :] - atoms[:, None, None], 0.0)  # (T, A, K)
    cost = np.einsum("tak,ik->tia", excess, cond)  # (T, m, A) — Eq. (15)
    act_all = np.argmin(cost, axis=2)  # (T, m)
    n_at = atoms.size
    dep = loss[act_all.reshape(-1), :].reshape(n_at, m, -1)  # (T, m, K) deployed losses
    dep_excess = np.maximum(dep - atoms[:, None, None], 0.0)
    joint = marg[:, None] * cond  # (m, K)
    g = atoms + np.einsum("tik,ik->t", dep_excess, joint) / alpha
    j = int(np.argmin(g))  # ties resolve to the smallest reserve
    t_star = float(atoms[j])
    actions = np.asarray(act_all[j], dtype=np.int64)
    dep_vals = np.asarray(dep[j], dtype=float).ravel()
    dep_probs = np.asarray(joint, dtype=float).ravel()
    var_a, cvar_w = weighted_cvar(dep_vals, dep_probs, alpha)
    coverage = float(np.sum(dep_probs * (dep_vals <= t_star).astype(float)))
    sets = np.asarray(loss[actions, :] <= t_star, dtype=bool)  # (m, K)
    return PredictionSetResult(
        actions=actions,
        reserve=t_star,
        risk=float(g[j]),
        var_alpha=var_a,
        cvar_weighted=cvar_w,
        coverage=coverage,
        exceedance=float(1.0 - coverage),
        prediction_sets=sets,
        loss_values=dep_vals,
        loss_probs=dep_probs,
    )


def optimal_policy_lp(
    loss_matrix: Array,
    cond_probs: Array,
    marginal: Array,
    oce: OCE,
    reserve: float,
) -> tuple[float, Array]:
    """Fixed-reserve policy via an LP over randomised policies (cvxpy).

    The inner objective of Eq. (7) is linear in the policy on the simplex, so
    the LP value equals the deterministic ``argmin_a`` route's expected
    penalty.  This is the cvxpy cross-check lane; conventions follow
    ``quant_fund.portfolio.wasserstein_dro`` (lazy import, CLARABEL, narrow
    except tuple, status and finiteness checks).
    """
    import cvxpy as cp

    loss, cond, marg, m, n_act = _check_distribution(loss_matrix, cond_probs, marginal)
    if not np.isfinite(reserve):
        raise ValueError("reserve must be finite")
    pen = oce.rho(loss - float(reserve))  # (A, K)
    cost = np.asarray(cond @ pen.T, dtype=float)  # (m, A)

    v = cp.Variable(m * n_act)
    sel = np.zeros((m, m * n_act))
    for i in range(m):
        sel[i, i * n_act : (i + 1) * n_act] = 1.0
    prob = cp.Problem(
        cp.Minimize(cost.ravel() @ v),
        [sel @ v == marg, v >= 0.0],
    )
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except (cp.error.SolverError, ArithmeticError, ValueError) as exc:
        # Narrow except tuple per the wasserstein_dro precedent: cvxpy raises
        # SolverError on solver failure; numpy raises LinAlgError (an
        # ArithmeticError) on factorization breakdowns; solvers may raise
        # ValueError on malformed problem data.  Exotic exceptions propagate.
        raise RuntimeError(f"OCE policy LP failed: solver error {exc}") from exc
    status = str(prob.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or v.value is None:
        raise RuntimeError(f"OCE policy LP failed: solver status {status}")
    vv = np.asarray(v.value, dtype=float).reshape(-1)
    if vv.size != m * n_act or not np.isfinite(vv).all():
        raise RuntimeError("OCE policy LP failed: invalid solver output")
    return float(prob.value), vv.reshape(m, n_act)


# --------------------------------------------------------------------------
# Data-driven calibration (paper Sec. IV, Algorithm 1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of Algorithm 1 on held-out calibration data.

    ``certified`` is True iff some grid reserve satisfies
    ``UCB_n(t) <= epsilon``; ``t_hat`` is then the UCB-argmin over the
    feasible reserves (Algorithm 1, line 8) and ``actions`` are the
    model-based actions ``a_hat(X_i, t_hat)`` on the calibration points.
    When nothing certifies, ``t_hat``/``ucb``/``mean_penalty``/``actions``
    are None and no policy may be deployed (fail-closed).  ``ucb_curve``,
    ``mean_penalty_curve``, ``radius_curve`` and the penalty-range curves
    are reported over the whole grid for diagnostics; ``hoeffding_radius``
    is the radius at ``t_hat`` (at the UCB-argmin when uncertified).
    """

    certified: bool
    t_hat: float | None
    ucb: float | None
    mean_penalty: float | None
    actions: IntArray | None
    grid: Array
    ucb_curve: Array
    mean_penalty_curve: Array
    radius_curve: Array
    penalty_lo_curve: Array
    penalty_hi_curve: Array
    hoeffding_radius: float
    n: int


def conformal_oce_calibration(
    loss_matrix: Array,
    model_probs: Array,
    calibration_states: IntArray,
    oce: OCE,
    reserve_grid: Array,
    epsilon: float,
    delta: float,
    penalty_range: tuple[float, float] | None = None,
) -> CalibrationResult:
    """High-probability OCE risk control from held-out data (Eq. 19-20).

    ``model_probs[i]`` is the synthetic likelihood model ``P_hat_{Y|X_i}``
    (a probability vector over the K states) for calibration point i;
    ``calibration_states[i]`` is the observed true state ``Y_i``.  For each
    reserve ``t`` on the grid the model-based policy ``a_hat(X_i, t)`` of
    Eq. (16) is computed, the realised penalties ``rho_i(t)`` of Eq. (17)
    are averaged, and the Hoeffding + union-bound certificate

        UCB_n(t) = t + mean_i rho_i(t)
                   + (hi(t) - lo(t)) * sqrt(log(|T| / delta) / (2 n))

    is formed, where ``[lo(t), hi(t)]`` bounds every penalty attainable at
    reserve ``t`` (derived from the loss bounds since ``rho`` is increasing
    — a strict tightening of the paper's single uniform constant B; a
    uniform pair can be declared via ``penalty_range``).  With
    probability >= 1 - delta over the calibration draw,
    ``OCE_rho(ell(a_hat(X, t), Y)) <= UCB_n(t)`` simultaneously for all grid
    points; selecting ``t_hat`` with ``UCB_n(t_hat) <= epsilon`` therefore
    certifies ``OCE_rho <= epsilon`` (for the CVaR penalty,
    ``CVaR_alpha <= epsilon``, paper Eq. 20).  The bound is marginal over
    calibration draws and controls the POPULATION OCE risk.

    ``penalty_range`` overrides the default range derived from the loss
    bounds and grid extremes (valid because ``rho`` is increasing).
    Fail-closed: malformed inputs, declared-range violations, and an empty
    feasible set (``certified=False``) never deploy an uncertified policy.
    """
    loss = np.asarray(loss_matrix, dtype=float)
    if loss.ndim != 2 or not np.isfinite(loss).all() or loss.shape[0] < 1 or loss.shape[1] < 1:
        raise ValueError("loss_matrix must be a finite non-empty 2-D array (A, K)")
    n_act, n_states = loss.shape
    mp = np.asarray(model_probs, dtype=float)
    if mp.ndim != 2 or mp.shape[1] != n_states or not np.isfinite(mp).all():
        raise ValueError("model_probs must be a finite (n, K) array matching loss_matrix")
    n = mp.shape[0]
    if n < 1:
        raise ValueError("model_probs must contain at least one calibration point")
    if (mp < 0.0).any() or not np.allclose(mp.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("model_probs rows must be non-negative and sum to one")
    states = np.asarray(calibration_states)
    if states.ndim != 1 or states.size != n:
        raise ValueError("calibration_states must be a 1-D array of length n")
    if not np.issubdtype(states.dtype, np.integer):
        raise ValueError("calibration_states must be integer state indices")
    si = states.astype(np.int64)
    if (si < 0).any() or (si >= n_states).any():
        raise ValueError("calibration_states must lie in [0, K)")
    gridr = np.asarray(reserve_grid, dtype=float).ravel()
    if gridr.size < 1 or not np.isfinite(gridr).all():
        raise ValueError("reserve_grid must be a finite non-empty 1-D array")
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    if penalty_range is None:
        # Per-reserve Hoeffding range: rho is increasing and every realised
        # residual lies in [loss_min - t, loss_max - t], so the penalty range
        # at reserve t is [rho(loss_min - t), rho(loss_max - t)] — a strict
        # tightening of the paper's single uniform constant B.
        lo_t = oce.rho(np.full(gridr.shape, float(loss.min())) - gridr)
        hi_t = oce.rho(np.full(gridr.shape, float(loss.max())) - gridr)
    else:
        p_lo, p_hi = float(penalty_range[0]), float(penalty_range[1])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_lo > p_hi:
            raise ValueError("penalty_range must be a finite (lo, hi) pair with lo <= hi")
        lo_t = np.full(gridr.shape, p_lo)
        hi_t = np.full(gridr.shape, p_hi)
    radius_t = np.asarray(
        (hi_t - lo_t) * np.sqrt(np.log(gridr.size / delta) / (2.0 * n)), dtype=float
    )

    penal = oce.rho(loss[None, :, :] - gridr[:, None, None])  # (T, A, K)
    exp_cost = np.einsum("tak,ik->tia", penal, mp)  # (T, n, A) — Eq. (16)
    acts = np.argmin(exp_cost, axis=2)  # (T, n)
    realised = loss[acts, si[None, :]]  # (T, n) — ell(a_hat(X_i, t), Y_i)
    pen = oce.rho(realised - gridr[:, None])  # (T, n) — Eq. (17)
    tol = 1e-9 * max(1.0, float(np.abs(hi_t).max()))
    if (pen < lo_t[:, None] - tol).any() or (pen > hi_t[:, None] + tol).any():
        raise ValueError(
            "realised penalties fall outside the declared Hoeffding range; "
            "the concentration bound does not apply (fail-closed)"
        )
    mean_pen = np.asarray(pen.mean(axis=1), dtype=float)  # (T,)
    ucb = gridr + mean_pen + radius_t  # Eq. (19)

    feasible = ucb <= epsilon
    certified = bool(feasible.any())
    t_hat: float | None = None
    ucb_hat: float | None = None
    mean_hat: float | None = None
    actions: IntArray | None = None
    if certified:
        feas_idx = np.flatnonzero(feasible)
        j = int(feas_idx[np.argmin(ucb[feas_idx])])  # Algorithm 1, line 8
    else:
        j = int(np.argmin(ucb))  # diagnostic anchor only — no policy deployed
    if certified:
        t_hat = float(gridr[j])
        ucb_hat = float(ucb[j])
        mean_hat = float(mean_pen[j])
        actions = np.asarray(acts[j], dtype=np.int64)
    return CalibrationResult(
        certified=certified,
        t_hat=t_hat,
        ucb=ucb_hat,
        mean_penalty=mean_hat,
        actions=actions,
        grid=gridr,
        ucb_curve=np.asarray(ucb, dtype=float),
        mean_penalty_curve=mean_pen,
        radius_curve=radius_t,
        penalty_lo_curve=np.asarray(lo_t, dtype=float),
        penalty_hi_curve=np.asarray(hi_t, dtype=float),
        hoeffding_radius=float(radius_t[j]),
        n=n,
    )


# --------------------------------------------------------------------------
# SYNTHETIC Monte-Carlo bench (correctness evidence, never market evidence)
# --------------------------------------------------------------------------


def _synthetic_calibration_dgp(seed: int) -> tuple[Array, Array, Array]:
    """SYNTHETIC beamforming-flavoured DGP (paper Sec. V-A analogue).

    Returns ``(loss_matrix (A,K), true_probs (G,K), model_probs (G,K))``
    with A=3 actions, K=5 states, G=2 observation groups.  The model is a
    deliberately misspecified synthetic likelihood: the digital twin believes
    the bimodal angle-error distribution is balanced (w = 0.5/0.5) while the
    truth is group-skewed, plus 10% uniform smoothing — the analogue of the
    paper's perturbed ray-tracer parameters.  Purely synthetic: no market
    data, no market claims.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, 5)  # quantised angle-error states
    offs = np.linspace(-0.9, 0.9, 3)  # beam-offset actions
    mismatch = np.abs(offs[:, None] - u[None, :])
    # Structured beam mismatch PLUS strong idiosyncratic per-(action, state)
    # fading so the tail composition — and hence the population CVaR — moves
    # with the per-trial Dirichlet draw instead of being pinned by the loss
    # matrix.
    loss = np.clip(0.30 * mismatch + rng.uniform(0.02, 0.55, size=mismatch.shape), 0.0, 0.9)
    modes = np.stack(
        [
            np.exp(-((u + 0.7) ** 2) / 0.18),
            np.exp(-((u - 0.7) ** 2) / 0.18),
        ]
    )

    def _mix(w: Array) -> Array:
        out = np.stack([w[g] * modes[0] + (1.0 - w[g]) * modes[1] for g in range(w.size)])
        return np.asarray(out / out.sum(axis=1, keepdims=True), dtype=float)

    true_probs = _mix(np.array([0.78, 0.26]))  # group-skewed truth
    model_probs = np.asarray(0.9 * _mix(np.array([0.5, 0.5])) + 0.1 / u.size, dtype=float)
    return np.asarray(loss, dtype=float), true_probs, model_probs


def _pop_cvar(
    loss: Array, group_actions: IntArray, prior: Array, true_probs: Array, alpha: float
) -> float:
    """Exact population CVaR_alpha of the deployed per-group policy."""
    vals = np.asarray(loss[group_actions, :], dtype=float).ravel()
    probs = np.asarray(prior[:, None] * true_probs, dtype=float).ravel()
    return weighted_cvar(vals, probs, alpha)[1]


def _mc_calibration_loop(
    seed: int,
    trials: int,
    n_cal: int,
    alpha: float,
    delta: float,
    epsilon: float,
    concentration: float = 1.2,
) -> dict[str, float]:
    """One Monte-Carlo sweep of Algorithm 1 over fresh calibration draws.

    Each trial perturbs the true conditional law by a seeded Dirichlet draw
    around the base DGP (the synthetic likelihood model stays fixed, so the
    model is misspecified by a trial-varying amount).  Violations are
    measured against the EXACT population CVaR of the deployed policy
    (weighted closed form), so no test-set sampling noise contaminates the
    guarantee check.  ``baseline_violation_rate`` deploys the naive
    model-greedy policy (risk-neutral expected loss under the model, no
    calibration, no certificate) on the same trials, to show the guarantee
    is not vacuous.  Seeded and deterministic.
    """
    loss, true_probs, model_probs = _synthetic_calibration_dgp(seed)
    n_groups = true_probs.shape[0]
    prior = np.full(n_groups, 1.0 / n_groups)
    oce = OCE(kind="cvar", alpha=alpha)
    grid = np.linspace(0.0, float(loss.max()), 21)
    baseline_actions = np.argmin(np.asarray(model_probs @ loss.T, dtype=float), axis=1)

    certified = 0
    violations = 0
    violations_certified = 0
    baseline_violations = 0
    plugin_certified = 0
    plugin_violations = 0
    gap_sum = 0.0
    radius = 0.0
    for trial in range(trials):
        rng = np.random.default_rng([seed + 10_000, trial])
        true_trial = np.stack(
            [rng.dirichlet(true_probs[g] * concentration) for g in range(n_groups)]
        )
        cum_true = np.cumsum(true_trial, axis=1)
        groups = rng.integers(0, n_groups, size=n_cal)
        states = (rng.random((n_cal, 1)) > cum_true[groups]).sum(axis=1)

        if _pop_cvar(loss, baseline_actions, prior, true_trial, alpha) > epsilon + 1e-9:
            baseline_violations += 1
        res = conformal_oce_calibration(
            loss, model_probs[groups], states, oce, grid, epsilon, delta
        )
        radius = float(res.radius_curve[0])  # reference radius at grid[0]; ~ 1/sqrt(n)
        # Plug-in ablation: same statistic WITHOUT the Hoeffding margin.
        plugin_ucb = res.grid + res.mean_penalty_curve
        plugin_ok = np.flatnonzero(plugin_ucb <= epsilon)
        if plugin_ok.size:
            plugin_certified += 1
            t_plugin = float(res.grid[int(plugin_ok[np.argmin(plugin_ucb[plugin_ok])])])
            pen_p = oce.rho(loss - t_plugin)
            act_p = np.argmin(np.asarray(model_probs @ pen_p.T, dtype=float), axis=1)
            if _pop_cvar(loss, act_p, prior, true_trial, alpha) > epsilon + 1e-9:
                plugin_violations += 1
        if not res.certified or res.t_hat is None:
            continue
        certified += 1
        pen = oce.rho(loss - res.t_hat)  # (A, K)
        group_actions = np.argmin(np.asarray(model_probs @ pen.T, dtype=float), axis=1)
        pop_cvar = _pop_cvar(loss, group_actions, prior, true_trial, alpha)
        if not (res.ucb is not None):
            raise ValueError("res.ucb is not None")
        gap_sum += res.ucb - pop_cvar
        if pop_cvar > epsilon + 1e-9:
            violations += 1
            violations_certified += 1
    return {
        "cert_rate": certified / trials,
        "violation_rate": violations / trials,
        "violation_rate_certified": violations_certified / max(certified, 1),
        "baseline_violation_rate": baseline_violations / trials,
        "plugin_cert_rate": plugin_certified / trials,
        "plugin_violation_rate": plugin_violations / max(plugin_certified, 1),
        "mean_ucb_gap": gap_sum / max(certified, 1),
        "hoeffding_radius": radius,
    }


def bench_oce_calibration(
    seed: int = 20260930,
    trials: int = 200,
    n_cal: int = 3000,
    n_cal_small: int = 750,
    alpha: float = 0.3,
    delta: float = 0.1,
    epsilon: float = 0.65,
    concentration: float = 1.2,
) -> dict[str, float | str]:
    """SYNTHETIC Monte-Carlo bench for the high-probability OCE control.

    Correctness evidence only — proper scores (population CVaR of losses,
    coverage of the Eq. 20 guarantee), never market evidence and never a
    performance claim.  ``violation_rate`` counts trials where the CERTIFIED
    policy's exact population CVaR exceeds ``epsilon`` (guarantee: <= delta
    in expectation); ``baseline_violation_rate`` deploys the risk-neutral
    model-greedy policy with no calibration (the uncertified contrast, cf.
    the paper's VaR baseline violating in 53.4% of trials);
    ``plugin_*`` keys ablate the Hoeffding margin.  The two calibration
    sizes expose graceful degradation: ``hoeffding_radius`` is the
    reference radius at ``grid[0]`` and scales as 1/sqrt(n), so
    ``radius_ratio`` equals ``sqrt(n_cal / n_cal_small)`` exactly.
    """
    if trials < 1 or n_cal < 2 or n_cal_small < 2:
        raise ValueError("trials must be >= 1 and calibration sizes >= 2")
    big = _mc_calibration_loop(seed, trials, n_cal, alpha, delta, epsilon, concentration)
    small = _mc_calibration_loop(seed, trials, n_cal_small, alpha, delta, epsilon, concentration)
    return {
        "dgp": "synthetic_bimodal_beam_digital_twin",
        "claim": "research_metric_only",
        "seed": float(seed),
        "alpha": float(alpha),
        "delta": float(delta),
        "epsilon": float(epsilon),
        "trials": float(trials),
        "n_cal": float(n_cal),
        "n_cal_small": float(n_cal_small),
        "cert_rate": big["cert_rate"],
        "cert_rate_small": small["cert_rate"],
        "violation_rate": big["violation_rate"],
        "violation_rate_certified": big["violation_rate_certified"],
        "violation_rate_small": small["violation_rate"],
        "baseline_violation_rate": big["baseline_violation_rate"],
        "plugin_cert_rate": big["plugin_cert_rate"],
        "plugin_violation_rate": big["plugin_violation_rate"],
        "hoeffding_radius": big["hoeffding_radius"],
        "hoeffding_radius_small": small["hoeffding_radius"],
        "radius_ratio": small["hoeffding_radius"] / big["hoeffding_radius"],
        "mean_ucb_gap": big["mean_ucb_gap"],
    }
