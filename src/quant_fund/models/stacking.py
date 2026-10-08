"""Bayesian stacking and pseudo-BMA(+) of predictive distributions (SYNTHETIC).

Combination rules for *distributional* forecasts, working on a pointwise
log predictive density matrix ``(N, K)`` whose row ``i`` holds
``log p_k(y_i)`` for each candidate model ``k``. Rows should be
out-of-sample — leave-one-out (LOO) or K-fold (e.g. the purged CV utilities
in ``quant_fund.metrics.purged_cv``); the matrix itself is the contract and
this module never refits models.

References:
- Yao, Vehtari, Simpson & Gelman (2018). Using stacking to average Bayesian
  predictive distributions. *Bayesian Analysis* 13(1):91–107,
  arXiv:1704.02030, https://doi.org/10.1214/17-BA1091 — ``stacking_weights``
  maximizes ``sum_i log(sum_k w_k p_k(y_i))`` over the simplex; the objective
  is concave (log of a linear function of ``w``), so multi-start SLSQP guards
  only against solver stalls, not local optima.
- Vehtari, Gelman & Gabry (2017). Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC. *Statistics and Computing*
  27(5):1413–1432, https://doi.org/10.1007/s11222-016-9696-4 — ELPD, the
  input to ``pseudo_bma_weights``.
- Clyde, Ghosh & Littman (2011). Bayesian adaptive sampling for predictive
  composition. *JASA* 106(494):801–816,
  https://doi.org/10.1198/jasa.2011.tm10581 — the Bayesian-bootstrap
  regularization behind pseudo-BMA+ (``bb=True``).
- Gneiting & Raftery (2007). Strictly proper scoring rules, calibration, and
  forecast sharpness. *JASA* 102(477):359–378,
  https://doi.org/10.1198/016214506000001437 — log score and CRPS are proper
  scores; ``stacked_log_score`` / ``stacked_crps`` reuse them (the CRPS
  Riemann-sum helper is imported from ``quant_fund.metrics.scoring``).

All evaluation is on proper scores (lab honesty contract) — never
Sharpe/P&L headlines. Fail-closed: empty inputs, non-finite entries,
negative or all-zero weights, crossing quantile grids, and shape mismatches
raise ``ValueError`` rather than returning a silent identity. No
live-trading or market-evidence claims.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy.special import logsumexp

from quant_fund.metrics.scoring import crps_from_quantiles

Array = NDArray[np.float64]

__all__ = [
    "stacking_weights",
    "pseudo_bma_weights",
    "stacked_quantiles",
    "stacked_cdf",
    "stacked_predictive_samples",
    "mixture_log_density",
    "stacked_log_score",
    "stacked_crps",
    "gaussian_log_dens_matrix",
]

_LOG_ZERO = -np.inf
_EXP_CAP = 700.0  # exp() overflow guard for pathological weight/density scales
_VERTEX_EPS = 0.02  # near-vertex starts keep all components strictly feasible


# ---------------------------------------------------------------------------
# validation (fail-closed)
# ---------------------------------------------------------------------------


def _as_log_dens(log_dens: Array, name: str = "log_dens") -> Array:
    m = np.asarray(log_dens, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2d (N, K) matrix of log predictive densities")
    if m.shape[0] == 0 or m.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty (got shape {m.shape})")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be finite — no NaN/inf log densities")
    return m


def _as_simplex_weights(weights: Array, k: int, name: str = "weights") -> Array:
    """Validate and normalize to the K-simplex; never silently repair."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != k:
        raise ValueError(f"{name} must have length K={k}, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must be finite")
    if np.any(w < 0.0):
        raise ValueError(f"{name} must be non-negative (simplex membership)")
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"{name} must have positive total mass — all-zero is degenerate")
    return np.asarray(w / s, dtype=float)


def _validate_prob_grid(levels: Array, name: str = "levels") -> Array:
    lv = np.asarray(levels, dtype=float).reshape(-1)
    if lv.size < 2:
        raise ValueError(f"{name} must have at least 2 probability levels")
    if not np.all(np.isfinite(lv)):
        raise ValueError(f"{name} must be finite")
    if np.any((lv <= 0.0) | (lv >= 1.0)):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    if np.any(np.diff(lv) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return lv


def _validate_taus(taus: Array, levels: Array) -> Array:
    tv = np.asarray(taus, dtype=float).reshape(-1)
    if tv.size == 0:
        raise ValueError("taus must be non-empty")
    if not np.all(np.isfinite(tv)):
        raise ValueError("taus must be finite")
    if np.any(np.diff(tv) <= 0.0):
        raise ValueError("taus must be strictly increasing")
    if tv[0] < levels[0] or tv[-1] > levels[-1]:
        raise ValueError(
            "taus must lie inside [levels[0], levels[-1]] — the stacked CDF is "
            "only identified on the components' common probability grid"
        )
    return tv


def _require_non_crossing(q: Array) -> None:
    if np.any(np.diff(q, axis=-1) < 0.0):
        raise ValueError(
            "component quantile grids must be non-crossing (nondecreasing "
            "in the level direction) — fix upstream before stacking"
        )


# ---------------------------------------------------------------------------
# stacking (Yao et al. 2018)
# ---------------------------------------------------------------------------


def _stacking_obj_grad(ld: Array, w: Array) -> tuple[float, Array]:
    """Negative stacked log score and its gradient, in log space.

    Objective ``-sum_i log(sum_k w_k exp(l_ik))`` via ``logsumexp``; the
    gradient is ``-sum_i exp(l_ik - lse_i)`` — well defined even when some
    ``w_k = 0`` (the simplex boundary is where stacking's sparsity lives).
    """
    with np.errstate(divide="ignore"):
        log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), _LOG_ZERO)
    lse = np.asarray(logsumexp(ld + log_w[None, :], axis=1), dtype=float)
    if not np.all(np.isfinite(lse)):
        # Only reachable for infeasible all-zero w; penalize instead of NaN.
        return 1e18, np.zeros(ld.shape[1])
    val = -float(np.sum(lse))
    expo = np.minimum(ld - lse[:, None], _EXP_CAP)
    grad = -np.asarray(np.exp(expo).sum(axis=0), dtype=float)
    return val, grad


def _weight_starts(k: int, n_starts: int, seed: int) -> list[Array]:
    """Deterministic multi-start set: uniform, near-vertices, seeded Dirichlet.

    The stacking objective is concave, so starts guard against solver stalls
    rather than local optima; the uniform + near-vertex block is always tried,
    and ``n_starts`` beyond ``1 + K`` adds ``Dirichlet(1, ..., 1)`` draws from
    a pinned rng (determinism for a given seed).
    """
    starts: list[Array] = [np.full(k, 1.0 / k)]
    for j in range(k):
        w = np.full(k, _VERTEX_EPS / k)
        w[j] = 1.0 - _VERTEX_EPS + _VERTEX_EPS / k
        starts.append(w)
    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        starts.append(np.asarray(rng.dirichlet(np.ones(k)), dtype=float))
    return starts[: max(n_starts, 1 + k)]


def stacking_weights(log_dens: Array, *, n_starts: int = 8, seed: int = 0) -> Array:
    """Stacking weights over the simplex (Yao, Vehtari, Simpson & Gelman 2018).

    Maximizes ``sum_i log(sum_k w_k p_k(y_i))`` — the out-of-sample log
    predictive density of the weighted mixture — given the pointwise log
    density matrix ``log_dens`` of shape ``(N, K)`` (rows ``log p_k(y_i)``,
    typically LOO or K-fold). Optimization is SLSQP on ``{w >= 0, sum w = 1}``
    with analytic gradient, run from multiple deterministic starts (uniform,
    near-vertices, then seeded Dirichlet draws up to ``n_starts``); the best
    converged solution wins. ``K = 1`` returns ``[1.0]`` without optimizing.

    Returns weights on the simplex (``w >= 0``, ``sum w = 1``); components
    that do not improve the mixture get exactly ``0`` — stacking's sparsity.
    Raises ``ValueError`` on empty/non-finite input or if every start fails
    to converge (fail-closed — never returns degenerate weights).
    """
    if isinstance(n_starts, bool) or not isinstance(n_starts, int) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    ld = _as_log_dens(log_dens)
    k = ld.shape[1]
    if k == 1:
        return np.ones(1)

    def _neg_obj(w: Array) -> float:
        val, _ = _stacking_obj_grad(ld, w)
        return val

    def _jac(w: Array) -> Array:
        _, g = _stacking_obj_grad(ld, w)
        return g

    best_w: Array | None = None
    best_val = np.inf
    for x0 in _weight_starts(k, n_starts, seed):
        res = opt.minimize(
            _neg_obj,
            x0,
            jac=_jac,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
        s = float(w.sum())
        if s <= 0.0:
            continue
        w = np.asarray(w / s, dtype=float)
        val, _ = _stacking_obj_grad(ld, w)
        # Tie-break margin: on a flat objective (e.g. identical components)
        # solver noise ~1e-12·N must not dethrone the earlier, more regular
        # start — prefer uniform unless a start is strictly better.
        if np.isfinite(val) and val < best_val - 1e-9:
            best_val, best_w = val, w
    if best_w is None:
        raise ValueError(
            "stacking optimization failed to converge from every start — "
            "refusing to return degenerate weights"
        )
    return best_w


# ---------------------------------------------------------------------------
# pseudo-BMA / pseudo-BMA+ (Yao et al. 2018 sec. 4; Clyde et al. 2011)
# ---------------------------------------------------------------------------


def _softmax(x: Array) -> Array:
    e = np.exp(x - float(np.max(x)))
    return np.asarray(e / e.sum(), dtype=float)


def pseudo_bma_weights(
    log_dens: Array, *, bb: bool = True, n_boot: int = 1000, seed: int = 0
) -> Array:
    """pseudo-BMA / pseudo-BMA+ weights (Yao et al. 2018; Clyde et al. 2011).

    With ``bb=False`` (plain pseudo-BMA): ``w_k ∝ exp(ELPD_k)`` where
    ``ELPD_k = sum_i log p_k(y_i)`` — winner-take-all when ELPD gaps are
    large, since the softmax of a sum over ``N`` rows is extremely sharp.

    With ``bb=True`` (pseudo-BMA+, default): Bayesian-bootstrap
    regularization — ``n_boot`` replicates of ``Dirichlet(1, ..., 1)``
    observation weights ``M^b`` produce per-replicate softmax weights
    ``w^b = softmax(sum_i M^b_i l_i·)`` (the BB posterior expectation of the
    log marginal likelihood, Clyde et al. 2011), averaged over replicates.
    The result stays on the simplex but keeps non-negligible mass on
    near-tied models. Determinism is pinned by ``seed``.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    ld = _as_log_dens(log_dens)
    n = ld.shape[0]
    elpd = np.asarray(ld.sum(axis=0), dtype=float)
    if not bb:
        return _softmax(elpd)
    rng = np.random.default_rng(seed)
    m = np.asarray(rng.dirichlet(np.ones(n), size=n_boot), dtype=float)  # (B, N)
    e = m @ ld  # (B, K) bootstrapped ELPD draws
    w = np.exp(e - e.max(axis=1, keepdims=True))
    w = np.asarray(w / w.sum(axis=1, keepdims=True), dtype=float)
    return np.asarray(w.mean(axis=0), dtype=float)


# ---------------------------------------------------------------------------
# stacked predictive distribution
# ---------------------------------------------------------------------------


def _stacked_quantiles_row(
    q_row: Array, w: Array, levels: Array, taus: Array, n_grid: int
) -> Array:
    """Invert the mixture CDF ``F(z) = sum_k w_k F_k(z)`` for one ``(K, Q)`` row."""
    lo = float(q_row.min())
    hi = float(q_row.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(
            "component quantile grid is degenerate (all values equal) — the "
            "stacked quantile function is unidentified"
        )
    z = np.linspace(lo, hi, n_grid)
    cdf = np.zeros(n_grid)
    for k in range(q_row.shape[0]):
        cdf += w[k] * np.interp(z, q_row[k], levels)
    cdf = np.maximum.accumulate(cdf)  # numerical monotonicity for the inversion
    return np.asarray(np.interp(taus, cdf, z), dtype=float)


def stacked_quantiles(
    quantiles: Array,
    weights: Array,
    levels: Array,
    taus: Array,
    *,
    n_grid: int = 1001,
) -> Array:
    """Quantiles of the stacked (mixture) predictive distribution.

    ``quantiles`` is ``(K, Q)`` — one non-crossing quantile grid per
    component on the common strictly increasing probability grid ``levels``
    — or batched ``(N, K, Q)``. Each component CDF ``F_k`` is the linear
    interpolation of its grid; the stacked CDF ``F = sum_k w_k F_k`` is
    evaluated on an ``n_grid``-point z-grid spanning the components' range
    and inverted at ``taus`` (linear interpolation, monotone by
    construction). ``taus`` must lie inside ``[levels[0], levels[-1]]`` —
    outside the common grid the mixture CDF is unidentified (fail-closed, no
    tail extrapolation). Returns shape ``(T,)`` for a ``(K, Q)`` input and
    ``(N, T)`` for ``(N, K, Q)``.
    """
    if isinstance(n_grid, bool) or not isinstance(n_grid, int) or n_grid < 65:
        raise ValueError("n_grid must be an integer >= 65")
    q = np.asarray(quantiles, dtype=float)
    lv = _validate_prob_grid(levels)
    tv = _validate_taus(taus, lv)
    if q.ndim == 2:
        k, nq = q.shape
        if nq != lv.size:
            raise ValueError(f"quantiles must have Q={lv.size} columns matching levels")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantiles must be finite")
        _require_non_crossing(q)
        w = _as_simplex_weights(weights, k)
        return _stacked_quantiles_row(q, w, lv, tv, n_grid)
    if q.ndim == 3:
        n, k, nq = q.shape
        if n == 0:
            raise ValueError("quantiles must be non-empty")
        if nq != lv.size:
            raise ValueError(f"quantiles must have Q={lv.size} columns matching levels")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantiles must be finite")
        _require_non_crossing(q)
        w = _as_simplex_weights(weights, k)
        out = np.empty((n, tv.size))
        for i in range(n):
            out[i] = _stacked_quantiles_row(q[i], w, lv, tv, n_grid)
        return out
    raise ValueError("quantiles must be (K, Q) or (N, K, Q)")


def stacked_cdf(z: Array, quantiles: Array, weights: Array, levels: Array) -> Array:
    """Stacked mixture CDF ``F(z) = sum_k w_k F_k(z)`` from ``(K, Q)`` grids.

    Each component CDF is the linear interpolation of its quantile grid:
    ``levels[0]`` below the lowest knot and ``levels[-1]`` above the highest
    (the grid's identified range — no tail extrapolation). ``z`` may be any
    finite non-empty vector; the output is monotone in ``z``.
    """
    zz = np.asarray(z, dtype=float).reshape(-1)
    if zz.size == 0:
        raise ValueError("z must be non-empty")
    if not np.all(np.isfinite(zz)):
        raise ValueError("z must be finite")
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2:
        raise ValueError("quantiles must be (K, Q)")
    lv = _validate_prob_grid(levels)
    if q.shape[1] != lv.size:
        raise ValueError(f"quantiles must have Q={lv.size} columns matching levels")
    if not np.all(np.isfinite(q)):
        raise ValueError("quantiles must be finite")
    _require_non_crossing(q)
    w = _as_simplex_weights(weights, q.shape[0])
    out = np.zeros(zz.size)
    for k in range(q.shape[0]):
        out += w[k] * np.interp(zz, q[k], lv)
    return np.asarray(out, dtype=float)


def stacked_predictive_samples(
    samples: Array, weights: Array, *, n_draws: int, seed: int = 0
) -> Array:
    """I.i.d. draws from the stacked mixture given component sample matrices.

    ``samples`` is ``(S, K)``: column ``k`` holds ``S`` draws from component
    ``k``'s predictive. Each output draw selects a component with
    probability ``w_k`` and a uniformly random row — exact draws from
    ``sum_k w_k F_k`` when the columns are i.i.d. per component. Determinism
    is pinned by ``seed``.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("samples must be a non-empty (S, K) matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    w = _as_simplex_weights(weights, x.shape[1])
    rng = np.random.default_rng(seed)
    comps = np.asarray(rng.choice(x.shape[1], size=n_draws, p=w))
    rows = np.asarray(rng.integers(0, x.shape[0], size=n_draws))
    return np.asarray(x[rows, comps], dtype=float)


# ---------------------------------------------------------------------------
# proper-score evaluation helpers (log score / CRPS)
# ---------------------------------------------------------------------------


def mixture_log_density(log_dens: Array, weights: Array) -> Array:
    """Pointwise log density of the stacked mixture (Yao et al. 2018, eq. 1).

    ``log(sum_k w_k p_k(y_i))`` computed stably via ``logsumexp``; zero-weight
    components contribute exactly nothing (``log 0 = -inf`` masked inside
    ``logsumexp``). Input ``(N, K)`` → output ``(N,)``.
    """
    ld = _as_log_dens(log_dens)
    w = _as_simplex_weights(weights, ld.shape[1])
    with np.errstate(divide="ignore"):
        log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), _LOG_ZERO)
    return np.asarray(logsumexp(ld + log_w[None, :], axis=1), dtype=float)


def stacked_log_score(log_dens: Array, weights: Array) -> float:
    """Mean stacked log predictive density — proper score (Gneiting–Raftery).

    This is ``ELPD_stacking / N``; higher is better. Research-diagnostic
    only — never a live-trading or market-evidence claim.
    """
    return float(np.mean(mixture_log_density(log_dens, weights)))


def stacked_crps(
    y: Array,
    quantiles: Array,
    weights: Array,
    levels: Array,
    taus: Array | None = None,
) -> float:
    """CRPS of the stacked predictive from ``(N, K, Q)`` component grids.

    Builds the stacked quantile function per row (``stacked_quantiles``) and
    integrates ``2 * pinball`` over ``taus`` via ``crps_from_quantiles``
    (Gneiting–Raftery Riemann-sum CRPS). ``taus`` defaults to ``levels``.
    Proper score — lower is better; research-diagnostic only.
    """
    v = np.asarray(y, dtype=float).reshape(-1)
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 3:
        raise ValueError("quantiles must be (N, K, Q) for stacked_crps")
    if v.size == 0:
        raise ValueError("y must be non-empty")
    if v.size != q.shape[0]:
        raise ValueError(f"length mismatch: y={v.size}, quantiles rows={q.shape[0]}")
    if not np.all(np.isfinite(v)):
        raise ValueError("y must be finite")
    lv = _validate_prob_grid(levels)
    tv = lv if taus is None else _validate_taus(taus, lv)
    sq = stacked_quantiles(q, weights, lv, tv)
    return float(crps_from_quantiles(v, sq, tv))


def gaussian_log_dens_matrix(y: Array, mu: Array, sigma: Array) -> Array:
    """Build the ``(N, K)`` pointwise log-density matrix for Gaussian forecasts.

    Convenience for the common case where each component's predictive is
    ``N(mu_ik, sigma_ik^2)``: row ``i`` holds ``log p_k(y_i)``. ``mu`` and
    ``sigma`` are ``(N, K)``; ``sigma`` must be strictly positive
    (fail-closed). For genuine LOO stacking the ``(mu, sigma)`` rows must be
    out-of-sample predictions — this function only evaluates densities.
    """
    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("y must be non-empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("y must be finite")
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if m.ndim != 2 or m.shape[0] != v.size:
        raise ValueError(f"mu must be a (N={v.size}, K) matrix matching y")
    if s.shape != m.shape:
        raise ValueError("sigma must have the same shape as mu")
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(s)):
        raise ValueError("mu and sigma must be finite")
    if np.any(s <= 0.0):
        raise ValueError("sigma must be positive")
    z = (v[:, None] - m) / s
    return np.asarray(-0.5 * np.log(2.0 * np.pi) - np.log(s) - 0.5 * z * z, dtype=float)
