"""Longstaff-Schwartz American Monte Carlo with an Andersen-Broadie dual bound.

Primal side (lower bound): Longstaff-Schwartz least-squares Monte Carlo (LSM).
A backward recursion on training paths regresses the discounted realised
continuation cashflow on polynomial basis functions of the state at each
exercise date, restricted to in-the-money paths (LS01, section 3).  The fitted
policy is then frozen and applied forward to an independent path set; the
out-of-sample stopped-payoff mean is the standard low-bias LSM estimator (the
in-sample estimator is biased high by the look-ahead in the regression fit).

Dual side (upper bound): Andersen-Broadie primal-dual, in the Rogers /
Haugh-Kogan dual representation, using a regression-based envelope.  With
``f_reg(t_i, .)`` the all-paths continuation regressions carried in the frozen
policy (clipped to ``[0, cap_i]``, the realised training continuation range,
to bound polynomial tail extrapolation), define the discounted envelope on
the primary paths

    g_i = max( e^{-r t_i} h(t_i, X_i),  e^{-r t_i} f_reg(t_i, X_i) ),

and estimate each conditional expectation ``E[g_{i+1} | F_{t_i}]`` by a nested
one-step sub-simulation: ``n_sub`` sub-paths per primary path per exercise
date, stepped from ``t_i`` to ``t_{i+1}`` and scored with the same envelope
function ``g_{i+1}`` (the AB04 inner simulation; t=0 also gets an independent
per-path batch so no sub-simulation noise is common-mode across primaries).
The dual estimator is

    upper = E[ max_i ( e^{-r t_i} h(t_i, X_i) - M_i ) ],
    M_i   = sum_{j<=i} ( g_j - E_hat[g_j | F_{t_{j-1}}] ),

the Doob decomposition of the envelope process.  As ``n_sub -> inf`` the
increments are exact conditional-mean-zero martingale increments, so the
estimator converges to the Rogers dual bound of the envelope process, which
dominates the optimal American price for ANY adapted envelope (Rogers 2002;
Haugh-Kogan 2004); tightness tracks how well ``f_reg`` approximates the true
continuation value.  At finite ``n_sub`` the only bias is the positive Jensen
bias of the pathwise max, O(1/n_sub), so the estimate sits above its limit
and the duality gap shrinks as ``n_sub`` grows (Andersen-Broadie 2004,
sections 3-4).  Sub-simulation streams are prefix-nested: for fixed seed,
primary path count and dates, a call with ``n_sub = M'`` consumes exactly the
first ``M'`` sub-paths of every per-date stream of a call with ``n_sub =
M > M'``, so bounds at different budgets are paired and comparable.

References
----------
1. Longstaff, F. A., Schwartz, E. S. (2001), "Valuing American options by
   simulation: A simple least-squares approach", *Review of Financial
   Studies* 14(1), 113-147. doi:10.1093/rfs/14.1.113
2. Andersen, L., Broadie, M. (2004), "Primal-dual simulation algorithm for
   pricing multidimensional American options", *Management Science* 50(9),
   1222-1234. doi:10.1287/mnsc.1040.0258
3. Rogers, L. C. G. (2002), "Monte Carlo valuation of American options",
   *Mathematical Finance* 12(3), 271-286. doi:10.1111/1467-9965.02010
4. Haugh, M., Kogan, L. (2004), "Pricing American options: A duality
   approach", *Operations Research* 52(2), 258-270. doi:10.1287/opre.1030.0070
5. Barone-Adesi, G., Whaley, R. (1987), "Efficient analytic approximation of
   American option values", *Journal of Finance* 42(2), 301-320 — the BAW
   benchmark (:mod:`quant_fund.models.american_baw`) used for validation here.

Numerics
--------
- GBM is sampled exactly between exercise dates (the log-Euler step is exact
  for GBM), so discretisation bias comes only from the Bermudan exercise-date
  grid and the regression, not from the diffusion scheme.
- Basis: full polynomial tensor of total degree <= ``degree`` (default 3) in
  the state scaled by ``basis_scale`` (pass the strike), constant included:
  4 terms in 1-d, 10 terms in 2-d at degree 3.  Regressions use ITM paths
  only and are solved with ``np.linalg.lstsq``; a date whose ITM path count
  is below basis size + 1 is skipped (policy = continue) and recorded.
- Dual sub-simulation budget: ``n_sub * n_paths * (n_dates - 1)`` one-step
  sub-paths in total (every date gets an independent per-primary-path batch,
  including t=0, so no sub-simulation noise is common-mode across primary
  paths), each scored with the regression envelope at its landing date.
- Fail-closed: invalid parameters, non-PSD correlation, degenerate payoffs
  and non-finite results all raise.  Everything is seeded via
  ``np.random.SeedSequence`` spawn keys (train=1, oos=2, dual primaries=0,
  dual sub-sims=(1000+date, step), benchmark helpers >=900) and deterministic
  across calls on the same platform.

SYNTHETIC research only — correctness benchmarks against known answers (BS
closed form, BAW approximation); no market evidence, no live-trading claims.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models.american_baw import baw_american

Array = NDArray[np.float64]
PayoffFn = Callable[[Array, float], Array]
# Scalar / sequence / array market parameters (s0, q, sigma).
FloatSeq = float | Sequence[float] | Array

__all__ = [
    "DualResult",
    "LSMPolicy",
    "LSMResult",
    "american_lsm_benchmarks",
    "basis_size",
    "bs_american_put_bench",
    "call_payoff",
    "deep_itm_boundary_bench",
    "dual_upper_bound",
    "european_limit_bench",
    "lsm_american_price",
    "max_call_2asset_bench",
    "max_call_payoff",
    "polynomial_basis",
    "put_payoff",
    "simulate_gbm",
]

# RNG spawn keys (documented stream budget; keep disjoint).
_STREAM_DUAL_PRIMARY = 0
_STREAM_TRAIN = 1
_STREAM_OOS = 2
_STREAM_BENCH_EURO = 900
_SUBSIM_KEY_BASE = 1000


# ---------------------------------------------------------------------------
# Validation helpers (fail-closed)
# ---------------------------------------------------------------------------


def _validate_seed(seed: int, name: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{name} must be an int")
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"{name} must be in [0, 2**32)")
    return seed


def _validate_dates(dates: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(dates, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("dates must be a 1-d sequence with at least 2 entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError("dates must be finite")
    if arr[0] != 0.0:
        raise ValueError("dates[0] must be 0.0 (valuation date)")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("dates must be strictly increasing")
    return tuple(float(x) for x in arr)


def _as_param_array(value: FloatSeq, d: int, name: str) -> Array:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.size == 1 and d > 1:
        arr = np.full(d, arr[0], dtype=np.float64)
    if arr.shape != (d,):
        raise ValueError(f"{name} must be a scalar or a length-{d} sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_positive(value: FloatSeq, d: int, name: str) -> Array:
    arr = _as_param_array(value, d, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def _infer_dim(s0: FloatSeq) -> int:
    arr = np.atleast_1d(np.asarray(s0, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("s0 must be a positive scalar or 1-d sequence")
    return int(arr.size)


def _corr_cholesky(corr: Sequence[Sequence[float]] | None, d: int) -> Array:
    if corr is None:
        return np.eye(d)
    mat = np.asarray(corr, dtype=np.float64)
    if mat.shape != (d, d):
        raise ValueError(f"corr must have shape ({d}, {d})")
    if not np.all(np.isfinite(mat)):
        raise ValueError("corr must be finite")
    if not np.allclose(mat, mat.T, atol=1e-10):
        raise ValueError("corr must be symmetric")
    if not np.allclose(np.diag(mat), 1.0, atol=1e-10):
        raise ValueError("corr must have unit diagonal")
    try:
        chol = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("corr must be positive definite") from exc
    return np.asarray(chol, dtype=np.float64)


def _check_payoff(payoff: PayoffFn, s0_arr: Array) -> None:
    if not callable(payoff):
        raise ValueError("payoff must be callable(states, t) -> (n,) values")
    probe = payoff(np.tile(s0_arr, (8, 1)), 0.0)
    out = np.asarray(probe, dtype=np.float64)
    if out.shape != (8,):
        raise ValueError("payoff must map (n, d) states to an (n,) array")
    if not np.all(np.isfinite(out)):
        raise ValueError("payoff must be finite at t=0")


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


def _exponents(d: int, degree: int) -> list[tuple[int, ...]]:
    exps = [e for e in itertools.product(range(degree + 1), repeat=d) if 0 < sum(e) <= degree]
    exps.sort(key=lambda e: (sum(e), e))
    return exps


def basis_size(d: int, degree: int) -> int:
    """Number of polynomial basis terms (constant included)."""
    return math.comb(d + degree, degree)


def polynomial_basis(states: Array, degree: int, scale: float = 1.0) -> Array:
    """Full polynomial tensor basis of total degree <= ``degree``.

    ``states`` is ``(n, d)``; the constant term comes first, then monomials in
    ``states / scale`` ordered by total degree.  Documented basis for the LSM
    regressions (LS01 use Laguerre polynomials; scaled monomials of the same
    total degree span the same space and are well conditioned for degree <= 3).
    """
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("states must be a 2-d (n, d) array")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    x = arr / scale
    cols = [np.ones(x.shape[0], dtype=np.float64)]
    for e in _exponents(x.shape[1], degree):
        col = np.ones(x.shape[0], dtype=np.float64)
        for a, p in enumerate(e):
            if p:
                col = col * x[:, a] ** p
        cols.append(col)
    return np.asarray(np.column_stack(cols), dtype=np.float64)


# ---------------------------------------------------------------------------
# Payoff factories and GBM simulation
# ---------------------------------------------------------------------------


def put_payoff(k: float) -> PayoffFn:
    """Vanilla put payoff on the first asset: max(k - S[:, 0], 0)."""
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("strike must be positive and finite")

    def payoff(states: Array, t: float) -> Array:
        return np.asarray(np.maximum(k - states[:, 0], 0.0), dtype=np.float64)

    return payoff


def call_payoff(k: float) -> PayoffFn:
    """Vanilla call payoff on the first asset: max(S[:, 0] - k, 0)."""
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("strike must be positive and finite")

    def payoff(states: Array, t: float) -> Array:
        return np.asarray(np.maximum(states[:, 0] - k, 0.0), dtype=np.float64)

    return payoff


def max_call_payoff(k: float) -> PayoffFn:
    """Multidimensional max-call payoff: max(max_j S[:, j] - k, 0) (AB04 headline)."""
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("strike must be positive and finite")

    def payoff(states: Array, t: float) -> Array:
        return np.asarray(np.maximum(states.max(axis=1) - k, 0.0), dtype=np.float64)

    return payoff


def simulate_gbm(
    s0: FloatSeq,
    r: float,
    q: FloatSeq,
    sigma: FloatSeq,
    dates: Sequence[float],
    n_paths: int,
    seed: int,
    corr: Sequence[Sequence[float]] | None = None,
    stream: int = 0,
) -> Array:
    """Exact risk-neutral GBM paths sampled at ``dates``.

    Returns ``(n_paths, n_dates, d)``.  Between dates the log-increment is
    exact: ``S *= exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)`` with ``Z``
    correlated through the Cholesky factor of ``corr``.  The stream is keyed by
    ``SeedSequence(entropy=seed, spawn_key=(stream,))``.
    """
    grid = _validate_dates(dates)
    _validate_seed(seed)
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive int")
    if not math.isfinite(r):
        raise ValueError("r must be finite")
    d = _infer_dim(s0)
    s0_arr = _validate_positive(s0, d, "s0")
    q_arr = _as_param_array(q, d, "q")
    sig_arr = _validate_positive(sigma, d, "sigma")
    chol = _corr_cholesky(corr, d)
    drift = r - q_arr - 0.5 * sig_arr**2
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(stream,)))
    out = np.empty((n_paths, len(grid), d), dtype=np.float64)
    out[:, 0, :] = s0_arr
    state = np.tile(s0_arr, (n_paths, 1))
    for j in range(len(grid) - 1):
        h = grid[j + 1] - grid[j]
        z = rng.standard_normal((n_paths, d)) @ chol.T
        state = state * np.exp(drift * h + math.sqrt(h) * sig_arr * z)
        out[:, j + 1, :] = state
    return out


def _bs_european(
    s: float, k: float, t: float, r: float, q: float, sigma: float, call: bool
) -> float:
    """Black-Scholes European closed form (benchmark reference)."""
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if call:
        return float(s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))
    return float(k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# LSM policy / results
# ---------------------------------------------------------------------------


class LSMPolicy(NamedTuple):
    """Frozen LSM policy exported for out-of-sample pricing and dual use.

    ``coefficients`` are the ITM-only exercise regressions (LS01, section 3)
    driving the stopping rule.  ``dual_coefficients`` are companion
    all-paths continuation regressions used only by :func:`dual_upper_bound`
    to build the envelope ``max(h, f)``; they are trained on every training
    path so the polynomial is constrained in the OTM region too, and their
    values are clipped to ``[0, continuation_caps[i]]`` — the empirical range
    of realised continuation cashflows at each date — to bound tail
    extrapolation error.
    """

    dates: tuple[float, ...]
    degree: int
    basis_scale: float
    coefficients: tuple[Array | None, ...]  # per date; None = never exercise there
    exercise_at_zero: bool
    zero_continuation: float  # training-set continuation estimate at t=0
    dual_coefficients: tuple[Array | None, ...]  # all-paths envelope regressions
    continuation_caps: tuple[float, ...]  # per-date clip ceiling for the envelope


class LSMResult(NamedTuple):
    in_sample_price: float  # high-bias (regression look-ahead), diagnostic only
    in_sample_stderr: float
    oos_price: float  # low-bias out-of-sample policy value (the reported lower bound)
    oos_stderr: float
    policy: LSMPolicy
    n_basis_terms: int
    dates_regressed: tuple[int, ...]
    t0_exercise: bool
    t0_exercise_fraction_oos: float


def _policy_forward(
    paths: Array,
    payoff: PayoffFn,
    r: float,
    dates: tuple[float, ...],
    policy: LSMPolicy,
) -> tuple[Array, float]:
    """Stop each path at the first date where immediate > fitted continuation.

    Returns discounted-to-0 stopped values ``(n,)`` and the t=0 exercise
    fraction (1.0 or 0.0: the t=0 state is identical across paths).
    """
    n = paths.shape[0]
    values = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    frac_t0 = 0.0
    h0 = float(payoff(paths[:1, 0], dates[0])[0])
    if policy.exercise_at_zero and h0 > 0.0 and h0 >= policy.zero_continuation:
        values[:] = h0
        active[:] = False
        frac_t0 = 1.0
    for i in range(1, len(dates) - 1):
        coef = policy.coefficients[i]
        if coef is None or not active.any():
            continue
        idx = np.flatnonzero(active)
        states = np.asarray(paths[idx, i], dtype=np.float64)
        imm = payoff(states, dates[i])
        cont = polynomial_basis(states, policy.degree, policy.basis_scale) @ coef
        ex = (imm > 0.0) & (imm > cont)
        sel = idx[ex]
        values[sel] = imm[ex] * math.exp(-r * dates[i])
        active[sel] = False
    if active.any():
        idx = np.flatnonzero(active)
        values[idx] = payoff(np.asarray(paths[idx, -1]), dates[-1]) * math.exp(-r * dates[-1])
    return values, frac_t0


def lsm_american_price(
    s0: FloatSeq,
    r: float,
    q: FloatSeq,
    sigma: FloatSeq,
    dates: Sequence[float],
    payoff: PayoffFn,
    *,
    n_paths: int = 20_000,
    n_oos_paths: int | None = None,
    degree: int = 3,
    basis_scale: float = 1.0,
    seed: int = 20260929,
    corr: Sequence[Sequence[float]] | None = None,
    exercise_at_zero: bool = False,
) -> LSMResult:
    """Longstaff-Schwartz American (Bermudan on ``dates``) Monte Carlo price.

    Trains the regression policy on ``n_paths`` paths and reports the low-bias
    out-of-sample price on ``n_oos_paths`` independent paths (the standard LSM
    bias control; defaults to ``n_paths``).  ``dates[0]`` must be 0.0 and the
    last date is maturity.  With ``exercise_at_zero=True`` the option is
    exercised at t=0 when intrinsic >= the training-set continuation estimate
    (immediate-exercise boundary sanity).
    """
    grid = _validate_dates(dates)
    _validate_seed(seed)
    d = _infer_dim(s0)
    s0_arr = _validate_positive(s0, d, "s0")
    _check_payoff(payoff, s0_arr)
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an int >= 1")
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 64:
        raise ValueError("n_paths must be an int >= 64")
    n_oos = n_paths if n_oos_paths is None else n_oos_paths
    if isinstance(n_oos, bool) or not isinstance(n_oos, int) or n_oos < 64:
        raise ValueError("n_oos_paths must be an int >= 64")
    if not math.isfinite(basis_scale) or basis_scale <= 0.0:
        raise ValueError("basis_scale must be positive and finite")
    n_basis = basis_size(d, degree)
    if n_paths < 8 * n_basis:
        raise ValueError(f"n_paths must be >= 8 * basis size ({8 * n_basis}) for degree={degree}")

    train = simulate_gbm(s0_arr, r, q, sigma, grid, n_paths, seed, corr, stream=_STREAM_TRAIN)
    oos = simulate_gbm(s0_arr, r, q, sigma, grid, n_oos, seed, corr, stream=_STREAM_OOS)

    n_dates = len(grid)
    cf = payoff(np.asarray(train[:, -1]), grid[-1]).copy()
    cf_time = np.full(n_paths, grid[-1], dtype=np.float64)
    coefficients: list[Array | None] = [None] * n_dates
    dual_coefficients: list[Array | None] = [None] * n_dates
    caps: list[float] = [0.0] * n_dates
    dates_regressed: list[int] = []
    for i in range(n_dates - 2, 0, -1):
        states = np.asarray(train[:, i], dtype=np.float64)
        imm = payoff(states, grid[i])
        itm = imm > 0.0
        if int(itm.sum()) < n_basis + 1:
            continue  # regression degenerate at this date; policy = continue (recorded)
        design = polynomial_basis(states[itm], degree, basis_scale)
        y = cf[itm] * np.exp(-r * (cf_time[itm] - grid[i]))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        coef = np.asarray(coef, dtype=np.float64)
        cont = design @ coef
        ex = imm[itm] > cont
        sel = np.flatnonzero(itm)[ex]
        cf[sel] = imm[sel]
        cf_time[sel] = grid[i]
        coefficients[i] = coef
        # Companion all-paths regression for the dual envelope (see LSMPolicy).
        design_all = polynomial_basis(states, degree, basis_scale)
        y_all = cf * np.exp(-r * (cf_time - grid[i]))
        coef_dual = np.linalg.lstsq(design_all, y_all, rcond=None)[0]
        dual_coefficients[i] = np.asarray(coef_dual, dtype=np.float64)
        caps[i] = float(y_all.max())
        dates_regressed.append(i)

    discounted = cf * np.exp(-r * cf_time)
    zero_cont = float(discounted.mean())
    in_price = zero_cont
    in_se = float(discounted.std(ddof=1) / math.sqrt(n_paths))
    h0 = float(payoff(s0_arr.reshape(1, d), grid[0])[0])
    t0_exercise = bool(exercise_at_zero and h0 > 0.0 and h0 >= zero_cont)
    if t0_exercise:
        in_price = h0
        in_se = 0.0

    policy = LSMPolicy(
        dates=grid,
        degree=degree,
        basis_scale=basis_scale,
        coefficients=tuple(coefficients),
        exercise_at_zero=exercise_at_zero,
        zero_continuation=zero_cont,
        dual_coefficients=tuple(dual_coefficients),
        continuation_caps=tuple(caps),
    )
    oos_values, frac_t0 = _policy_forward(oos, payoff, r, grid, policy)
    if t0_exercise != (frac_t0 > 0.0):
        raise RuntimeError("t=0 exercise decision inconsistent between train and oos passes")
    oos_price = float(oos_values.mean())
    oos_se = float(oos_values.std(ddof=1) / math.sqrt(n_oos))
    if not (math.isfinite(in_price) and math.isfinite(oos_price)):
        raise RuntimeError("LSM produced a non-finite price; inputs are degenerate")
    return LSMResult(
        in_sample_price=in_price,
        in_sample_stderr=in_se,
        oos_price=oos_price,
        oos_stderr=oos_se,
        policy=policy,
        n_basis_terms=n_basis,
        dates_regressed=tuple(sorted(dates_regressed)),
        t0_exercise=t0_exercise,
        t0_exercise_fraction_oos=float(frac_t0),
    )


# ---------------------------------------------------------------------------
# Andersen-Broadie dual upper bound
# ---------------------------------------------------------------------------


class DualResult(NamedTuple):
    upper_bound: float
    stderr: float
    policy_value: float  # primal value of the frozen policy on the same primary paths
    policy_value_stderr: float
    n_sub: int
    n_paths: int
    n_sub_paths_total: int  # documented sub-simulation budget actually consumed


def dual_upper_bound(
    s0: FloatSeq,
    r: float,
    q: FloatSeq,
    sigma: FloatSeq,
    dates: Sequence[float],
    payoff: PayoffFn,
    policy: LSMPolicy,
    *,
    n_paths: int = 2_000,
    n_sub: int = 64,
    seed: int = 20260929,
    corr: Sequence[Sequence[float]] | None = None,
) -> DualResult:
    """Andersen-Broadie (2004) primal-dual upper bound for a frozen LSM policy.

    The martingale is the Doob decomposition of the regression envelope
    ``g_i = max(e^{-r t_i} h_i, e^{-r t_i} f_reg(t_i, X_i))`` built from the
    policy's all-paths dual regressions (clipped to the training continuation
    range): each increment ``g_{i+1} - E_hat[g_{i+1} | F_{t_i}]`` estimates
    the conditional expectation by ``n_sub`` one-step sub-paths per primary
    path (an independent per-path batch at every date, t=0 included), scored
    with the same envelope at ``t_{i+1}``.  The reported bound is
    ``mean_p max_i(e^{-r t_i} h_i - M_i)``
    over the primary paths.  Requires dual regression coverage at every
    interior exercise date (raises otherwise); sub-simulation streams are
    prefix-nested in ``n_sub`` so bounds at different budgets under the same
    seed are paired.
    """
    grid = _validate_dates(dates)
    _validate_seed(seed)
    d = _infer_dim(s0)
    s0_arr = _validate_positive(s0, d, "s0")
    _check_payoff(payoff, s0_arr)
    q_arr = _as_param_array(q, d, "q")
    sig_arr = _validate_positive(sigma, d, "sigma")
    chol = _corr_cholesky(corr, d)
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 16:
        raise ValueError("n_paths must be an int >= 16")
    if isinstance(n_sub, bool) or not isinstance(n_sub, int) or n_sub < 8:
        raise ValueError("n_sub must be an int >= 8")
    if not isinstance(policy, LSMPolicy):
        raise ValueError("policy must be an LSMPolicy from lsm_american_price")
    if policy.dates != grid:
        raise ValueError("policy.dates must match the dual exercise dates")
    if not math.isfinite(r):
        raise ValueError("r must be finite")
    n_dates = len(grid)
    for i in range(1, n_dates - 1):
        if policy.dual_coefficients[i] is None:
            raise ValueError(
                f"policy has no continuation regression at interior date index {i}; "
                "the dual estimator requires LSM coverage at every interior exercise date"
            )

    drift = r - q_arr - 0.5 * sig_arr**2
    disc = np.exp(-r * np.asarray(grid, dtype=np.float64))
    paths = simulate_gbm(
        s0_arr, r, q_arr, sig_arr, grid, n_paths, seed, corr, stream=_STREAM_DUAL_PRIMARY
    )
    v = np.empty((n_dates, n_paths), dtype=np.float64)
    for i in range(n_dates):
        v[i] = disc[i] * payoff(np.asarray(paths[:, i]), grid[i])

    def _envelope(states: Array, i: int) -> Array:
        """Discounted dual envelope g_i = max(h_i, clipped f_reg_i) at date index i."""
        imm = payoff(states, grid[i])
        coef = policy.dual_coefficients[i]
        if i == n_dates - 1 or coef is None:
            return np.asarray(disc[i] * imm, dtype=np.float64)
        f_reg = polynomial_basis(states, policy.degree, policy.basis_scale) @ coef
        f_reg = np.clip(f_reg, 0.0, policy.continuation_caps[i])
        return np.asarray(disc[i] * np.maximum(imm, f_reg), dtype=np.float64)

    # g on the primary paths (g_0 = v_0 is unused: M_0 = 0).
    g = np.empty((n_dates, n_paths), dtype=np.float64)
    g[0] = v[0]
    for i in range(1, n_dates):
        g[i] = _envelope(np.asarray(paths[:, i], dtype=np.float64), i)

    # e_hat[i] = E_hat[g_{i+1} | F_{t_i}] from n_sub one-step sub-paths.  The
    # t=0 batch is per-primary-path too (states are identical, streams are
    # not), so every conditional-expectation error is independent across
    # primary paths and averages out in the reported mean/stderr.
    e_hat = np.zeros((n_dates, n_paths), dtype=np.float64)
    for i in range(n_dates - 1):
        start = np.asarray(paths[:, i], dtype=np.float64)
        rng = np.random.default_rng(
            np.random.SeedSequence(entropy=seed, spawn_key=(_SUBSIM_KEY_BASE + i,))
        )
        h_step = grid[i + 1] - grid[i]
        z = rng.standard_normal((n_sub, n_paths, d)) @ chol.T
        nxt = start[None, :, :] * np.exp(drift * h_step + math.sqrt(h_step) * sig_arr * z)
        env = _envelope(np.asarray(nxt.reshape(-1, d), dtype=np.float64), i + 1)
        e_hat[i] = env.reshape(n_sub, n_paths).mean(axis=0)

    # Doob martingale of the envelope and the Rogers/Haugh-Kogan dual estimator.
    m_cum = np.zeros((n_dates, n_paths), dtype=np.float64)
    for i in range(1, n_dates):
        m_cum[i] = m_cum[i - 1] + (g[i] - e_hat[i - 1])
    dual_values = np.max(v - m_cum, axis=0)
    upper = float(dual_values.mean())
    upper_se = float(dual_values.std(ddof=1) / math.sqrt(n_paths))

    pol_values, _ = _policy_forward(paths, payoff, r, grid, policy)
    pol_value = float(pol_values.mean())
    pol_se = float(pol_values.std(ddof=1) / math.sqrt(n_paths))
    if not (math.isfinite(upper) and math.isfinite(pol_value)):
        raise RuntimeError("dual estimator produced non-finite values; inputs are degenerate")
    return DualResult(
        upper_bound=upper,
        stderr=upper_se,
        policy_value=pol_value,
        policy_value_stderr=pol_se,
        n_sub=n_sub,
        n_paths=n_paths,
        n_sub_paths_total=n_sub * n_paths * (n_dates - 1),
    )


# ---------------------------------------------------------------------------
# SYNTHETIC benchmarks with known answers
# ---------------------------------------------------------------------------

_LABEL = "SYNTHETIC"
_REFS = (
    "Longstaff-Schwartz (2001) doi:10.1093/rfs/14.1.113; "
    "Andersen-Broadie (2004) doi:10.1287/mnsc.1040.0258; "
    "Rogers (2002) doi:10.1111/1467-9965.02010; "
    "Haugh-Kogan (2004) doi:10.1287/opre.1030.0070; "
    "Barone-Adesi-Whaley (1987) via models/american_baw.py"
)


def bs_american_put_bench(
    *,
    seed: int = 20260929,
    s0: float = 100.0,
    k: float = 100.0,
    t: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    n_dates: int = 13,
    n_paths: int = 20_000,
    n_dual_paths: int = 2_000,
    n_sub: int = 64,
    degree: int = 3,
) -> dict[str, Any]:
    """BS American put: LSM vs BAW agreement, LSM inside [lower, dual upper].

    Documented tolerance: BAW is an approximation (accuracy ~0.1-0.5% of price
    for 1y ATM puts) and the LSM grid is Bermudan (monthly), so agreement is
    asserted at 3.5 MC standard errors plus a 0.03 BAW/grid margin.
    """
    if n_dates < 3:
        raise ValueError("n_dates must be >= 3 for the American benchmark")
    grid = tuple(float(x) for x in np.linspace(0.0, t, n_dates))
    payoff = put_payoff(k)
    lsm = lsm_american_price(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        n_paths=n_paths,
        n_oos_paths=n_paths,
        degree=degree,
        basis_scale=k,
        seed=seed,
    )
    baw = baw_american(s0, k, t, r, q, sigma, "put")
    dual = dual_upper_bound(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        lsm.policy,
        n_paths=n_dual_paths,
        n_sub=n_sub,
        seed=seed,
    )
    euro = _bs_european(s0, k, t, r, q, sigma, call=False)
    lower = lsm.oos_price
    upper = dual.upper_bound
    se_pair = math.hypot(lsm.oos_stderr, dual.stderr)
    tol_agree = 3.5 * lsm.oos_stderr + 0.03
    return {
        "label": _LABEL,
        "benchmark": "bs_american_put",
        "references": _REFS,
        "params": {
            "s0": s0,
            "k": k,
            "t": t,
            "r": r,
            "q": q,
            "sigma": sigma,
            "n_dates": n_dates,
            "n_paths": n_paths,
            "n_dual_paths": n_dual_paths,
            "n_sub": n_sub,
            "degree": degree,
            "seed": seed,
        },
        "lsm_in_sample_price": lsm.in_sample_price,
        "lsm_in_sample_stderr": lsm.in_sample_stderr,
        "lsm_oos_lower_price": lower,
        "lsm_oos_stderr": lsm.oos_stderr,
        "baw_reference_price": baw,
        "european_put_closed_form": euro,
        "abs_lsm_oos_vs_baw": abs(lower - baw),
        "baw_agreement_tolerance": tol_agree,
        "baw_agreement_ok": bool(abs(lower - baw) <= tol_agree),
        "dual_upper_bound": upper,
        "dual_stderr": dual.stderr,
        "duality_gap": upper - lower,
        "lower_le_upper": bool(lower <= upper),
        "baw_within_dual_bracket": bool(lower - 3.0 * se_pair <= baw <= upper + 3.0 * se_pair),
        "early_exercise_premium_vs_european": lower - euro,
        "premium_nonneg_ok": bool(lower >= euro - 3.5 * lsm.oos_stderr),
        "n_sub_paths_total": dual.n_sub_paths_total,
    }


def deep_itm_boundary_bench(
    *,
    seed: int = 20260929,
    s0: float = 100.0,
    k: float = 200.0,
    t: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    n_dates: int = 13,
    n_paths: int = 8_000,
    degree: int = 3,
) -> dict[str, Any]:
    """Immediate-exercise boundary: deep-ITM American put exercised at t=0.

    With r > 0 the intrinsic value k - s0 exceeds any continuation value
    (interest on the strike dominates), so the t=0 exercise decision must fire
    and the price must equal intrinsic exactly; BAW also returns intrinsic
    below its critical price.
    """
    grid = tuple(float(x) for x in np.linspace(0.0, t, n_dates))
    payoff = put_payoff(k)
    lsm = lsm_american_price(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        n_paths=n_paths,
        n_oos_paths=n_paths,
        degree=degree,
        basis_scale=k,
        seed=seed,
        exercise_at_zero=True,
    )
    intrinsic = k - s0
    baw = baw_american(s0, k, t, r, q, sigma, "put")
    return {
        "label": _LABEL,
        "benchmark": "deep_itm_immediate_exercise",
        "references": _REFS,
        "params": {
            "s0": s0,
            "k": k,
            "t": t,
            "r": r,
            "q": q,
            "sigma": sigma,
            "n_dates": n_dates,
            "n_paths": n_paths,
            "degree": degree,
            "seed": seed,
        },
        "intrinsic_value": intrinsic,
        "continuation_estimate_t0": lsm.policy.zero_continuation,
        "continuation_below_intrinsic": bool(lsm.policy.zero_continuation < intrinsic),
        "exercised_at_zero": bool(lsm.t0_exercise),
        "t0_exercise_fraction_oos": lsm.t0_exercise_fraction_oos,
        "lsm_oos_price": lsm.oos_price,
        "price_equals_intrinsic": bool(abs(lsm.oos_price - intrinsic) <= 1e-10),
        "baw_reference_price": baw,
        "baw_equals_intrinsic": bool(abs(baw - intrinsic) <= 1e-10),
    }


def european_limit_bench(
    *,
    seed: int = 20260929,
    s0: float = 100.0,
    k: float = 100.0,
    t: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    n_paths: int = 40_000,
) -> dict[str, Any]:
    """European limit: LSM with exercise dates (0, T) recovers the BS closed form.

    With no intermediate exercise date there is no regression and the LSM
    machinery reduces to plain discounted-payoff Monte Carlo, which must match
    Black-Scholes within MC error (asserted at 4 standard errors + 0.01).
    """
    grid = (0.0, float(t))
    payoff = call_payoff(k)
    lsm = lsm_american_price(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        n_paths=n_paths,
        n_oos_paths=n_paths,
        seed=seed,
        basis_scale=k,
    )
    bs = _bs_european(s0, k, t, r, q, sigma, call=True)
    tol = 4.0 * lsm.oos_stderr + 0.01
    return {
        "label": _LABEL,
        "benchmark": "european_limit_bs_call",
        "references": _REFS,
        "params": {
            "s0": s0,
            "k": k,
            "t": t,
            "r": r,
            "q": q,
            "sigma": sigma,
            "n_paths": n_paths,
            "seed": seed,
        },
        "lsm_price": lsm.oos_price,
        "lsm_stderr": lsm.oos_stderr,
        "bs_closed_form": bs,
        "abs_diff_vs_closed_form": abs(lsm.oos_price - bs),
        "mc_error_tolerance": tol,
        "within_mc_error": bool(abs(lsm.oos_price - bs) <= tol),
        "dates_regressed": list(lsm.dates_regressed),
    }


def max_call_2asset_bench(
    *,
    seed: int = 20260929,
    s0: tuple[float, float] = (100.0, 100.0),
    k: float = 100.0,
    t: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: tuple[float, float] = (0.2, 0.2),
    rho: float = 0.5,
    n_dates: int = 5,
    n_paths: int = 20_000,
    n_dual_paths: int = 1_500,
    n_sub: int = 64,
    degree: int = 3,
) -> dict[str, Any]:
    """AB04 headline use case: 2-asset American max-call, LSM lower <= dual upper."""
    if n_dates < 3:
        raise ValueError("n_dates must be >= 3 for the American benchmark")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1)")
    grid = tuple(float(x) for x in np.linspace(0.0, t, n_dates))
    corr = ((1.0, rho), (rho, 1.0))
    payoff = max_call_payoff(k)
    lsm = lsm_american_price(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        n_paths=n_paths,
        n_oos_paths=n_paths,
        degree=degree,
        basis_scale=k,
        seed=seed,
        corr=corr,
    )
    dual = dual_upper_bound(
        s0,
        r,
        q,
        sigma,
        grid,
        payoff,
        lsm.policy,
        n_paths=n_dual_paths,
        n_sub=n_sub,
        seed=seed,
        corr=corr,
    )
    euro_paths = simulate_gbm(
        np.asarray(s0, dtype=np.float64),
        r,
        q,
        sigma,
        grid,
        8_000,
        seed,
        corr,
        stream=_STREAM_BENCH_EURO,
    )
    euro_vals = payoff(np.asarray(euro_paths[:, -1]), grid[-1]) * math.exp(-r * t)
    euro = float(euro_vals.mean())
    euro_se = float(euro_vals.std(ddof=1) / math.sqrt(euro_vals.size))
    lower = lsm.oos_price
    upper = dual.upper_bound
    se_trio = math.hypot(math.hypot(lsm.oos_stderr, dual.stderr), euro_se)
    return {
        "label": _LABEL,
        "benchmark": "max_call_2asset_american",
        "references": _REFS,
        "params": {
            "s0": list(s0),
            "k": k,
            "t": t,
            "r": r,
            "q": q,
            "sigma": list(sigma),
            "rho": rho,
            "n_dates": n_dates,
            "n_paths": n_paths,
            "n_dual_paths": n_dual_paths,
            "n_sub": n_sub,
            "degree": degree,
            "seed": seed,
        },
        "lsm_in_sample_price": lsm.in_sample_price,
        "lsm_oos_lower_price": lower,
        "lsm_oos_stderr": lsm.oos_stderr,
        "dual_upper_bound": upper,
        "dual_stderr": dual.stderr,
        "duality_gap": upper - lower,
        "lower_le_upper": bool(lower <= upper),
        "dual_policy_value": dual.policy_value,
        "european_max_call_mc": euro,
        "european_max_call_stderr": euro_se,
        "early_exercise_premium_est": lower - euro,
        "premium_nonneg_ok": bool(lower >= euro - 3.0 * se_trio),
        "n_basis_terms": lsm.n_basis_terms,
        "n_sub_paths_total": dual.n_sub_paths_total,
    }


def american_lsm_benchmarks(*, seed: int = 20260929, fast: bool = False) -> dict[str, Any]:
    """Run all four SYNTHETIC benchmarks; ``fast=True`` shrinks MC budgets."""
    _validate_seed(seed)
    if fast:
        put_kw: dict[str, Any] = {"n_paths": 8_000, "n_dual_paths": 800, "n_sub": 32}
        euro_kw: dict[str, Any] = {"n_paths": 12_000}
        max_kw: dict[str, Any] = {"n_paths": 8_000, "n_dual_paths": 600, "n_sub": 32}
        deep_kw: dict[str, Any] = {"n_paths": 4_000}
    else:
        put_kw, euro_kw, max_kw, deep_kw = {}, {}, {}, {}
    return {
        "label": _LABEL,
        "seed": seed,
        "fast": bool(fast),
        "benchmarks": {
            "bs_american_put": bs_american_put_bench(seed=seed, **put_kw),
            "deep_itm_boundary": deep_itm_boundary_bench(seed=seed, **deep_kw),
            "european_limit": european_limit_bench(seed=seed, **euro_kw),
            "max_call_2asset": max_call_2asset_bench(seed=seed, **max_kw),
        },
    }
