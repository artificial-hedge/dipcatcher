"""Large-deviations theory for portfolio risk: Gärtner-Ellis, Cramér-Chernoff,
and exponential-tilting importance sampling for VaR/ES estimation.

Core references
---------------
- Dembo & Zeitouni (1998/2010). *Large Deviations Techniques and Applications*,
  Springer.  Classical text on the Gärtner-Ellis theorem, the Fenchel-Legendre
  rate function, and Cramér-type theorems for i.i.d. and weakly dependent
  sequences.
- Glasserman & Li (2005). Importance sampling for portfolio credit risk.
  *Management Science* 51(11):1643–1656.  Uses the Gärtner-Ellis large-
  deviations rate function to derive an asymptotically optimal importance-
  sampling (IS) measure for multi-asset credit portfolios.
- Glasserman, Heidelberger & Shahabuddin (2002). Portfolio value-at-risk with
  heavy-tailed risk factors. *Mathematical Finance* 12(3):239–268.  IS with
  exponential tilting based on the large-deviations rate for VaR estimation.

All VaR/ES numbers produced here are SYNTHETIC — correctness evidence, never
market risk estimates.  IS is a variance-reduction technique.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize

Array = NDArray[np.float64]
LogMGFFn = Callable[[Array], float]
GradLogMGFFn = Callable[[Array], Array]
RateFn = Callable[[Array], float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_array(x: Array | list[float] | float, name: str = "x") -> Array:
    """Coerce to flat float64 array, fail-closed on non-finite entries."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _ensure_2d(cov: Array, name: str = "cov") -> Array:
    """Validate square symmetric float64 covariance."""
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-d array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.allclose(arr, arr.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _check_domain(lam: Array, bounds: tuple[float, float] | None = None) -> None:
    """Fail-closed when lambda is outside the effective domain of the log-MGF."""
    if bounds is not None:
        lo, hi = bounds
        if np.any(lam < lo) or np.any(lam > hi):
            raise ValueError(
                f"lambda values must be in [{lo}, {hi}], got min={lam.min()}, max={lam.max()}"
            )


# ---------------------------------------------------------------------------
# 1. Log-MGF implementations ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def gaussian_log_mgf(lam: Array, mu: Array, cov: Array) -> float:
    """Log-MGF of a multivariate Gaussian: Λ(λ) = μ·λ + ½ λᵀ Σ λ.

    Parameters
    ----------
    lam : (d,) float array
        Tilt direction.
    mu : (d,) float array
        Mean vector.
    cov : (d, d) float array
        Covariance matrix (must be symmetric positive-definite).

    Returns
    -------
    Λ(λ) ∈ ℝ
    """
    lm = _as_array(lam, "lam")
    m = _as_array(mu, "mu")
    c = _ensure_2d(cov, "cov")
    if lm.size != m.size or c.shape[0] != m.size:
        raise ValueError(
            f"dimension mismatch: lam={lm.size}, mu={m.size}, cov=({c.shape[0]},{c.shape[1]})"
        )
    # This raises LinAlgError if cov is not positive-definite.
    # We catch and re-raise with a domain-error message consistent with the lane.
    try:
        quad = float(lm @ (c @ lm))
    except linalg.LinAlgError:
        raise ValueError("covariance must be positive-definite") from None
    return float(np.dot(lm, m)) + 0.5 * quad


def gaussian_log_mgf_grad(lam: Array, mu: Array, cov: Array) -> Array:
    """Gradient of the multivariate Gaussian log-MGF: ∇Λ(λ) = μ + Σ λ."""
    lm = _as_array(lam, "lam")
    m = _as_array(mu, "mu")
    c = _ensure_2d(cov, "cov")
    if lm.size != m.size or c.shape[0] != m.size:
        raise ValueError(
            f"dimension mismatch: lam={lm.size}, mu={m.size}, cov=({c.shape[0]},{c.shape[1]})"
        )
    return np.asarray(m + c @ lm, dtype=np.float64)


def exponential_log_mgf(lam: Array, rates: Array) -> float:
    """Log-MGF of independent Exponentials: Λ(λ) = -∑ log(1 - λᵢ/rateᵢ).

    Defined only for λᵢ < rateᵢ (all i).  A ``ValueError`` is raised when
    any component violates the domain constraint.
    """
    lm = _as_array(lam, "lam")
    rt = _as_array(rates, "rates")
    if lm.size != rt.size:
        raise ValueError(f"dimension mismatch: lam={lm.size}, rates={rt.size}")
    if np.any(rt <= 0.0):
        raise ValueError("rates must be strictly positive")
    if np.any(lm >= rt):
        raise ValueError(f"lambda must satisfy λ_i < rate_i for all i; got lam max={lm.max()}")
    return float(-np.sum(np.log(1.0 - lm / rt)))


def exponential_log_mgf_grad(lam: Array, rates: Array) -> Array:
    """Gradient of the independent-exponential log-MGF: ∂Λ/∂λᵢ = 1/(rateᵢ - λᵢ)."""
    lm = _as_array(lam, "lam")
    rt = _as_array(rates, "rates")
    if lm.size != rt.size:
        raise ValueError(f"dimension mismatch: lam={lm.size}, rates={rt.size}")
    if np.any(rt <= 0.0):
        raise ValueError("rates must be strictly positive")
    if np.any(lm >= rt):
        raise ValueError(f"lambda must satisfy λ_i < rate_i for all i; got lam max={lm.max()}")
    return np.asarray(1.0 / (rt - lm), dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Rate functions (Fenchel-Legendre transform) ────────────────────────────
# ---------------------------------------------------------------------------


def gaussian_rate(x: Array, mu: Array, cov: Array) -> float:
    """Multivariate Gaussian rate function (closed form).

    Λ*(x) = ½ (x - μ)ᵀ Σ⁻¹ (x - μ)

    This is the Cramér rate for the sample mean: if X₁, …, Xₙ i.i.d. N(μ, Σ)
    then P(‖X̄ₙ - μ‖ ≥ ε) decays at exponential rate Λ*(μ + ε·u) for any
    direction u, as n → ∞.
    """
    xx = _as_array(x, "x")
    m = _as_array(mu, "mu")
    c = _ensure_2d(cov, "cov")
    if xx.size != m.size or c.shape[0] != m.size:
        raise ValueError(
            f"dimension mismatch: x={xx.size}, mu={m.size}, cov=({c.shape[0]},{c.shape[1]})"
        )
    diff = xx - m
    try:
        quad = float(diff @ linalg.solve(c, diff, assume_a="pos"))
    except linalg.LinAlgError:
        raise ValueError("covariance must be positive-definite") from None
    return 0.5 * max(quad, 0.0)


def exponential_rate(x: Array, rates: Array) -> float:
    """Rate function for independent Exponentials (closed form).

    For Exponential(rateᵢ) components:

        Λ*(x) = Σᵢ (rateᵢ·xᵢ - 1 - log(rateᵢ·xᵢ)),   all xᵢ > 0
        Λ*(x) = +∞,                                      any xᵢ ≤ 0

    The rate is +∞ for x outside the convex hull of the support (ℝ₊ᵈ).
    """
    xx = _as_array(x, "x")
    rt = _as_array(rates, "rates")
    if xx.size != rt.size:
        raise ValueError(f"dimension mismatch: x={xx.size}, rates={rt.size}")
    if np.any(rt <= 0.0):
        raise ValueError("rates must be strictly positive")
    if np.any(xx <= 0.0):
        return float("inf")
    scaled = rt * xx
    return float(np.sum(scaled - 1.0 - np.log(scaled)))


def fenchel_legendre(
    x: Array,
    log_mgf: LogMGFFn,
    grad_log_mgf: GradLogMGFFn | None = None,
    *,
    lam0: Array | None = None,
    lam_bounds: tuple[float, float] | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> float:
    """Numerical Fenchel-Legendre transform.

    Computes the large-deviations rate function

        Λ*(x) = sup_{λ ∈ dom(Λ)} { λ·x − Λ(λ) }

    via L-BFGS-B minimization of ``−obj(λ)``.  Passing the gradient
    ``∇Λ(λ)`` speeds up convergence.

    Parameters
    ----------
    x : (d,) float array
        Point at which to evaluate Λ*.
    log_mgf : callable
        Λ(λ) → ℝ, must be convex on its effective domain.
    grad_log_mgf : callable, optional
        ∇Λ(λ) → ℝᵈ.
    lam0 : (d,) array, optional
        Initial λ for the optimizer (default: zeros).
    lam_bounds : (float, float), optional
        Box bounds [lo, hi] on every component of λ (e.g. ``(-inf, min(rate))``
        for exponential). ``None`` means unconstrained.
    tol : float
        Optimizer tolerance (``ftol`` for L-BFGS-B).
    max_iter : int
        Maximum L-BFGS-B iterations.

    Returns
    -------
    Λ*(x) ∈ [0, ∞].  Returns ``inf`` when the optimizer fails to converge or
    when ``x`` is outside the convex hull of the support.
    """
    xx = _as_array(x, "x")
    d = xx.size

    if lam0 is None:
        lam0 = np.zeros(d, dtype=np.float64)
    elif isinstance(lam0, np.ndarray) and lam0.shape == (d,):
        lam0 = lam0.astype(np.float64, copy=True).ravel()
    else:
        raise ValueError("lam0 must be None or an array of shape (d,)")

    def neg_obj(lam: Array) -> float:
        return float(log_mgf(lam) - np.dot(xx, lam))

    jac_fn = None
    if grad_log_mgf is not None:

        def _jac(lam: Array) -> Array:
            return np.asarray(grad_log_mgf(lam) - xx, dtype=np.float64)

        jac_fn = _jac

    bounds = None
    if lam_bounds is not None:
        bounds = [lam_bounds] * d  # type: ignore[assignment]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            result = optimize.minimize(
                neg_obj,
                lam0,
                method="L-BFGS-B",
                jac=jac_fn,
                bounds=bounds,  # type: ignore[arg-type]
                options={"ftol": tol, "maxiter": max_iter, "maxls": 50},
            )
        except (ValueError, linalg.LinAlgError, optimize.OptimizeWarning):
            return float("inf")

    if not result.success:
        # Non-convergence typically means x is outside the domain.
        # Return inf — the rate is infinite.
        return float("inf")

    rate = -float(result.fun)
    # Clip numerical noise near zero.
    if rate < 0 and rate > -1e-10:
        rate = 0.0
    if rate < 0:
        # Rate function must be non-negative by convexity of log-MGF (Λ(0)=0).
        # Negative rates indicate numerical instability; fail closed.
        return float("inf")
    return rate


# ---------------------------------------------------------------------------
# 3. Cramér-Chernoff bound ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def chernoff_bound(
    t: float,
    w: Array,
    log_mgf: LogMGFFn,
    grad_log_mgf: GradLogMGFFn | None = None,
    *,
    theta_max: float = 1e4,
    tol: float = 1e-8,
) -> float:
    """Cramér-Chernoff upper bound on the tail of a linear combination.

    For portfolio loss ``L = w·X`` and threshold ``t > w·E[X]``:

        P(L ≥ t) ≤ exp(−I(t)),

        I(t) = sup_{θ ≥ 0} { θ·t − Λ(θ·w) }

    where Λ(λ) = log E[exp(λ·X)] is the log-MGF of the risk-factor vector X.

    The scalar rate function ``I(t)`` is the Fenchel-Legendre of the 1-d
    log-MGF ``θ ↦ Λ(θ·w)`` evaluated at ``t``.

    Parameters
    ----------
    t : float
        Tail threshold (must be finite).
    w : (d,) float array
        Portfolio weight / exposure vector.
    log_mgf : callable
        Multivariate log-MGF Λ(λ).
    grad_log_mgf : callable, optional
        Gradient ∇Λ(λ).  When supplied, is used to compute the derivative of
        the scalar objective analytically.
    theta_max : float
        Upper bound for the scalar tilting parameter (ensures the optimizer
        does not escape to +∞).
    tol : float
        Optimizer tolerance.

    Returns
    -------
    exp(−I(t)) ∈ (0, 1].  Returns ``1.0`` when ``t`` is below the mean
    (the bound is uninformative) and raises ``ValueError`` when the log-MGF
    is non-finite at the optimal θ.
    """
    ww = _as_array(w, "w")
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    d = ww.size

    # Pre-compute the mean to detect the trivial case.
    # For Gaussian: E[w·X] = w·μ.  For generic: use Λ'(0)·w.
    mean_loss: float | None = None

    if grad_log_mgf is not None:
        try:
            g0 = grad_log_mgf(np.zeros(d, dtype=np.float64))
            mean_loss = float(np.dot(ww, g0))
        except (ValueError, linalg.LinAlgError):
            mean_loss = None

    if mean_loss is not None and t <= mean_loss:
        return 1.0

    def scalar_neg_rate(theta: float) -> float:
        if theta < 0:
            return float("inf")
        try:
            lam = theta * ww
            return float(theta * t - log_mgf(lam))
        except (ValueError, linalg.LinAlgError):
            return float("-inf")

    # Bounded scalar optimization: θ ∈ [0, θ_max].
    result = optimize.minimize_scalar(
        lambda th: -scalar_neg_rate(th),
        bounds=(0.0, theta_max),
        method="bounded",
        options={"xatol": tol, "maxiter": 500},
    )

    if not result.success:
        # Fallback: try golden-section on a tighter bracket.
        result = optimize.minimize_scalar(
            lambda th: -scalar_neg_rate(th),
            bounds=(0.0, min(theta_max, 100.0)),
            method="bounded",
            options={"xatol": tol, "maxiter": 200},
        )

    rate = -result.fun  # result.fun = min(−obj) = −sup(obj)
    if not math.isfinite(rate) or rate < 0:
        return 1.0
    bound = float(np.exp(-rate))
    return min(bound, 1.0)


# ---------------------------------------------------------------------------
# 4. Optimal exponential-tilting parameter ─────────────────────────────────
# ---------------------------------------------------------------------------


def optimal_tilt(
    t: float,
    w: Array,
    grad_log_mgf: GradLogMGFFn,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Solve for the optimal exponential-tilting scalar θ*.

    Finds θ* ≥ 0 that satisfies

        w·∇Λ(θ*·w) = t

    i.e. the tilted mean of the linear loss ``w·X`` equals the threshold ``t``.

    Uses scalar bisection on the monotonic function ``g(θ) = w·∇Λ(θ·w)``.
    When the untransformed mean ``w·∇Λ(0) ≥ t`` (t is not in the right tail),
    returns ``0.0`` (no tilt).

    Parameters
    ----------
    t : float
        Target mean under the IS measure (e.g., VaR threshold).
    w : (d,) float array
        Loss direction (portfolio weights).
    grad_log_mgf : callable
        Gradient ∇Λ(λ) → ℝᵈ.
    tol : float
        Convergence tolerance on |g(θ) − t|.
    max_iter : int
        Maximum bisection steps.

    Returns
    -------
    θ* ≥ 0

    Raises
    ------
    ValueError
        When the mean function is degenerate or t cannot be reached.
    """
    ww = _as_array(w, "w")
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    d = ww.size

    # Mean under the original measure: g(0) = w·∇Λ(0).
    g0 = grad_log_mgf(np.zeros(d, dtype=np.float64))
    mean0 = float(np.dot(ww, g0))

    if not math.isfinite(mean0):
        raise ValueError("grad_log_mgf at zero must be finite")

    if t <= mean0:
        return 0.0

    def g(theta: float) -> float:
        lam = theta * ww
        return float(np.dot(ww, grad_log_mgf(lam)))

    # Find an upper bound via exponential search.
    hi = 1.0
    for _ in range(20):
        try:
            g_hi = g(hi)
            if not math.isfinite(g_hi) or g_hi >= t:
                break
        except (ValueError, linalg.LinAlgError):
            # MGF domain violation — back off.
            hi *= 0.5
            if hi < 1e-10:
                raise ValueError("cannot reach target mean within log-MGF domain") from None
            continue
        hi *= 2.0
    else:
        raise ValueError(
            f"cannot reach target mean t={t} within the log-MGF domain; g({hi})={g(hi):.4g}"
        )

    # Bisection on [0, hi].
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        try:
            g_mid = g(mid)
        except (ValueError, linalg.LinAlgError):
            hi = mid
            continue

        if not math.isfinite(g_mid):
            hi = mid
            continue

        if g_mid < t:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol * max(1.0, abs(mid)):
            break

    return float(0.5 * (lo + hi))


def optimal_tilt_gaussian(t: float, w: Array, mu: Array, cov: Array) -> float:
    """Closed-form optimal tilting scalar for Gaussian loss L = w·X.

    θ* = max(0, (t − w·μ) / (wᵀ Σ w))

    Derived from ∇Λ(λ) = μ + Σ λ and the equation w·(μ + θ Σ w) = t.
    """
    ww = _as_array(w, "w")
    m = _as_array(mu, "mu")
    c = _ensure_2d(cov, "cov")
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    if ww.size != m.size or c.shape[0] != m.size:
        raise ValueError(
            f"dimension mismatch: w={ww.size}, mu={m.size}, cov=({c.shape[0]},{c.shape[1]})"
        )
    mean_loss = float(np.dot(ww, m))
    if t <= mean_loss:
        return 0.0
    var_loss = float(ww @ (c @ ww))
    if var_loss <= 0.0:
        raise ValueError("portfolio loss variance must be positive")
    return (t - mean_loss) / var_loss


# ---------------------------------------------------------------------------
# 5. Importance-sampling tail-probability estimators ────────────────────────
# ---------------------------------------------------------------------------


def gaussian_is_tail_prob(
    rng: np.random.Generator,
    n_is: int,
    n_crude: int,
    w: Array,
    mu: Array,
    cov: Array,
    threshold: float,
    theta: float | None = None,
) -> dict[str, object]:
    """Gaussian IS estimate of P(w·X ≥ threshold).

    Draws ``n_crude`` paths from N(μ, Σ) and ``n_is`` paths from the
    exponentially tilted N(μ + θ Σ w, Σ).  The optimal tilting scalar is

        θ* = max(0, (threshold − w·μ) / (wᵀ Σ w)).

    Both the IS mean and the naive MC mean are unbiased; their sample
    variances are used to compute the variance-reduction factor.

    SYNTHETIC — no market data.
    """
    ww = _as_array(w, "w")
    m = _as_array(mu, "mu")
    c = _ensure_2d(cov, "cov")
    d = ww.size
    if m.size != d or c.shape[0] != d:
        raise ValueError("dimension mismatch")
    if n_is < 2 or n_crude < 2:
        raise ValueError("n_is and n_crude must be at least 2")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if theta is None:
        theta = optimal_tilt_gaussian(threshold, ww, m, c)

    tilt_vec = float(theta) * ww

    # --- Naive MC (crude) ---
    crude_samples = rng.multivariate_normal(m, c, size=n_crude)
    crude_losses = np.asarray(crude_samples @ ww, dtype=np.float64)
    crude_hits = (crude_losses >= threshold).astype(np.float64)
    crude_est = float(np.mean(crude_hits))
    crude_var = float(np.var(crude_hits, ddof=1)) / n_crude

    # --- IS ---
    is_samples = rng.multivariate_normal(m + c @ tilt_vec, c, size=n_is)
    is_losses = np.asarray(is_samples @ ww, dtype=np.float64)
    is_hits = (is_losses >= threshold).astype(np.float64)

    # Likelihood ratio: dP/dQ = exp(−θ w·(X−μ) + ½ θ² wᵀ Σ w)
    #   = exp(−θ·L + θ·w·μ + ½ θ²·wᵀ Σ w)
    log_wts = (
        -float(theta) * is_losses
        + float(theta) * float(np.dot(ww, m))
        + 0.5 * (float(theta) ** 2) * float(ww @ (c @ ww))
    )
    weights = np.asarray(np.exp(log_wts), dtype=np.float64)

    weighted_hits = weights * is_hits
    is_est = float(np.mean(weighted_hits))
    is_var = float(np.var(weighted_hits, ddof=1)) / n_is

    # Variance-reduction factor
    vrf: float | None
    reason: str | None = None
    infinite = False
    if is_var <= 0.0 and crude_var > 0.0:
        vrf = None
        infinite = True
        reason = "IS weighted outcomes have zero sample variance"
    elif is_var <= 0.0 or not math.isfinite(is_var):
        vrf = None
        reason = "IS variance is not positive"
    elif crude_var <= 0.0:
        vrf = None
        reason = "crude variance is not positive (no tail hits)"
    else:
        vrf = crude_var / is_var

    return {
        "method": "gaussian_exponential_tilting_is",
        "synthetic": True,
        "threshold": float(threshold),
        "optimal_theta": float(theta),
        "n_is": n_is,
        "n_crude": n_crude,
        "crude_estimate": float(crude_est),
        "is_estimate": float(is_est),
        "crude_variance": float(crude_var),
        "is_variance": float(is_var),
        "variance_reduction_factor": vrf,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "Both estimators are unbiased for P(w·X ≥ threshold). "
            "VRF = Var(crude) / Var(IS).  SYNTHETIC."
        ),
    }


def importance_sampling_report(
    crude_values: Array,
    is_weighted_outcomes: Array,
) -> dict[str, object]:
    """Compute a variance-reduction report from pre-computed outcomes.

    Useful when sample paths are produced externally (e.g., from a factor
    model where the likelihood ratio is path-dependent).

    Parameters
    ----------
    crude_values : (n,) float array
        Indicator or loss values under the original measure.
    is_weighted_outcomes : (m,) float array
        Weight × indicator outcomes under the IS measure.

    Returns
    -------
    Dictionary with estimates, variances, and VRF.
    """
    c = np.asarray(crude_values, dtype=np.float64).ravel()
    w_out = np.asarray(is_weighted_outcomes, dtype=np.float64).ravel()
    if c.size < 2 or w_out.size < 2:
        return {
            "method": "importance_sampling_generic",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "need at least 2 observations in each sample",
        }
    if not np.isfinite(c).all() or not np.isfinite(w_out).all():
        return {
            "method": "importance_sampling_generic",
            "variance_reduction_factor": None,
            "reason": "non-finite outcomes",
        }
    n_c = int(c.size)
    n_is = int(w_out.size)
    var_crude = float(np.var(c, ddof=1)) / n_c
    var_is = float(np.var(w_out, ddof=1)) / n_is
    vrf: float | None = None
    infinite = False
    reason: str | None = None
    if var_is == 0.0 and var_crude > 0.0:
        infinite = True
        reason = "IS weighted outcomes have zero sample variance"
    elif var_is <= 0.0 or not math.isfinite(var_is):
        reason = "IS variance is not positive"
    elif var_crude <= 0.0 or not math.isfinite(var_crude):
        reason = "crude variance is not positive"
    else:
        vrf = var_crude / var_is
    return {
        "method": "importance_sampling_generic",
        "synthetic": True,
        "n_crude": n_c,
        "n_is": n_is,
        "crude_estimate": float(np.mean(c)),
        "is_estimate": float(np.mean(w_out)),
        "variance_crude_estimator": var_crude,
        "variance_is_estimator": var_is,
        "variance_reduction_factor": vrf,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": "VRF = Var(crude mean) / Var(IS mean).  SYNTHETIC.",
    }


# ---------------------------------------------------------------------------
# 6. Credit-risk factor model ───────────────────────────────────────────────
# ---------------------------------------------------------------------------


def factor_model_covariance(
    betas: Array,
    psi: Array,
    factor_cov: Array | None = None,
) -> Array:
    """Covariance matrix of a linear factor model X = B·F + ε.

    Σ = B Σ_F Bᵀ + diag(ψ)

    where B is (d, k) loadings, ψ is the vector of idiosyncratic variances,
    and Σ_F is the factor covariance (default: identity).
    """
    b = np.asarray(betas, dtype=np.float64)
    p = _as_array(psi, "psi")
    if b.ndim != 2:
        raise ValueError("betas must be a 2-d array")
    d, k = b.shape
    if p.size != d:
        raise ValueError(f"dimension mismatch: psi={p.size}, betas rows={d}")
    if np.any(p <= 0.0):
        raise ValueError("idiosyncratic variances (psi) must be positive")
    if factor_cov is None:
        f_cov = np.eye(k, dtype=np.float64)
    else:
        f_cov = _ensure_2d(factor_cov, "factor_cov")
        if f_cov.shape[0] != k:
            raise ValueError(
                f"factor_cov shape mismatch: ({f_cov.shape[0]},{f_cov.shape[1]}) vs k={k}"
            )
    cov = b @ f_cov @ b.T + np.diag(p)
    return np.asarray(cov, dtype=np.float64)


def factor_model_sample(
    rng: np.random.Generator,
    n: int,
    w: Array,
    betas: Array,
    psi: Array,
    *,
    df: float = 3.0,
    factor_mean: float = 0.0,
    factor_cov: Array | None = None,
) -> Array:
    """Sample portfolio losses from a factor model with t-distributed idiosyncratic risk.

    Model:
        X_j = Σ_f β_{jf} F_f + ε_j,
        F ~ N(μ_F·𝟏, Σ_F),  ε_j ~ σ_j · t_ν  (independent)

    where σ_j = √(ψ_j / (ν/(ν−2))) to standardize the t innovations
    (unit variance for ν > 2).

    Returns ``(n,)`` array of portfolio losses ``L = w·X``.

    SYNTHETIC.
    """
    ww = _as_array(w, "w")
    p = _as_array(psi, "psi")
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("betas must be 2-d")
    d, k = b.shape
    if d != ww.size or d != p.size:
        raise ValueError(f"dimension mismatch: w={ww.size}, psi={p.size}, betas rows={d}")
    if df <= 2.0:
        raise ValueError("df must be > 2 for finite variance")
    if not math.isfinite(factor_mean):
        raise ValueError("factor_mean must be finite")

    if factor_cov is None:
        f_cov = np.eye(k, dtype=np.float64)
    else:
        f_cov = _ensure_2d(factor_cov, "factor_cov")

    # t-scale to achieve unit idiosyncratic variance
    t_scale = np.sqrt(p * (df - 2.0) / df)

    # Systematic factors
    f_mean_vec = np.full(k, factor_mean, dtype=np.float64)
    factors = rng.multivariate_normal(f_mean_vec, f_cov, size=n)  # (n, k)

    # Idiosyncratic
    eps = rng.standard_t(df, size=(n, d)) * t_scale[np.newaxis, :]  # (n, d)

    # Asset returns
    systematic = factors @ b.T  # (n, d)
    X = systematic + eps
    losses = np.asarray(X @ ww, dtype=np.float64)
    return losses


def factor_model_is_tail_prob(
    rng: np.random.Generator,
    n_is: int,
    n_crude: int,
    w: Array,
    betas: Array,
    psi: Array,
    threshold: float,
    *,
    df: float = 3.0,
    theta: float | None = None,
    factor_cov: Array | None = None,
) -> dict[str, object]:
    """IS estimate of P(L ≥ threshold) for the factor model.

    Only the *systematic Gaussian factors* are tilted (Glasserman-Li style).
    The idiosyncratic t-distributed innovations are drawn from their original
    distribution.

    The optimal tilting scalar for a homogeneous factor is

        θ* = (threshold − w·(B·μ_F)) / ‖wᵀ B‖²

    when factor_cov = I and factor mean = μ_F · 1.
    """
    ww = _as_array(w, "w")
    p = _as_array(psi, "psi")
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("betas must be 2-d")
    d, k = b.shape
    if d != ww.size or d != p.size:
        raise ValueError("dimension mismatch")
    if df <= 2.0:
        raise ValueError("df must be > 2 for finite variance")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if n_is < 2 or n_crude < 2:
        raise ValueError("n_is and n_crude must be at least 2")

    if factor_cov is None:
        f_cov = np.eye(k, dtype=np.float64)
    else:
        f_cov = _ensure_2d(factor_cov, "factor_cov")

    # Effective systematic loading for the portfolio
    # Under crude measure: F ~ N(0, I_k), L = w·B·F + w·ε
    # E[L] = 0
    # Under tilted measure: F ~ N(θ* w·B, I_k) for the scalar tilt case
    # But for multi-factor, the optimal tilt direction is w·B.

    load_vec = b.T @ ww  # (k,): loading of each factor on the portfolio loss

    if theta is None:
        mean_loss0 = 0.0  # since F has mean 0, ε has mean 0
        var_systematic = float(np.dot(load_vec, f_cov @ load_vec))
        if var_systematic <= 0.0:
            raise ValueError("systematic variance of portfolio loss is zero")
        theta = max(0.0, (threshold - mean_loss0) / var_systematic)

    tilt_direction = f_cov @ load_vec  # (k,) direction in factor space

    # --- Naive MC ---
    crude_losses = factor_model_sample(
        rng, n_crude, ww, b, p, df=df, factor_mean=0.0, factor_cov=f_cov
    )
    crude_hits = (crude_losses >= threshold).astype(np.float64)
    crude_est = float(np.mean(crude_hits))
    crude_var = float(np.var(crude_hits, ddof=1)) / n_crude

    # --- IS (tilted factors only) ---
    t_scale = np.sqrt(p * (df - 2.0) / df)

    # Draw tilted factors
    factor_tilt_mean = float(theta) * tilt_direction
    is_factors = rng.multivariate_normal(factor_tilt_mean, f_cov, size=n_is)

    # Idiosyncratic
    is_eps = rng.standard_t(df, size=(n_is, d)) * t_scale[np.newaxis, :]

    # Portfolio loss under IS
    is_systematic = is_factors @ b.T
    is_X = is_systematic + is_eps
    is_losses = np.asarray(is_X @ ww, dtype=np.float64)
    is_hits = (is_losses >= threshold).astype(np.float64)

    # Likelihood ratio: tilt only F.
    # dP/dQ = exp(-θ wᵀ B·F_tilted + ½ θ² ‖wᵀ B‖²)
    # where F_tilted ~ N(θ Σ_F wᵀ B, Σ_F).
    # For factor_cov=I: weight = exp(-θ load·F_tilted + ½ θ² ‖load‖²)
    log_wts: Array = np.zeros(n_is, dtype=np.float64)
    for i in range(n_is):
        log_wts[i] = -float(theta) * float(np.dot(load_vec, is_factors[i])) + 0.5 * float(
            theta
        ) ** 2 * float(np.dot(load_vec, f_cov @ load_vec))
    weights = np.asarray(np.exp(log_wts), dtype=np.float64)
    weighted_hits = weights * is_hits
    is_est = float(np.mean(weighted_hits))
    is_var = float(np.var(weighted_hits, ddof=1)) / n_is

    # VRF
    vrf: float | None = None
    infinite = False
    reason: str | None = None
    if is_var <= 0.0 and crude_var > 0.0:
        vrf = None
        infinite = True
        reason = "IS weighted outcomes have zero sample variance"
    elif is_var <= 0.0 or not math.isfinite(is_var):
        vrf = None
        reason = "IS variance is not positive"
    elif crude_var <= 0.0:
        vrf = None
        reason = "crude variance is not positive (no tail hits)"
    else:
        vrf = crude_var / is_var

    return {
        "method": "factor_model_exponential_tilting_is",
        "synthetic": True,
        "df": df,
        "threshold": float(threshold),
        "optimal_theta": float(theta),
        "n_is": n_is,
        "n_crude": n_crude,
        "crude_estimate": float(crude_est),
        "is_estimate": float(is_est),
        "crude_variance": float(crude_var),
        "is_variance": float(is_var),
        "variance_reduction_factor": vrf,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "Only systematic Gaussian factors are tilted (Glasserman-Li 2005). "
            "Idiosyncratic t-distributed innovations are untransformed.  SYNTHETIC."
        ),
    }


# ---------------------------------------------------------------------------
# 7. Subadditivity check for rate functions ────────────────────────────────
# ---------------------------------------------------------------------------


def rate_subadditivity(
    rate_sum: RateFn,
    rate_x: RateFn,
    rate_y: RateFn,
    t_values: Array,
    *,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Verify rate-function subadditivity for independent components.

    For independent random vectors X, Y, the Cramér rate function Λ* for
    the sum X+Y satisfies

        Λ*_{X+Y}(t) ≥ (Λ*_X □ Λ*_Y)(t) = inf_s { Λ*_X(s) + Λ*_Y(t−s) }

    This is the infimal-convolution property — a consequence of independence
    and the fact that the log-MGF of the sum is the sum of the log-MGFs.

    The check computes both sides on a grid of ``t_values`` and reports
    violations (instances where the inequality is not satisfied beyond
    numerical tolerance).

    Parameters
    ----------
    rate_sum : callable
        Rate function of X+Y (scalar argument).
    rate_x, rate_y : callable
        Rate functions of X and Y respectively (scalar arguments).
    t_values : (m,) float array
        Scalar t values at which to check.
    tol : float
        Tolerance for violation detection.

    Returns
    -------
    dict
        Keys: ``t_values``, ``rate_sum``, ``rate_convolution``, ``violations``,
        ``max_violation``, ``passed``.
    """
    tt = _as_array(t_values, "t_values")
    n = tt.size

    rate_sum_vals = np.empty(n, dtype=np.float64)
    conv_vals = np.empty(n, dtype=np.float64)

    for j, t_val in enumerate(tt):
        r_sum = rate_sum(np.array([t_val]))
        rate_sum_vals[j] = r_sum

        # Compute infimal convolution: inf_s { Λ*_X(s) + Λ*_Y(t-s) }
        # 1-d optimization over s.  (_t binds the loop variable per B023.)
        def conv_obj(s: float, _t: float = float(t_val)) -> float:
            return rate_x(np.array([s])) + rate_y(np.array([_t - s]))

        # Search over s ∈ [0, t] when rates are infinite outside [0, ∞).
        # For Gaussian case, unbounded is fine.
        result = optimize.minimize_scalar(
            conv_obj,
            bounds=(0.0, float(t_val)),
            method="bounded",
            options={"xatol": tol},
        )
        conv_vals[j] = float(result.fun)

    diff = rate_sum_vals - conv_vals
    violations = diff < -tol
    n_violations = int(np.sum(violations))
    max_violation = float(np.min(diff)) if n_violations > 0 else 0.0

    return {
        "method": "rate_function_subadditivity",
        "n_points": n,
        "t_values": [float(v) for v in tt.tolist()],
        "rate_sum": [float(v) for v in rate_sum_vals.tolist()],
        "rate_convolution": [float(v) for v in conv_vals.tolist()],
        "n_violations": n_violations,
        "max_violation": float(max_violation),
        "passed": n_violations == 0,
        "note": (
            "inf_{s} {rate_x(s) + rate_y(t-s)} ≤ rate_sum(t). "
            "Violations beyond numerical tolerance indicate a defect. "
            "SYNTHETIC."
        ),
    }
