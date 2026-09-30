"""Anytime-valid confidence sequences (CS) for the mean of a data stream.

A (1 - alpha) confidence sequence is a sequence of intervals (CI_t) with
time-uniform coverage P(forall t >= 1: mu_t in CI_t) >= 1 - alpha — valid at
ARBITRARY stopping times, with no peeking correction (Howard, Ramdas,
McAuliffe & Sekhon, 2021, "Time-uniform, nonparametric, nonasymptotic
confidence sequences", Annals of Statistics 49(2):1055-1080,
arXiv:1810.08240, Eq. (1) and Lemma 3).

1. Sub-Gaussian CS (Howard et al., 2021, arXiv:1810.08240). The estimand is
   mu_t = t^-1 sum_{i<=t} E_{i-1}[X_i]; S_t = sum_{i<=t}(X_i - E_{i-1}X_i) is
   sub-Gaussian with intrinsic time V_t = t*sigma^2 for a declared proxy
   variance sigma^2, and CI_t = Xbar_t -+ u(V_t)/t for a uniform boundary u
   with crossing probability alpha (their Eq. (3)):
   - "two_sided_mixture": the two-sided normal-mixture (mixture-of-
     martingales over lambda ~ N(0, 1/rho)) closed form, their Eq. (14):
     u(v) = sqrt((v + rho) * log((v + rho) / (alpha^2 * rho))). Tuning rho
     is optimized for a target intrinsic time via their Proposition 3(b)
     (Lambert W_{-1} formula) when not supplied.
   - "poly_stitching": the polynomial-stitching (finite-LIL) boundary of
     their Theorem 1 / Eqs. (8) and (10), one-sided, union-bounded over both
     tails at alpha/2 each (their Section 2 recipe). Defaults eta=2, s=1.4
     reproduce their worked example Eq. (11).
   - "poly_hedge_epsilon": the hedge-epsilon boundary of Jamieson, Malloy,
     Nowak & Bubeck (2014, COLT, arXiv:1405.3269, Lemma 1), as tabulated in
     Howard et al. (2021, App. G, Eq. (134)), translated to intrinsic time
     (tuning rho = sigma^2 converts v to nominal time v/rho), stitched over
     geometric epochs eta^k of intrinsic time with polynomial alpha-spending
     h(k) = (k+1)^s * zeta(s) (sum_k alpha/h(k) = alpha). One-sided; the
     two-sided CS splits alpha/2 per tail.
2. Betting CS [WSR] (Waudby-Smith & Ramdas, 2024, "Estimating means of
   bounded random variables by betting", JRSS-A 187(1):1-27,
   arXiv:2010.09686). For X_t in [lower, upper], the hedged capital process
   (their Theorem 3, Eqs. (24)-(26))
       K_t^+(m)  = prod_{i<=t} (1 + lambda_i^+(m) (Y_i - m)),
       K_t^-(m)   = prod_{i<=t} (1 - lambda_i^-(m) (Y_i - m)),
       K_t^±(m) = max{theta K_t^+(m), (1-theta) K_t^-(m)},
   on Y = (X - lower)/(upper - lower), with predictable truncated bets
   lambda_i^+(m) = min(lambda~_i, c/m), lambda_i^-(m) = min(lambda~_i, c/(1-m))
   and their recommended plug-in lambda~_t = sqrt(2 log(2/alpha) /
   (sigma^hat_{t-1} t log(t+1))). Truncation c in (0, 1) keeps every wealth
   factor >= 1 - c > 0 (nonnegative wealth, no bankruptcy). K_t^±(mu) is
   upper-bounded by the test supermartingale theta K^+ + (1-theta) K^-, so
   Ville gives time-uniform coverage of CI_t = {m : K_t^±(m) < 1/alpha},
   which is an interval (their Lemma 2: K^+ nonincreasing, K^- nondecreasing
   in m). Endpoints are located by bisection and reported on the REJECTED
   side of the bracket, so the reported interval is a superset of the exact
   CS — conservative, coverage can only improve.
3. Empirical-Bernstein CS for bounded data (Howard et al., 2021, Theorem 4):
   with any predictable [a, b]-valued predictor X^hat_i (default: running
   mean Xbar_{i-1}, midpoint at i=1) and V^hat_t = sum_{i<=t}(X_i - X^hat_i)^2,
       CI_t = Xbar_t -+ u(V^hat_t)/t
   has coverage >= 1 - 2*alpha_u for any sub-exponential uniform boundary u
   with crossing probability alpha_u at scale c = b - a. We plug in the
   polynomial-stitching sub-gamma boundary at scale c (valid as
   sub-exponential at the same scale by their Table 1 row (8)) with
   alpha_u = alpha/2, mirroring their worked example Eq. (24). The radius
   scales with the EMPIRICAL variance, so the CS adapts to low-variance
   streams inside a wide declared support.

Heavy tails: methods 2 and 3 need a declared bounded support; method 1 needs
a finite sub-Gaussian proxy variance. A stream with only a finite variance
(e.g. Student-t with df = 3) admits none of these directly — the documented
relaxation is a symmetric clip to [−B, B], which preserves the mean of a
symmetric distribution, after which methods 2 and 3 apply to the clipped
stream. Truly robust heavy-tailed CS (e.g. Catoni-style) are out of scope.

Honesty: all Monte-Carlo evidence in the accompanying tests uses seeded
SYNTHETIC streams and is a correctness check of the error control, never
market evidence. Intervals here are proper inferential objects (time-uniform
coverage), not Sharpe/P&L style performance claims, and nothing here implies
live-trading capability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw, zeta
from scipy.stats import norm

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "ConfidenceSequence",
    "check_time_uniform_coverage",
    "cs_widths",
    "empirical_bernstein_cs",
    "fixed_sample_ci_width",
    "hedge_epsilon_boundary",
    "normal_mixture_boundary",
    "optimal_mixture_rho",
    "poly_hedge_epsilon_boundary",
    "poly_stitching_boundary",
    "subgaussian_cs",
    "width_decay_slope",
    "width_ratio_vs_fixed",
    "wsr_cs",
]

# Bisection cost of wsr_cs is O(T^2) cells per iterate; cap both the stream
# length and the working-set size so a mis-sized call fails closed instead of
# exhausting memory.
_WSR_MAX_T = 4096
_WSR_CELL_BUDGET = 4_000_000


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return v


def _check_stream(x: Array | Iterable[float], name: str = "x") -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1, 2) or arr.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D (T,) or 2-D (R, T) array")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_bounds(lower: float, upper: float) -> tuple[float, float]:
    lo, hi = float(lower), float(upper)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("lower/upper must be finite with lower < upper")
    return lo, hi


def _check_support(arr: Array, lower: float, upper: float) -> None:
    if bool(np.any(arr < lower)) or bool(np.any(arr > upper)):
        raise ValueError("observations outside declared [lower, upper] support")


def _check_intrinsic_time(v: Array | Iterable[float]) -> Array:
    arr = np.asarray(v, dtype=float)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("v must be nonempty and finite")
    if bool(np.any(arr < 0.0)):
        raise ValueError("intrinsic time v must be nonnegative")
    return arr


def _check_epoch_params(eta: float, s: float) -> tuple[float, float]:
    e = float(eta)
    if not np.isfinite(e) or e <= 1.0:
        raise ValueError("eta must be finite and > 1")
    sp = float(s)
    if not np.isfinite(sp) or sp <= 1.0:
        raise ValueError("s must be finite and > 1 (zeta(s) must converge)")
    return e, sp


def normal_mixture_boundary(v: Array | Iterable[float], alpha: float, rho: float) -> Array:
    """Two-sided normal-mixture sub-Gaussian uniform boundary (Eq. (14)).

    u(v) = sqrt((v + rho) * log((v + rho) / (alpha^2 * rho))), the
    mixture-of-martingales boundary for lambda ~ N(0, 1/rho) over BOTH signs
    (Howard et al. 2021, Section 3.2): P(exists t: |S_t| >= u(V_t)) <= alpha
    for any sub-Gaussian (S_t, V_t) pair. ``rho`` is the tuning parameter in
    intrinsic-time units; the boundary is valid for every rho > 0 and tightest
    near the design time (their Proposition 3).
    """
    a = _check_alpha(alpha)
    r = _check_positive(rho, "rho")
    vv = _check_intrinsic_time(v)
    total = vv + r
    return np.asarray(np.sqrt(total * np.log(total / (a * a * r))), dtype=np.float64)


def optimal_mixture_rho(v_opt: float, alpha: float) -> float:
    """Mixture tuning rho optimizing the boundary (14) at intrinsic time v_opt.

    Howard et al. (2021, Proposition 3(b)): the optimal rho satisfies
    (v_opt + rho)/rho = -W_{-1}(-1/(e*k)) with k = (1/alpha)^2 (l0 = 1),
    i.e. rho = v_opt / (r* - 1). As alpha -> 1 the ratio r* -> 1 and rho is
    clamped to stay finite (the boundary remains valid for any rho > 0, only
    the tuning degrades).
    """
    v = _check_positive(v_opt, "v_opt")
    a = _check_alpha(alpha)
    k = 1.0 / (a * a)
    w = float(np.real(lambertw(-1.0 / (np.e * k), -1)))
    r_star = max(-w, 1.0 + 1e-9)
    return float(v / (r_star - 1.0))


def poly_stitching_boundary(
    v: Array | Iterable[float],
    alpha: float,
    *,
    c: float = 0.0,
    eta: float = 2.0,
    s: float = 1.4,
    m: float = 1.0,
) -> Array:
    """One-sided polynomial-stitching sub-gamma boundary (Theorem 1, Eqs. (8)/(10)).

    u(v) = S_alpha(v | m) with, writing ell(v) = s*log(log(eta*v/m)) +
    log(zeta(s)/(alpha*log^s(eta))), k1 = (eta^(1/4) + eta^(-1/4))/sqrt(2),
    k2 = (sqrt(eta) + 1)/2:

        S_alpha(v) = sqrt(k1^2 v ell(v) + k2^2 c^2 ell(v)^2) + k2 c ell(v).

    Howard et al. (2021, Theorem 1): v -> S_alpha(v | m) is a sub-gamma
    uniform boundary with crossing probability alpha at scale c; c = 0 gives
    the sub-Gaussian (finite-LIL) case, matching their Eq. (11) at the
    defaults eta=2, s=1.4, m=1. Since ell(v | m) >= log(zeta(s)/alpha) > 0
    for v >= m, the budget term is strictly positive; a nonpositive value
    (only reachable with invalid parameters) raises.
    """
    a = _check_alpha(alpha)
    scale = float(c)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("c must be finite and >= 0")
    e, sp = _check_epoch_params(eta, s)
    floor = _check_positive(m, "m")
    vv = np.maximum(_check_intrinsic_time(v), floor)
    log_eta = float(np.log(e))
    budget = float(np.log(float(zeta(sp)) / (a * log_eta**sp)))
    ell = sp * np.log(np.log(e * vv / floor)) + budget
    if bool(np.any(ell <= 0.0)):
        raise ValueError("stitching budget ell(v) must be positive")
    k1 = (e**0.25 + e**-0.25) / float(np.sqrt(2.0))
    k2 = (float(np.sqrt(e)) + 1.0) / 2.0
    out = np.sqrt((k1 * k1) * vv * ell + (k2 * scale) ** 2 * ell * ell) + k2 * scale * ell
    return np.asarray(out, dtype=np.float64)


def _hedge_delta(alpha: float | Array, epsilon: float) -> Array:
    """Jamieson et al. (2014, Lemma 1) budget: crossing prob ((2+e)/e)(d/log(1+e))^(1+e)."""
    out = np.log1p(epsilon) * (alpha * epsilon / (2.0 + epsilon)) ** (1.0 / (1.0 + epsilon))
    return np.asarray(out, dtype=np.float64)


def hedge_epsilon_boundary(
    v: Array | Iterable[float],
    alpha: float,
    *,
    rho: float,
    epsilon: float = 0.5,
) -> Array:
    """One-sided hedge-epsilon sub-Gaussian boundary (Jamieson et al. 2014).

    Lemma 1 of Jamieson, Malloy, Nowak & Bubeck (2014, arXiv:1405.3269), as
    tabulated in Howard et al. (2021, App. G, Eq. (134) / Table 2), written in
    intrinsic time with rho the per-observation proxy variance (nominal time
    is v/rho):

        u(v) = (1+sqrt(eps)) sqrt(2 (1+eps) v log(log((1+eps) v/rho) / delta)),
        delta = log(1+eps) * (alpha*eps/(2+eps))^(1/(1+eps)),

    so that P(exists t: S_t >= u(V_t)) <= alpha. The budget choice makes the
    paper's crossing bound exactly alpha; delta <= log(1+eps) for every
    alpha < 1, so the boundary is well-defined and nondecreasing from the
    clamp point v = rho onward (v is floored at rho, which only enlarges the
    boundary and preserves validity).
    """
    a = _check_alpha(alpha)
    r = _check_positive(rho, "rho")
    eps = _check_positive(epsilon, "epsilon")
    vv = np.maximum(_check_intrinsic_time(v), r)
    delta = _hedge_delta(a, eps)
    ratio = np.log((1.0 + eps) * vv / r) / delta
    out = (1.0 + np.sqrt(eps)) * np.sqrt(2.0 * (1.0 + eps) * vv * np.log(ratio))
    return np.asarray(out, dtype=np.float64)


def poly_hedge_epsilon_boundary(
    v: Array | Iterable[float],
    alpha: float,
    *,
    rho: float,
    epsilon: float = 0.5,
    eta: float = 2.0,
    s: float = 1.4,
) -> Array:
    """Polynomial-budget stitching of hedge-epsilon boundaries (finite-LIL).

    Epochs E_k = {rho*eta^k <= v < rho*eta^(k+1)} (disjoint, covering
    v >= rho) each carry the hedge-epsilon boundary at budget
    alpha_k = alpha / ((k+1)^s * zeta(s)); sum_k alpha_k = alpha, so the
    union bound over epochs gives crossing probability <= alpha. The
    polynomial spending reproduces the poly-stitching rate
    u(v) = O(sqrt(v log log v)) with the (1+sqrt(eps)) hedge slack instead of
    the stitching constants (construction as in Howard et al. 2021, Theorem 1,
    with the per-epoch linear boundaries replaced by Jamieson et al. 2014's
    hedge boundary; see :func:`hedge_epsilon_boundary`). v is floored at rho.
    """
    a = _check_alpha(alpha)
    r = _check_positive(rho, "rho")
    eps = _check_positive(epsilon, "epsilon")
    e, sp = _check_epoch_params(eta, s)
    vv = np.maximum(_check_intrinsic_time(v), r)
    epoch = np.floor(np.log(vv / r) / np.log(e))
    alpha_k = a / ((epoch + 1.0) ** sp * float(zeta(sp)))
    delta = _hedge_delta(alpha_k, eps)
    ratio = np.log((1.0 + eps) * vv / r) / delta
    out = (1.0 + np.sqrt(eps)) * np.sqrt(2.0 * (1.0 + eps) * vv * np.log(ratio))
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class ConfidenceSequence:
    """Output of the CS builders: per-time interval endpoints plus metadata.

    ``lower`` / ``upper`` align with the input stream along the last axis:
    shape (T,) for a single stream, (R, T) for a batch of independent
    replicates. ``method`` records the construction for receipts/benches.
    """

    lower: Array
    upper: Array
    alpha: float
    method: str

    @property
    def widths(self) -> Array:
        """Interval width upper - lower at every sample time."""
        return np.asarray(self.upper - self.lower, dtype=np.float64)

    @property
    def n(self) -> int:
        """Number of sample times T along the last axis."""
        return int(self.lower.shape[-1])


_SUBGAUSSIAN_BOUNDARIES = ("two_sided_mixture", "poly_stitching", "poly_hedge_epsilon")


def subgaussian_cs(
    x: Array | Iterable[float],
    *,
    sigma: float,
    alpha: float = 0.05,
    boundary: str = "two_sided_mixture",
    rho: float | None = None,
    eta: float = 2.0,
    s: float = 1.4,
    m: float | None = None,
    epsilon: float = 0.5,
) -> ConfidenceSequence:
    """Sub-Gaussian mixture/stitching CS for the stream mean (Howard et al. 2021).

    ``sigma`` is the DECLARED sub-Gaussian proxy standard deviation of the
    increments: validity requires E[exp(l (X_i - E_{i-1} X_i)) | F_{i-1}] <=
    exp(l^2 sigma^2 / 2) a.s. for all l (Gaussian with sd <= sigma; bounded
    X in [a, b] with sigma = (b-a)/2 by Hoeffding's lemma). Intrinsic time is
    V_t = t*sigma^2 and CI_t = Xbar_t -+ u(V_t)/t.

    ``boundary`` selects u (see module docstring for citations):
    - "two_sided_mixture": Eq. (14), two-sided at level alpha directly;
      tuning ``rho`` defaults to :func:`optimal_mixture_rho` at the planned
      horizon v_opt = T*sigma^2 (a deterministic design constant — the
      boundary stays valid for every rho > 0 and for sampling beyond T).
    - "poly_stitching" / "poly_hedge_epsilon": one-sided boundaries run at
      alpha/2 per tail (union bound); ``rho`` is ignored, the stitching floor
      ``m`` defaults to sigma^2 (one observation of intrinsic time), and
      ``eta``/``s``/``epsilon`` tune the epochs and the hedge slack.

    Accepts (T,) or (R, T) input; returns endpoints of matching shape.
    """
    a = _check_alpha(alpha)
    sig = _check_positive(sigma, "sigma")
    arr = _check_stream(x)
    kind = str(boundary)
    if kind not in _SUBGAUSSIAN_BOUNDARIES:
        raise ValueError(f"boundary must be one of {_SUBGAUSSIAN_BOUNDARIES}")
    n = int(arr.shape[-1])
    t = np.arange(1, n + 1, dtype=float)
    v = (sig * sig) * t
    mean = np.cumsum(arr, axis=-1) / t
    u: Array
    if kind == "two_sided_mixture":
        r = _check_positive(rho, "rho") if rho is not None else None
        if r is None:
            r = optimal_mixture_rho(float(n) * sig * sig, a)
        u = normal_mixture_boundary(v, a, r)
    elif kind == "poly_stitching":
        m_eff = _check_positive(m, "m") if m is not None else sig * sig
        u = poly_stitching_boundary(v, a / 2.0, c=0.0, eta=eta, s=s, m=m_eff)
    else:
        u = poly_hedge_epsilon_boundary(v, a / 2.0, rho=sig * sig, epsilon=epsilon, eta=eta, s=s)
    radius = np.asarray(u / t, dtype=np.float64)
    return ConfidenceSequence(
        lower=np.asarray(mean - radius, dtype=np.float64),
        upper=np.asarray(mean + radius, dtype=np.float64),
        alpha=a,
        method=f"subgaussian:{kind}",
    )


def _wsr_lambda_tilde(y: Array, alpha: float) -> Array:
    """Recommended predictable plug-in bets (Waudby-Smith & Ramdas 2024, Eq. (26)).

    lambda~_t = sqrt(2 log(2/alpha) / (sigma^hat^2_{t-1} * t * log(t+1))) with
    regularized running estimates on the [0, 1] scale:
    mu^hat_t = (1/2 + sum_{i<=t} y_i)/(t+1),
    sigma^hat^2_t = (1/4 + sum_{i<=t}(y_i - mu^hat_i)^2)/(t+1),
    sigma^hat^2_0 = 1/4. Predictable: lambda~_t uses y_1..y_{t-1} only.
    """
    n = int(y.shape[-1])
    t = np.arange(1, n + 1, dtype=float)
    mu_hat = (0.5 + np.cumsum(y, axis=-1)) / (t + 1.0)
    sig2 = (0.25 + np.cumsum((y - mu_hat) ** 2, axis=-1)) / (t + 1.0)
    lead = np.full(y.shape[:-1] + (1,), 0.25, dtype=float)
    sig2_prev = np.concatenate([lead, sig2[..., :-1]], axis=-1)
    lam = np.sqrt(2.0 * np.log(2.0 / alpha) / (sig2_prev * t * np.log(t + 1.0)))
    return np.asarray(lam, dtype=np.float64)


def _wsr_log_wealth(
    y: Array,
    lam: Array,
    m_cand: Array,
    truncation: float,
    side: str,
) -> Array:
    """log K_t^{side}(m_t) for a per-time candidate vector m (shape (C, T)).

    Builds the (C, T, T) log-factor matrix (row t evaluates every factor i
    at that row's own candidate m_t), masks i > t, and reads the diagonal of
    the prefix sums. Factors are >= 1 - truncation > 0 by the bet caps, so
    log1p is safe and the wealth cannot go negative or overflow in log space.
    """
    mv = m_cand[..., None]
    lam_row = lam[..., None, :]
    y_row = y[..., None, :]
    if side == "+":
        lam_eff = np.minimum(lam_row, truncation / mv)
        arg = lam_eff * (y_row - mv)
    else:
        lam_eff = np.minimum(lam_row, truncation / (1.0 - mv))
        arg = -lam_eff * (y_row - mv)
    total = np.cumsum(np.tril(np.log1p(arg)), axis=-1)
    idx = np.arange(int(total.shape[-1]))
    return np.asarray(total[..., idx, idx], dtype=np.float64)


def wsr_cs(
    x: Array | Iterable[float],
    *,
    lower: float,
    upper: float,
    alpha: float = 0.05,
    theta: float = 0.5,
    truncation: float = 0.5,
    n_bisect: int = 30,
) -> ConfidenceSequence:
    """Betting (WSR) hedged-capital CS for a bounded stream's mean.

    Waudby-Smith & Ramdas (2024, Theorem 3): inverting the hedged capital
    process at 1/alpha yields a time-uniform (1 - alpha) CS that is an
    interval at every t (their Lemma 2). Observations must lie in
    [lower, upper] — out-of-support values raise (fail-closed: the payoff
    truncation only guarantees nonnegative wealth on the declared support).

    ``theta`` splits initial wealth between the two hedges; ``truncation`` is
    the bet cap c in (0, 1) (their recommended 1/2 or 3/4); ``n_bisect``
    bisection steps locate each endpoint to 2^-n_bisect of the support width.
    Reported endpoints sit on the rejected side of the final bracket, so the
    returned interval is a superset of the exact CS (conservative). Cost is
    O(T^2) per bisection iterate; streams longer than ``_WSR_MAX_T`` raise.
    Accepts (T,) or (R, T) input (rows are independent replicates).
    """
    a = _check_alpha(alpha)
    lo_b, up_b = _check_bounds(lower, upper)
    th = float(theta)
    if not np.isfinite(th) or not 0.0 < th < 1.0:
        raise ValueError("theta must lie in the open interval (0, 1)")
    c = float(truncation)
    if not np.isfinite(c) or not 0.0 < c < 1.0:
        raise ValueError("truncation must lie in the open interval (0, 1)")
    steps = int(n_bisect)
    if steps < 1:
        raise ValueError("n_bisect must be >= 1")
    arr = _check_stream(x)
    _check_support(arr, lo_b, up_b)
    n = int(arr.shape[-1])
    if n > _WSR_MAX_T:
        raise ValueError(f"wsr_cs supports streams up to T={_WSR_MAX_T} (O(T^2) bisection)")
    y = (arr - lo_b) / (up_b - lo_b)
    rows = np.atleast_2d(y)
    lam = _wsr_lambda_tilde(rows, a)
    thr_plus = float(np.log(1.0 / (a * th)))
    thr_minus = float(np.log(1.0 / (a * (1.0 - th))))
    chunk = max(1, _WSR_CELL_BUDGET // max(n * n, 1))
    lo_out = np.empty_like(rows)
    hi_out = np.empty_like(rows)
    for start in range(0, int(rows.shape[0]), chunk):
        y_c = rows[start : start + chunk]
        lam_c = lam[start : start + chunk]
        a_lo = np.zeros_like(y_c)
        b_lo = np.ones_like(y_c)
        a_hi = np.zeros_like(y_c)
        b_hi = np.ones_like(y_c)
        for _ in range(steps):
            mid = 0.5 * (a_lo + b_lo)
            rej = _wsr_log_wealth(y_c, lam_c, mid, c, "+") >= thr_plus
            a_lo = np.where(rej, mid, a_lo)  # K^+ nonincreasing: reject small m
            b_lo = np.where(rej, b_lo, mid)
            mid = 0.5 * (a_hi + b_hi)
            rej = _wsr_log_wealth(y_c, lam_c, mid, c, "-") >= thr_minus
            a_hi = np.where(rej, a_hi, mid)  # K^- nondecreasing: reject large m
            b_hi = np.where(rej, mid, b_hi)
        lo_out[start : start + chunk] = a_lo
        hi_out[start : start + chunk] = b_hi
    width = up_b - lo_b
    lower_cs = np.asarray(lo_b + lo_out * width, dtype=np.float64).reshape(arr.shape)
    upper_cs = np.asarray(lo_b + hi_out * width, dtype=np.float64).reshape(arr.shape)
    return ConfidenceSequence(lower=lower_cs, upper=upper_cs, alpha=a, method="wsr:hedged-capital")


def empirical_bernstein_cs(
    x: Array | Iterable[float],
    *,
    lower: float,
    upper: float,
    alpha: float = 0.05,
    eta: float = 2.0,
    s: float = 1.4,
    m: float | None = None,
) -> ConfidenceSequence:
    """Empirical-Bernstein CS for a bounded stream (Howard et al. 2021, Thm 4).

    Radius u(V^hat_t)/t with V^hat_t = sum_{i<=t}(X_i - X^hat_i)^2, the
    predictable predictor X^hat_1 = (a+b)/2, X^hat_i = Xbar_{i-1}, and u the
    polynomial-stitching sub-gamma boundary at scale c = b - a and crossing
    probability alpha/2 (valid as a sub-exponential boundary at the same
    scale by their Table 1 row (8); the alpha/2 split turns Theorem 4's
    1 - 2*alpha_u into 1 - alpha). The intrinsic-time floor ``m`` defaults to
    c^2, matching the V^hat | 1 clamp of their example Eq. (24). Endpoints
    are clipped to the declared support (mu_t lies in [a, b], so clipping can
    only improve coverage). Adapts to the empirical variance: much tighter
    than a Hoeffding-proxy sub-Gaussian CS on low-variance streams.
    Out-of-support observations raise (fail-closed). Accepts (T,) or (R, T).
    """
    a = _check_alpha(alpha)
    lo_b, up_b = _check_bounds(lower, upper)
    arr = _check_stream(x)
    _check_support(arr, lo_b, up_b)
    n = int(arr.shape[-1])
    t = np.arange(1, n + 1, dtype=float)
    mean = np.cumsum(arr, axis=-1) / t
    mid = 0.5 * (lo_b + up_b)
    lead = np.full(arr.shape[:-1] + (1,), mid, dtype=float)
    pred = np.concatenate([lead, mean[..., :-1]], axis=-1)
    v_hat = np.cumsum((arr - pred) ** 2, axis=-1)
    c = up_b - lo_b
    m_eff = _check_positive(m, "m") if m is not None else c * c
    radius = np.asarray(
        poly_stitching_boundary(v_hat, a / 2.0, c=c, eta=eta, s=s, m=m_eff) / t,
        dtype=np.float64,
    )
    return ConfidenceSequence(
        lower=np.asarray(np.maximum(mean - radius, lo_b), dtype=np.float64),
        upper=np.asarray(np.minimum(mean + radius, up_b), dtype=np.float64),
        alpha=a,
        method="empirical-bernstein:stitching",
    )


def check_time_uniform_coverage(
    lower: Array | Iterable[float],
    upper: Array | Iterable[float],
    mu: float | Array | Iterable[float],
) -> dict[str, object]:
    """Time-uniform coverage diagnostic for CS endpoints against a true mean.

    ``lower``/``upper`` are (T,) or (R, T) endpoints; ``mu`` is the estimand:
    a scalar (constant mean), a (T,) mean path, or an (R, T) array
    broadcastable to the endpoints. A replicate counts as covered only if
    mu_t lies in [lower_t, upper_t] at EVERY sample time (the time-uniform
    event); per-time marginal coverage is reported separately.

    Keys: ``time_uniform_coverage`` (fraction of replicates covered at all
    times), ``violation_rate`` (= 1 - that), ``n_reps``,
    ``per_time_coverage`` (T,), ``first_violations`` (n_reps,) int array of
    0-based first violation indices, -1 where never violated. NaN in any
    input raises (fail-closed); +/-inf endpoints are allowed and comparable.
    """
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    if lo.shape != up.shape or lo.ndim not in (1, 2) or lo.size == 0:
        raise ValueError("lower/upper must be nonempty matching (T,) or (R, T) arrays")
    if bool(np.any(np.isnan(lo))) or bool(np.any(np.isnan(up))):
        raise ValueError("CS endpoints must not contain NaN")
    mu_arr = np.asarray(mu, dtype=float)
    if bool(np.any(np.isnan(mu_arr))):
        raise ValueError("mu must not contain NaN")
    try:
        mu_b = np.broadcast_to(mu_arr, lo.shape)
    except ValueError as exc:
        raise ValueError("mu must be a scalar or broadcastable to the CS shape") from exc
    covered = (lo <= mu_b) & (mu_b <= up)
    cov2 = np.atleast_2d(covered)
    missed = ~cov2
    ever = missed.any(axis=-1)
    first = np.where(ever, np.argmax(missed, axis=-1), -1).astype(np.int64)
    return {
        "time_uniform_coverage": float(1.0 - float(ever.mean())),
        "violation_rate": float(ever.mean()),
        "n_reps": int(cov2.shape[0]),
        "per_time_coverage": np.asarray(cov2.mean(axis=0), dtype=np.float64),
        "first_violations": np.asarray(first, dtype=np.int64),
    }


def cs_widths(lower: Array | Iterable[float], upper: Array | Iterable[float]) -> Array:
    """Pointwise interval width upper - lower (shape-preserving)."""
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    if lo.shape != up.shape or lo.size == 0:
        raise ValueError("lower/upper must be nonempty matching shapes")
    return np.asarray(up - lo, dtype=np.float64)


def width_decay_slope(
    lower: Array | Iterable[float],
    upper: Array | Iterable[float],
    *,
    from_frac: float = 0.25,
) -> float:
    """Log-log decay slope of the CS width over the tail of the stream.

    Least-squares slope of log(width_t) against log(t) for
    t >= ceil(from_frac*T); a 1/sqrt(t)-shrinking CS returns about -0.5
    (mixture/stitching boundaries add a slow loglog factor, pushing the
    slope slightly above -0.5). 1-D (T,) endpoints only; non-positive or
    non-finite widths on the fit window raise (fail-closed).
    """
    w = cs_widths(lower, upper)
    if w.ndim != 1:
        raise ValueError("width_decay_slope expects 1-D (T,) endpoints")
    frac = float(from_frac)
    if not np.isfinite(frac) or not 0.0 <= frac < 1.0:
        raise ValueError("from_frac must lie in [0, 1)")
    n = int(w.size)
    t0 = max(2, int(np.ceil(frac * n)))
    if n - t0 + 1 < 2:
        raise ValueError("stream too short for a slope fit")
    tail = w[t0 - 1 :]
    if not bool(np.all(np.isfinite(tail))) or bool(np.any(tail <= 0.0)):
        raise ValueError("widths must be finite and positive on the fit window")
    lt = np.log(np.arange(t0, n + 1, dtype=float))
    lw = np.log(tail)
    denom = float(np.mean(lt * lt) - np.mean(lt) ** 2)
    slope = float(np.mean(lt * lw) - np.mean(lt) * np.mean(lw)) / denom
    return float(slope)


def fixed_sample_ci_width(
    x: Array | Iterable[float],
    alpha: float,
    *,
    sigma: float | None = None,
) -> float:
    """Width of the fixed-sample normal CI at the final sample size.

    2 * z_{1-alpha/2} * sd / sqrt(T), with sd the declared ``sigma`` or the
    sample standard deviation (ddof=1). This is the pointwise (NOT
    anytime-valid) benchmark used by :func:`width_ratio_vs_fixed`; 1-D input.
    """
    a = _check_alpha(alpha)
    arr = _check_stream(x)
    if arr.ndim != 1:
        raise ValueError("fixed_sample_ci_width expects a 1-D (T,) stream")
    n = int(arr.shape[-1])
    sd = float(sigma) if sigma is not None else float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("sigma must be finite and > 0")
    z = float(norm.ppf(1.0 - a / 2.0))
    return float(2.0 * z * sd / np.sqrt(n))


def width_ratio_vs_fixed(
    lower: Array | Iterable[float],
    upper: Array | Iterable[float],
    x: Array | Iterable[float],
    alpha: float,
    *,
    sigma: float | None = None,
) -> float:
    """Terminal CS width divided by the fixed-sample CI width (same alpha).

    The price of anytime validity: Howard et al. (2021, Section 1) report the
    mixture CS stays within a factor of about two of the fixed-sample width
    over many orders of magnitude in t, so values near 1-2 at large T are
    expected; a growing ratio signals a mis-tuned boundary.
    """
    w = cs_widths(lower, upper)
    if w.ndim != 1:
        raise ValueError("width_ratio_vs_fixed expects 1-D (T,) endpoints")
    fixed = fixed_sample_ci_width(x, alpha, sigma=sigma)
    return float(w[-1] / fixed)
