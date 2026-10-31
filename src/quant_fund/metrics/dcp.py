"""DCP: Distribution-aware Conformal Prediction. No Sharpe.

Implements the Distribution-aware Conformal Prediction (DCP) framework of

    Schweizer, D., Kuhn, P., Sharma, J., Dubey, S., von Ramin, M. &
    Brockt-Haßauer, C. (2026). "Distribution-Aware Conformal Prediction: A
    Framework for generating efficient prediction intervals for time series."
    arXiv:2605.26569 [stat.ML].
    Citation verified against https://arxiv.org/abs/2605.26569 and the full
    HTML text (fetched 2026-09-30): title, author list and the four-step
    framework (distribution-generating predictor -> distribution-aware
    nonconformity score -> global conformal threshold -> numerical inversion
    of the prediction set) all match this implementation.

Paper machinery implemented here (equation/section numbers refer to the
fetched text)
------------------------------------------------------------------------------

1. Predictor abstraction (paper Sec. 3 step i + Sec. 4.3): any
   distribution-generating predictor (DGP) — MC dropout, deep/bootstrap
   ensembles, or quantile regression — is a black box that produces a fixed
   size-M draw vector ``yhat(x) = [(yhat(x))_j]_{j=1..M}`` per input (paper
   Eq. 4). Quantile-regression predictors supply their quantile grid as a
   deterministic pseudo-draw vector (uniform over tau), exactly as the paper
   does for its 99-quantile head. ``DrawPredictor`` maps an ``(n, d)``
   feature matrix to an ``(n, M)`` draw matrix; ``predict_draws`` validates.

2. Nonconformity-score registry (paper Sec. 4.4): three generic families,
   all implementable as ``ScoreFn`` callables ``s(y, yhat)`` vectorized over
   candidate ``y`` values against one draw vector —

   - ``"residual"``   s_R  = |y - mu_hat|,          mu_hat = mean of draws
                                                        (Eq. 13);
   - ``"zscore"``     s_z  = |y - mu_hat| / sigma,  sigma = draw std (Eq. 14)
                      — the variance-scaled MC-CP-style score of Bethell et
                      al. (2024);
   - ``"interval"`` / ``"hdi"`` (+ ``*_scaled``): the interval-violation score
                      s_int = max{C_low - y, y - C_up} (Eq. 15) against a base
                      inner band — empirical alpha/2 / 1-alpha/2 quantiles of
                      the draws, or the (1-alpha) highest-density interval.
                      Negative inside the band so qhat < 0 shrinks the set.
                      The HDI variant is exactly the CMC score of Mehdiyev et
                      al. (2025); the quantile-band variant is exactly the CQR
                      score of Romano et al. (2019). Scaled variants divide by
                      the base-band width (standardize-rescale, Sec. 4.4.2);
   - ``"knn"``        s_KNN = d_k(y, yhat) / d_tilde(yhat) (Eqs. 16-18): the
                      median of the k nearest |y - yhat_j| distances (the
                      paper's argmin-of-absolute-deviations form is the
                      median), normalized by the median of all M^2 pairwise
                      distances inside the draw vector. Shape-sensitive and
                      possibly nonmonotone in y. Paper uses k = 10.

   ``register_score`` plugs in arbitrary user ``ScoreFn``s; conformal
   validity only needs the same score at calibration and inference, with any
   strictly positive predictor-only scaling folded into the score.

3. Numerical inversion (paper Sec. 3 step iv + Sec. 3.1-3.2 + Table 1):
   ``invert_interval`` solves f(y) = s(y, yhat) - qhat = 0 by evaluating f on
   an ordered symmetric geometric grid ``median(yhat) + {0, +-h0 gamma^j}``
   (defaults h0 = 1e-6, gamma = 1.167, tol = 1e-10, depth = 100 — the paper's
   Table 1 constants), detecting sign changes, taking the OUTERMOST pair of
   crossings as brackets (nonmonotone-safe), and refining each bracket by
   bisection to ``tol``. Failure policy is the paper's Sec. 3.2 verbatim:
   fewer than two sign changes triggers a retry that shrinks h0 and grows
   depth; a single persistent sign change yields a degenerate interval at
   that root; no sign change yields a degenerate interval at the predictive
   median. Degeneracy is always surfaced in ``IntervalResult.status`` /
   ``n_roots`` — never silent. ``strict=True`` instead raises
   ``NonInvertibleScoreError``; non-finite score maps always raise.

4. Calibration (paper Alg. 1 lines 5-9): ``dcp_fit`` scores the calibration
   set and takes qhat as the finite-sample split-conformal quantile
   ceil((n+1)(1-alpha))-th smallest score — reused by import from
   ``metrics.conformal.conformal_quantile``, never reimplemented.
   ``dcp_online_intervals`` is the paper's Sec. 4 sliding-window (adaptive /
   prequential) variant: the window is seeded with the calibration scores and
   re-estimates qhat after each revealed test target.

5. Interval efficiency metrics (paper Sec. 4.5): empirical coverage / PICP
   (Eq. 19, via ``metrics.conformal``), the minimal acceptable coverage
   C_a = 1 - alpha - zeta sqrt(alpha(1-alpha)/N_t) with zeta = 1.645
   (Eq. 20), PINAW (Eq. 21-23), the width coefficient of variation CV_delta
   (Eq. 24), the Winkler interval score W_i = delta_i + e_i with miss penalty
   slope 2/alpha (Eqs. 25-26), and the paper's modified mean Winkler score

       MMW = (1/N) sum_i (delta_i + P_uc e_i),
       Delta C = max(0, C_a - C_hat),  rho = Delta C / (1 - C_hat),
       P_uc = exp(kappa rho),  kappa = 2                          (Eqs. 27-28)

   which amplifies the per-unit miss cost the further empirical coverage
   falls below C_a.

6. ``dcp_battery`` / ``bench_dcp``: predictor-score batteries on planted
   SYNTHETIC DGPs and flat ``dict[str, float]`` bench output, comparing
   against the split-conformal residual baseline and a CQR-style baseline —
   both composed from ``metrics.conformal`` (``cqr_scores``,
   ``conformal_quantile``, ``expand_interval``), not reimplemented.

Composition notes: the conformal quantile, CQR violation score, interval
expansion and coverage/set metrics are owned by
``quant_fund.metrics.conformal`` (Vovk/Lei split-conformal machinery already
in the tree) and are imported, not duplicated. Coverage evaluation reuses
``metrics.conformal.covered`` / ``set_metrics``.

Honesty: every interval here is a calibrated prediction set evaluated only
by proper interval scores (PICP, PINAW, CV_delta, Winkler, MMW) — no
Sharpe/Sortino/Calmar/P&L/NAV anywhere. All benchmarks are SYNTHETIC Monte
Carlo correctness checks (planted DGPs, labeled ``SYNTHETIC_`` in bench keys);
they are correctness material, never market evidence. The conformal coverage
guarantee is marginal and requires exchangeability; the paper's own Sec. 5
shows distribution shift can still break coverage, so coverage is always
reported next to the tolerance-adjusted acceptable level C_a. Research
diagnostics only; no live-trading claims.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import (
    conformal_quantile,
    covered,
    cqr_scores,
    expand_interval,
    set_metrics,
)

Array = NDArray[np.float64]

#: A nonconformity score ``s(y, yhat)`` vectorized over candidate targets:
#: maps a ``(K,)`` candidate vector and an ``(M,)`` draw vector to a ``(K,)``
#: score vector. Must return finite values on finite inputs.
ScoreFn = Callable[[Array, Array], Array]

#: A distribution-generating predictor (paper step i): maps an ``(n,)`` or
#: ``(n, d)`` feature array to an ``(n, M)`` matrix of predictive draws.
DrawPredictor = Callable[[Array], Array]

#: A factory binding score hyperparameters to a ``ScoreFn``.
ScoreFactory = Callable[..., ScoreFn]

IntervalStatus = Literal["ok", "single_root", "no_root"]

#: Built-in score names (paper Sec. 4.4).
SCORE_NAMES: tuple[str, ...] = (
    "residual",
    "zscore",
    "interval",
    "interval_scaled",
    "hdi",
    "hdi_scaled",
    "knn",
)

#: Paper Sec. 4.4.3 / experiments: k = 10 nearest neighbors.
KNN_DEFAULT_K = 10
#: Paper Sec. 4.5: zeta = 1.645 (one-sided 95% margin), kappa = 2 growth.
COVERAGE_ZETA = 1.645
MMW_KAPPA = 2.0


class NonInvertibleScoreError(ValueError):
    """Score map cannot be inverted: non-finite evaluations or no usable root."""


# ---------------------------------------------------------------------------
# Validation helpers (fail-closed)
# ---------------------------------------------------------------------------


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_mass(mass: float, name: str = "band mass") -> float:
    m = float(mass)
    if not np.isfinite(m) or not 0.0 < m < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return m


def _finite_1d(values: object, name: str, min_size: int = 1) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size < min_size:
        raise ValueError(f"{name} must have at least {min_size} finite entries")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def _check_draws_1d(draws: object, min_size: int = 2) -> Array:
    return _finite_1d(draws, "draws", min_size=min_size)


def _check_draws_2d(draws: object, name: str = "draws", min_size: int = 2) -> Array:
    a = np.asarray(draws, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < min_size:
        raise ValueError(f"{name} must be a non-empty (n, M) draw matrix with M >= {min_size}")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    return a


def _check_y(y: object, n: int) -> Array:
    v = _finite_1d(y, "y")
    if v.size != n:
        raise ValueError(f"y must have length {n} to match the draw matrix")
    return v


def predictive_median(draws: Array) -> float:
    """Median of the draw vector — the paper's root-search anchor."""
    return float(np.median(_check_draws_1d(draws)))


def predictive_mean(draws: Array) -> float:
    d = _check_draws_1d(draws)
    return float(np.mean(d))


def predictive_std(draws: Array) -> float:
    """Population std of the draws (Eq. 14 denominator). Fail-closed at 0."""
    d = _check_draws_1d(draws)
    sd = float(np.std(d))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("draw vector is degenerate (zero dispersion)")
    return sd


# ---------------------------------------------------------------------------
# Inner bands (paper Sec. 4.4.2): equal-tail quantiles or HDI of the draws
# ---------------------------------------------------------------------------

BandKind = Literal["quantile", "hdi"]


def inner_band_quantile(draws: Array, mass: float) -> tuple[float, float]:
    """Equal-tail inner band [q_{(1-m)/2}, q_{(1+m)/2}] of the draw vector."""
    m = _check_mass(mass)
    d = _check_draws_1d(draws)
    tail = (1.0 - m) / 2.0
    lo, hi = np.quantile(d, [tail, 1.0 - tail])
    return float(lo), float(hi)


def inner_band_hdi(draws: Array, mass: float) -> tuple[float, float]:
    """Shortest interval containing ``mass`` of the draws (highest-density).

    Standard empirical HDI on the sorted draw vector: the tightest window of
    ``ceil(mass * M)`` consecutive order statistics. On tie windows the
    lowest-positioned one is returned (deterministic argmin).
    """
    m = _check_mass(mass)
    d = np.sort(_check_draws_1d(draws))
    n = int(d.size)
    w = int(np.ceil(m * n))
    w = min(max(w, 1), n)
    widths = d[w - 1 :] - d[: n - w + 1]
    j = int(np.argmin(widths))
    return float(d[j]), float(d[j + w - 1])


def inner_band(draws: Array, mass: float, kind: BandKind = "quantile") -> tuple[float, float]:
    if kind == "quantile":
        return inner_band_quantile(draws, mass)
    if kind == "hdi":
        return inner_band_hdi(draws, mass)
    raise ValueError("band kind must be 'quantile' or 'hdi'")


# ---------------------------------------------------------------------------
# Nonconformity scores (paper Sec. 4.4) — all vectorized over candidate y
# ---------------------------------------------------------------------------


def score_residual() -> ScoreFn:
    """Absolute residual score s_R = |y - mu_hat| (paper Eq. 13)."""

    def _s(y: Array, draws: Array) -> Array:
        yy = np.asarray(y, dtype=float)
        mu = float(np.mean(draws))
        return np.asarray(np.abs(yy - mu), dtype=float)

    return _s


def score_zscore() -> ScoreFn:
    """Standardized residual s_z = |y - mu_hat| / sigma_hat (paper Eq. 14).

    The MC-CP-style variance-scaled score; sigma is recomputed per draw
    vector, is strictly positive (fail-closed on degenerate draws), and the
    conformal guarantee is preserved because the scaling depends only on the
    predictor.
    """

    def _s(y: Array, draws: Array) -> Array:
        yy = np.asarray(y, dtype=float)
        sd = predictive_std(draws)
        return np.asarray(np.abs(yy - float(np.mean(draws))) / sd, dtype=float)

    return _s


def score_interval(*, mass: float, kind: BandKind = "quantile", scaled: bool = False) -> ScoreFn:
    """Interval-violation score s_int = max{C_low - y, y - C_up} (Eq. 15).

    Negative inside the base band, zero on the boundary, positive outside —
    qhat may legitimately be negative, shrinking the interval relative to a
    conservative base band. ``kind="quantile"`` is exactly the CQR score
    (Romano et al. 2019) on the draw vector's quantile band; ``kind="hdi"``
    is exactly the CMC score (Mehdiyev et al. 2025) on the draws' HDI.
    ``scaled`` divides by the base-band width (Sec. 4.4.2 standardize-rescale
    pairing) so calibrated qhat re-multiplies by local width at inference.
    """
    m = _check_mass(mass)

    def _s(y: Array, draws: Array) -> Array:
        yy = np.asarray(y, dtype=float)
        lo, hi = inner_band(draws, m, kind)
        raw = np.maximum(lo - yy, yy - hi)
        if scaled:
            width = hi - lo
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError("base inner band is degenerate (zero width)")
            raw = raw / width
        return np.asarray(raw, dtype=float)

    return _s


def _median_knn_distance(y: Array, draws: Array, k: int) -> Array:
    """Median of the k nearest |y - yhat_j| distances, vectorized in y.

    The paper's Eq. 16 argmin over d' of sum_j |d' - d_(j)| is the median of
    the k smallest distances by the L1-median characterization.
    """
    dist = np.abs(np.asarray(y, dtype=float)[:, None] - np.asarray(draws, dtype=float)[None, :])
    nearest = np.sort(dist, axis=1)[:, :k]
    return np.asarray(np.median(nearest, axis=1), dtype=float)


def _median_pairwise_distance(draws: Array) -> float:
    """Median of all M^2 pairwise |yhat_j - yhat_j'| distances (Eq. 18 denom).

    The paper's sum over j, j' = 1..M includes the M zero diagonal terms;
    that is the faithful definition and is what is implemented here.
    """
    d = np.asarray(draws, dtype=float)
    pairwise = np.abs(d[:, None] - d[None, :])
    return float(np.median(pairwise))


def score_knn(*, k: int = KNN_DEFAULT_K) -> ScoreFn:
    """KNN density-surrogate score s_KNN = d_k / d_tilde (paper Eqs. 16-18).

    d_k is the median of the k nearest |y - yhat_j| distances; d_tilde is the
    median of all M^2 pairwise distances inside the draw vector — strictly
    positive, predictor-only, so conformal validity is preserved. Shape
    sensitive: nonmonotone under multimodal predictive distributions.
    """
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be >= 1")

    def _s(y: Array, draws: Array) -> Array:
        d = _check_draws_1d(draws)
        if kk > d.size:
            raise ValueError(f"k={kk} exceeds draw vector size {d.size}")
        denom = _median_pairwise_distance(d)
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError("draw vector is degenerate (zero median pairwise distance)")
        return np.asarray(_median_knn_distance(np.asarray(y, dtype=float), d, kk) / denom)

    return _s


# ---------------------------------------------------------------------------
# Score registry — pluggable user ScoreFns alongside the built-ins
# ---------------------------------------------------------------------------

_USER_SCORES: dict[str, ScoreFn] = {}


def register_score(name: str, fn: ScoreFn) -> None:
    """Register a pluggable nonconformity score under ``name``.

    The callable contract is ``fn(y_candidates (K,), draws (M,)) -> (K,)``
    returning finite values. Registered names may shadow built-ins
    deliberately (documented behavior — the registry is last-write-wins).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("score name must be a non-empty string")
    if not callable(fn):
        raise ValueError("score must be callable")
    _USER_SCORES[name.strip()] = fn


def get_score(
    name: str,
    *,
    alpha: float = 0.1,
    band_mass: float | None = None,
    knn_k: int = KNN_DEFAULT_K,
) -> ScoreFn:
    """Resolve a score by name.

    ``band_mass`` is the inner-band probability mass for the interval family
    and defaults to ``1 - alpha`` (the paper's 0.05/0.95 quantiles and 90%
    HDI at alpha = 0.1). User scores registered via ``register_score`` take
    precedence over built-ins of the same name.
    """
    if name in _USER_SCORES:
        return _USER_SCORES[name]
    a = _check_alpha(alpha)
    mass = _check_mass(band_mass) if band_mass is not None else 1.0 - a
    builtins: dict[str, ScoreFn] = {
        "residual": score_residual(),
        "zscore": score_zscore(),
        "interval": score_interval(mass=mass, kind="quantile", scaled=False),
        "interval_scaled": score_interval(mass=mass, kind="quantile", scaled=True),
        "hdi": score_interval(mass=mass, kind="hdi", scaled=False),
        "hdi_scaled": score_interval(mass=mass, kind="hdi", scaled=True),
        "knn": score_knn(k=knn_k),
    }
    try:
        return builtins[name]
    except KeyError:
        raise ValueError(
            f"unknown score {name!r}; built-ins {sorted(builtins)} "
            f"plus registered {sorted(_USER_SCORES)}"
        ) from None


def predict_draws(predictor: DrawPredictor, x: Array) -> Array:
    """Call a distribution-generating predictor and validate its draw matrix.

    The predictor must return a finite ``(n, M)`` matrix with ``n`` matching
    the number of feature rows of ``x`` and ``M >= 2`` (paper step i).
    """
    xx = np.asarray(x, dtype=float)
    n = xx.shape[0] if xx.ndim >= 1 else 1
    out = np.asarray(predictor(xx), dtype=float)
    if out.ndim != 2 or out.shape[0] != n or out.shape[1] < 2:
        raise ValueError(f"predictor must return an (n={n}, M>=2) draw matrix; got {out.shape}")
    if not bool(np.all(np.isfinite(out))):
        raise ValueError("predictor draws must be finite")
    return out


# ---------------------------------------------------------------------------
# Numerical inversion (paper Sec. 3 step iv, Sec. 3.1-3.2, Table 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootFinderConfig:
    """Geometric-grid + bisection root finder (paper Table 1 defaults).

    ``h0`` initial grid step, ``gamma`` geometric expansion, ``tol``
    bisection absolute tolerance, ``depth`` grid steps per side. On fewer
    than two sign changes the search retries with ``h0 * shrink_factor`` and
    ``depth * depth_growth`` up to ``max_retries`` times (paper Sec. 3.2
    automatic retry). ``strict`` upgrades the paper's degenerate-interval
    failure policy into a ``NonInvertibleScoreError``.
    """

    h0: float = 1e-6
    gamma: float = 1.167
    tol: float = 1e-10
    depth: int = 100
    max_retries: int = 2
    shrink_factor: float = 0.1
    depth_growth: float = 2.0
    strict: bool = False
    max_bisect_iter: int = 300

    def __post_init__(self) -> None:
        if not np.isfinite(self.h0) or self.h0 <= 0.0:
            raise ValueError("h0 must be finite and > 0")
        if not np.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and > 1")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and > 0")
        if int(self.depth) < 1:
            raise ValueError("depth must be >= 1")
        if int(self.max_retries) < 0:
            raise ValueError("max_retries must be >= 0")
        if not 0.0 < float(self.shrink_factor) < 1.0:
            raise ValueError("shrink_factor must be in (0, 1)")
        if not float(self.depth_growth) > 1.0:
            raise ValueError("depth_growth must be > 1")
        if int(self.max_bisect_iter) < 10:
            raise ValueError("max_bisect_iter must be >= 10")


@dataclass(frozen=True)
class IntervalResult:
    """One inverted prediction interval with its inversion status."""

    low: float
    high: float
    status: IntervalStatus
    n_roots: int


def _grid_points(anchor: float, h0: float, gamma: float, depth: int) -> Array:
    """Ordered symmetric geometric grid anchor + {0, +-h0 gamma^j} (paper iv)."""
    steps = h0 * np.power(gamma, np.arange(depth, dtype=float))
    offsets = np.concatenate(([0.0], steps, -steps))
    return np.asarray(np.sort(anchor + offsets), dtype=float)


def _bisect(
    fn: Callable[[Array], Array], a: float, b: float, fa: float, tol: float, max_iter: int
) -> float:
    """Bisection on a sign-changing bracket to absolute tolerance."""
    lo, hi = float(a), float(b)
    flo = float(fa)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = float(fn(np.array([mid]))[0])
        if not np.isfinite(fmid):
            raise NonInvertibleScoreError("score produced non-finite values during bisection")
        if hi - lo <= tol or fmid == 0.0:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _ordered_events(
    pts: Array, fvals: Array
) -> list[tuple[float, tuple[float, float] | None, float]]:
    """Root events on the ordered grid, sorted by position.

    Each event is ``(position, bracket, f_left)``: an exact zero gives
    ``bracket=None`` and the grid point as position; a sign change gives the
    ``(a, b)`` adjacent pair and ``f(a)`` for bisection.
    """
    events: list[tuple[float, tuple[float, float] | None, float]] = []
    sgn = np.sign(fvals)
    cross = np.nonzero(sgn[:-1] * sgn[1:] < 0.0)[0]
    for i in cross:
        a, b = float(pts[int(i)]), float(pts[int(i) + 1])
        events.append((a, (a, b), float(fvals[int(i)])))
    for j in np.nonzero(fvals == 0.0)[0]:
        p = float(pts[int(j)])
        events.append((p, None, 0.0))
    events.sort(key=lambda e: e[0])
    return events


def _resolve_event(
    fn: Callable[[Array], Array],
    event: tuple[float, tuple[float, float] | None, float],
    cfg: RootFinderConfig,
) -> float:
    """Resolve one root event: exact zeros directly, brackets by bisection."""
    pos, bracket, fa = event
    if bracket is None:
        return float(pos)
    a, b = bracket
    return float(_bisect(fn, a, b, fa, cfg.tol, cfg.max_bisect_iter))


def invert_interval(
    score: ScoreFn,
    draws: Array,
    qhat: float,
    config: RootFinderConfig | None = None,
) -> IntervalResult:
    """Invert the prediction set C = {y : s(y, yhat) <= qhat} by root-finding.

    Paper step (iv): evaluate f(y) = s(y, yhat) - qhat on the ordered
    geometric grid anchored at the predictive median, take the OUTERMOST
    sign changes as brackets, and bisect each to ``config.tol``. Handles
    nonmonotone scores (more than two crossings -> outermost pair) and
    negative qhat. Failure policy (paper Sec. 3.2): retry with shrunk h0 and
    grown depth when fewer than two sign changes appear; a single persistent
    crossing gives a degenerate interval at that root; none gives a
    degenerate interval at the median. With ``config.strict`` those cases
    raise ``NonInvertibleScoreError`` instead; a score map that is non-finite
    anywhere on the search grid always raises.
    """
    cfg = config if config is not None else RootFinderConfig()
    d = _check_draws_1d(draws)
    q = float(qhat)
    if not np.isfinite(q):
        raise ValueError("qhat must be finite")

    def _f(y: Array) -> Array:
        out = np.asarray(score(np.asarray(y, dtype=float), d), dtype=float)
        return np.asarray(out - q, dtype=float)

    anchor = float(np.median(d))
    h0, depth = cfg.h0, int(cfg.depth)
    events: list[tuple[float, tuple[float, float] | None, float]] = []

    for _attempt in range(cfg.max_retries + 1):
        pts = _grid_points(anchor, h0, cfg.gamma, depth)
        fvals = _f(pts)
        if not bool(np.all(np.isfinite(fvals))):
            raise NonInvertibleScoreError(
                "score map produced non-finite values on the search grid; non-invertible"
            )
        events = _ordered_events(pts, fvals)
        if len(events) >= 2:
            # Outermost sign changes (paper: leftmost and rightmost pairs).
            low = _resolve_event(_f, events[0], cfg)
            high = _resolve_event(_f, events[-1], cfg)
            return IntervalResult(
                low=float(min(low, high)),
                high=float(max(low, high)),
                status="ok",
                n_roots=len(events),
            )
        h0 *= cfg.shrink_factor
        depth = max(depth + 1, int(math.ceil(depth * cfg.depth_growth)))

    if cfg.strict:
        raise NonInvertibleScoreError(
            f"score map non-invertible after {cfg.max_retries + 1} attempts: "
            f"{len(events)} sign change(s) found"
        )
    if len(events) == 1:
        root = _resolve_event(_f, events[0], cfg)
        return IntervalResult(low=root, high=root, status="single_root", n_roots=1)
    return IntervalResult(low=anchor, high=anchor, status="no_root", n_roots=0)


# ---------------------------------------------------------------------------
# Calibration + prediction (paper Alg. 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DCPFit:
    """Calibrated DCP model: score + global conformal threshold qhat."""

    score_name: str
    score_fn: ScoreFn
    alpha: float
    qhat: float
    n_cal: int
    cal_scores: Array


def _resolve_score(score: str | ScoreFn, alpha: float) -> tuple[str, ScoreFn]:
    if callable(score):
        return getattr(score, "__name__", "custom"), score
    return score, get_score(score, alpha=alpha)


def calibration_scores(y_cal: Array, draws_cal: Array, score_fn: ScoreFn) -> Array:
    """Nonconformity scores eps_i = s(y_i, yhat_i^c) (paper Eq. 5)."""
    dc = _check_draws_2d(draws_cal, name="draws_cal")
    yc = _check_y(y_cal, dc.shape[0])
    out = np.empty(dc.shape[0], dtype=float)
    for i in range(dc.shape[0]):
        out[i] = float(score_fn(np.array([yc[i]]), dc[i])[0])
    if not bool(np.all(np.isfinite(out))):
        raise ValueError("nonconformity scores on the calibration set must be finite")
    return out


def dcp_fit(
    y_cal: Array,
    draws_cal: Array,
    *,
    score: str | ScoreFn = "interval",
    alpha: float = 0.1,
) -> DCPFit:
    """Calibration phase (paper Alg. 1 lines 5-9).

    Scores the calibration set and takes qhat as the finite-sample
    split-conformal quantile — the ceil((n+1)(1-alpha))-th smallest score,
    computed by ``metrics.conformal.conformal_quantile`` (composed, not
    reimplemented).
    """
    a = _check_alpha(alpha)
    name, fn = _resolve_score(score, a)
    eps = calibration_scores(y_cal, draws_cal, fn)
    qhat = conformal_quantile(eps, a)
    return DCPFit(
        score_name=name,
        score_fn=fn,
        alpha=a,
        qhat=float(qhat),
        n_cal=int(eps.size),
        cal_scores=eps,
    )


@dataclass(frozen=True)
class DCPIntervals:
    """Batched prediction intervals with per-point inversion status."""

    low: Array
    high: Array
    status: tuple[IntervalStatus, ...]
    qhat: float

    @property
    def width(self) -> Array:
        return np.asarray(self.high - self.low, dtype=float)

    @property
    def n_degenerate(self) -> int:
        return int(sum(1 for s in self.status if s != "ok"))


def dcp_predict(
    fit: DCPFit,
    draws_test: Array,
    config: RootFinderConfig | None = None,
) -> DCPIntervals:
    """Inference phase (paper Alg. 1 Predict): invert each test draw vector."""
    dt = _check_draws_2d(draws_test, name="draws_test")
    lo = np.empty(dt.shape[0], dtype=float)
    hi = np.empty(dt.shape[0], dtype=float)
    status: list[IntervalStatus] = []
    for i in range(dt.shape[0]):
        res = invert_interval(fit.score_fn, dt[i], fit.qhat, config)
        lo[i], hi[i] = res.low, res.high
        status.append(res.status)
    return DCPIntervals(low=lo, high=hi, status=tuple(status), qhat=fit.qhat)


def dcp_intervals(
    y_cal: Array,
    draws_cal: Array,
    draws_test: Array,
    *,
    score: str | ScoreFn = "interval",
    alpha: float = 0.1,
    config: RootFinderConfig | None = None,
) -> DCPIntervals:
    """One-shot DCP: fit on the calibration split, invert on the test split."""
    return dcp_predict(dcp_fit(y_cal, draws_cal, score=score, alpha=alpha), draws_test, config)


def dcp_online_intervals(
    y_cal: Array,
    draws_cal: Array,
    y_test: Array,
    draws_test: Array,
    *,
    score: str | ScoreFn = "interval",
    alpha: float = 0.1,
    config: RootFinderConfig | None = None,
) -> DCPIntervals:
    """Sliding-window online split conformal (paper Sec. 4 adaptive variant).

    The fixed-length window is seeded with the calibration scores; after each
    revealed test target its score is appended and the oldest element dropped,
    so qhat is re-estimated before every prediction and tracks gradual drift.
    Because qhat is per-point online, ``DCPIntervals.qhat`` reports NaN here.
    """
    a = _check_alpha(alpha)
    _name, fn = _resolve_score(score, a)
    window = list(calibration_scores(y_cal, draws_cal, fn))
    n_win = len(window)
    dt = _check_draws_2d(draws_test, name="draws_test")
    yt = _check_y(y_test, dt.shape[0])
    lo = np.empty(dt.shape[0], dtype=float)
    hi = np.empty(dt.shape[0], dtype=float)
    status: list[IntervalStatus] = []
    for i in range(dt.shape[0]):
        qhat = conformal_quantile(np.asarray(window, dtype=float), a)
        res = invert_interval(fn, dt[i], qhat, config)
        lo[i], hi[i] = res.low, res.high
        status.append(res.status)
        window.append(float(fn(np.array([yt[i]]), dt[i])[0]))
        del window[: len(window) - n_win]
    return DCPIntervals(low=lo, high=hi, status=tuple(status), qhat=float("nan"))


# ---------------------------------------------------------------------------
# Baselines composed from metrics.conformal (never reimplemented)
# ---------------------------------------------------------------------------


def split_conformal_interval(
    y_cal: Array,
    draws_cal: Array,
    draws_test: Array,
    *,
    alpha: float = 0.1,
) -> tuple[Array, Array]:
    """Classical split-conformal baseline: mu_hat +/- qhat (paper Eq. 9).

    Residual scores on the draw means, conformal quantile, symmetric
    expansion — entirely via ``metrics.conformal`` primitives.
    """
    a = _check_alpha(alpha)
    dc = _check_draws_2d(draws_cal, name="draws_cal")
    yc = _check_y(y_cal, dc.shape[0])
    dt = _check_draws_2d(draws_test, name="draws_test")
    mu_c = np.mean(dc, axis=1)
    eps = np.abs(yc - mu_c)
    qhat = conformal_quantile(eps, a)
    mu_t = np.mean(dt, axis=1)
    return expand_interval(mu_t, mu_t, qhat)


def cqr_interval(
    y_cal: Array,
    draws_cal: Array,
    draws_test: Array,
    *,
    alpha: float = 0.1,
    kind: BandKind = "quantile",
) -> tuple[Array, Array]:
    """CQR-style baseline: inner band +/- qhat (paper Eq. 10).

    Violation scores via ``metrics.conformal.cqr_scores`` (the same closed
    form as the ``"interval"``/``"hdi"`` DCP scores) and expansion via
    ``expand_interval`` — the analytic inverse DCP must reproduce up to
    bisection tolerance.
    """
    a = _check_alpha(alpha)
    dc = _check_draws_2d(draws_cal, name="draws_cal")
    yc = _check_y(y_cal, dc.shape[0])
    dt = _check_draws_2d(draws_test, name="draws_test")
    mass = 1.0 - a
    lo_c = np.array([inner_band(dc[i], mass, kind)[0] for i in range(dc.shape[0])])
    hi_c = np.array([inner_band(dc[i], mass, kind)[1] for i in range(dc.shape[0])])
    eps = cqr_scores(yc, lo_c, hi_c)
    qhat = conformal_quantile(eps, a)
    lo_t = np.array([inner_band(dt[i], mass, kind)[0] for i in range(dt.shape[0])])
    hi_t = np.array([inner_band(dt[i], mass, kind)[1] for i in range(dt.shape[0])])
    return expand_interval(lo_t, hi_t, qhat)


# ---------------------------------------------------------------------------
# Efficiency metrics (paper Sec. 4.5)
# ---------------------------------------------------------------------------


def picp(y: Array, low: Array, high: Array) -> float:
    """Empirical coverage / PICP (paper Eq. 19) via metrics.conformal."""
    return float(np.mean(covered(y, low, high)))


def acceptable_coverage(n_test: int, alpha: float, zeta: float = COVERAGE_ZETA) -> float:
    """Minimal acceptable coverage C_a (paper Eq. 20).

    One-sided 95% binomial margin below nominal: sampling variability is not
    a coverage failure. Converges to 1 - alpha as N_t grows.
    """
    a = _check_alpha(alpha)
    n = int(n_test)
    if n < 1:
        raise ValueError("n_test must be >= 1")
    z = float(zeta)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("zeta must be finite and >= 0")
    return float(1.0 - a - z * math.sqrt(a * (1.0 - a) / n))


def pinaw(y: Array, low: Array, high: Array) -> float:
    """Prediction Interval Normalized Average Width (paper Eqs. 21-23)."""
    yy = _finite_1d(y, "y")
    lo = _finite_1d(low, "low")
    hi = _finite_1d(high, "high")
    if lo.size != yy.size or hi.size != yy.size:
        raise ValueError("y, low, high must have identical lengths")
    xi = float(np.max(yy) - np.min(yy))
    if xi <= 0.0:
        raise ValueError("target range is zero; PINAW undefined")
    return float(np.mean(hi - lo) / xi)


def width_cv(low: Array, high: Array) -> float:
    """Coefficient of variation of interval width CV_delta (paper Eq. 24)."""
    lo = _finite_1d(low, "low")
    hi = _finite_1d(high, "high")
    if lo.size != hi.size:
        raise ValueError("low and high must have identical lengths")
    if lo.size < 2:
        raise ValueError("need at least 2 intervals for CV_delta")
    delta = hi - lo
    mu = float(np.mean(delta))
    if mu <= 0.0:
        raise ValueError("mean width must be > 0 for CV_delta")
    return float(np.std(delta, ddof=1) / mu)


def winkler_components(y: Array, low: Array, high: Array, alpha: float) -> tuple[Array, Array]:
    """Per-point width delta_i and miss penalty e_i (paper Eqs. 25-26)."""
    a = _check_alpha(alpha)
    yy = _finite_1d(y, "y")
    lo = _finite_1d(low, "low")
    hi = _finite_1d(high, "high")
    if lo.size != yy.size or hi.size != yy.size:
        raise ValueError("y, low, high must have identical lengths")
    delta = hi - lo
    miss_low = np.maximum(lo - yy, 0.0)
    miss_high = np.maximum(yy - hi, 0.0)
    e = (2.0 / a) * (miss_low + miss_high)
    return np.asarray(delta, dtype=float), np.asarray(e, dtype=float)


def winkler_score(y: Array, low: Array, high: Array, alpha: float) -> float:
    """Mean Winkler interval score (1/N) sum_i (delta_i + e_i) (Eq. 25)."""
    delta, e = winkler_components(y, low, high, alpha)
    return float(np.mean(delta + e))


def undercoverage_penalty_factor(
    empirical_coverage: float,
    n_test: int,
    alpha: float,
    *,
    kappa: float = MMW_KAPPA,
    zeta: float = COVERAGE_ZETA,
) -> float:
    """Undercoverage penalty P_uc = exp(kappa * rho) (paper Eq. 27).

    rho = DeltaC / (1 - C_hat) with DeltaC = max(0, C_a - C_hat). Equals 1.0
    (no amplification) whenever coverage meets the tolerance-adjusted level.
    """
    a = _check_alpha(alpha)
    c = float(empirical_coverage)
    if not np.isfinite(c) or not 0.0 <= c <= 1.0:
        raise ValueError("empirical_coverage must be in [0, 1]")
    k = float(kappa)
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("kappa must be finite and >= 0")
    c_a = acceptable_coverage(n_test, a, zeta)
    delta_c = max(0.0, c_a - c)
    if delta_c <= 0.0:
        return 1.0
    rho = delta_c / max(1.0 - c, 1e-12)
    return float(math.exp(k * rho))


@dataclass(frozen=True)
class ModifiedWinklerResult:
    """MMW plus its decomposition — penalty and coverage gap always surfaced."""

    mmw: float
    winkler: float
    penalty_factor: float
    coverage: float
    acceptable_coverage: float
    mean_width: float
    mean_miss: float


def modified_winkler_score(
    y: Array,
    low: Array,
    high: Array,
    alpha: float,
    *,
    kappa: float = MMW_KAPPA,
    zeta: float = COVERAGE_ZETA,
) -> ModifiedWinklerResult:
    """Modified mean Winkler score MMW = (1/N) sum_i (delta_i + P_uc e_i).

    Paper Eq. 28: amplifies the per-unit miss penalty when empirical coverage
    drops below the tolerance-adjusted level C_a, favoring wider intervals
    over larger miss distances when undercoverage is pronounced.
    """
    delta, e = winkler_components(y, low, high, alpha)
    cov = picp(y, low, high)
    p_uc = undercoverage_penalty_factor(
        cov, int(_finite_1d(y, "y").size), alpha, kappa=kappa, zeta=zeta
    )
    c_a = acceptable_coverage(int(_finite_1d(y, "y").size), alpha, zeta)
    return ModifiedWinklerResult(
        mmw=float(np.mean(delta + p_uc * e)),
        winkler=float(np.mean(delta + e)),
        penalty_factor=float(p_uc),
        coverage=float(cov),
        acceptable_coverage=float(c_a),
        mean_width=float(np.mean(delta)),
        mean_miss=float(np.mean(e)),
    )


def interval_metrics(y: Array, low: Array, high: Array, alpha: float) -> dict[str, float]:
    """Flat metric bundle for one interval set (validity/sharpness/efficiency)."""
    a = _check_alpha(alpha)
    res = modified_winkler_score(y, low, high, a)
    sm = set_metrics(y, low, high)
    delta = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
    return {
        "coverage": float(sm.coverage),
        "acceptable_coverage": float(res.acceptable_coverage),
        "coverage_ok": float(sm.coverage >= res.acceptable_coverage),
        "mean_width": float(sm.mean_width),
        "median_width": float(sm.median_width),
        "pinaw": float(pinaw(y, low, high)),
        "width_cv": float(width_cv(low, high))
        if float(np.min(delta)) < float(np.max(delta))
        else 0.0,
        "winkler": float(res.winkler),
        "mmw": float(res.mmw),
        "undercoverage_penalty": float(res.penalty_factor),
        "n": float(sm.n),
    }


# ---------------------------------------------------------------------------
# Predictor-score battery + seeded SYNTHETIC bench
# ---------------------------------------------------------------------------


def dcp_battery(
    y_cal: Array,
    draws_cal: Array,
    y_test: Array,
    draws_test: Array,
    *,
    scores: tuple[str, ...] = SCORE_NAMES,
    alpha: float = 0.1,
    config: RootFinderConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Run every requested score plus the baselines on one dataset.

    Rows keyed ``dcp_<score>`` plus ``split_conformal`` and ``cqr``; each row
    is the ``interval_metrics`` bundle. Degenerate inversions are counted in
    ``n_degenerate`` — surfaced, never hidden.
    """
    a = _check_alpha(alpha)
    out: dict[str, dict[str, float]] = {}
    for name in scores:
        fit = dcp_fit(y_cal, draws_cal, score=name, alpha=a)
        pred = dcp_predict(fit, draws_test, config)
        row = interval_metrics(y_test, pred.low, pred.high, a)
        row["n_degenerate"] = float(pred.n_degenerate)
        out[f"dcp_{name}"] = row
    lo_s, hi_s = split_conformal_interval(y_cal, draws_cal, draws_test, alpha=a)
    out["split_conformal"] = interval_metrics(y_test, lo_s, hi_s, a)
    lo_c, hi_c = cqr_interval(y_cal, draws_cal, draws_test, alpha=a)
    out["cqr"] = interval_metrics(y_test, lo_c, hi_c, a)
    return out


def bench_dcp(
    *,
    seed: int = 0,
    n_cal: int = 400,
    n_test: int = 250,
    n_draws: int = 100,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Seeded SYNTHETIC DCP bench on a planted heteroscedastic DGP.

    Planted world: x ~ U(-3, 3), mu(x) = sin(2x), sigma(x) = 0.08 + 0.45x^2/9,
    y = mu(x) + sigma(x) eps, eps ~ N(0, 1). The predictor is a well-specified
    ensemble oracle drawing N(mu(x), sigma(x)^2) — correctness material only,
    never market evidence (AGENTS.md honesty contract #2). Returns a flat
    ``dict[str, float]`` whose keys are all prefixed ``SYNTHETIC_``.
    """
    a = _check_alpha(alpha)
    rng = np.random.default_rng(int(seed))
    nc, nt, m = int(n_cal), int(n_test), int(n_draws)
    if nc < 20 or nt < 20 or m < 5:
        raise ValueError("bench needs n_cal>=20, n_test>=20, n_draws>=5")

    def _mu(x: Array) -> Array:
        return np.sin(2.0 * x)

    def _sigma(x: Array) -> Array:
        return 0.08 + 0.45 * x * x / 9.0

    def _make(n: int) -> tuple[Array, Array, Array]:
        x = rng.uniform(-3.0, 3.0, size=n)
        sig = _sigma(x)
        y = _mu(x) + sig * rng.standard_normal(n)
        draws = _mu(x)[:, None] + sig[:, None] * rng.standard_normal((n, m))
        return x, y, draws

    _xc, y_cal, draws_cal = _make(nc)
    _xt, y_test, draws_test = _make(nt)

    battery = dcp_battery(
        y_cal,
        draws_cal,
        y_test,
        draws_test,
        scores=("residual", "zscore", "interval", "knn"),
        alpha=a,
    )
    out: dict[str, float] = {
        "SYNTHETIC_seed": float(int(seed)),
        "SYNTHETIC_alpha": float(a),
        "SYNTHETIC_n_cal": float(nc),
        "SYNTHETIC_n_test": float(nt),
    }
    for method, row in battery.items():
        for key in (
            "coverage",
            "acceptable_coverage",
            "coverage_ok",
            "mean_width",
            "pinaw",
            "width_cv",
            "winkler",
            "mmw",
            "n_degenerate",
        ):
            if key in row:
                out[f"SYNTHETIC_{method}_{key}"] = float(row[key])
    return out


__all__ = [
    "COVERAGE_ZETA",
    "KNN_DEFAULT_K",
    "MMW_KAPPA",
    "SCORE_NAMES",
    "Array",
    "DCPFit",
    "DCPIntervals",
    "DrawPredictor",
    "IntervalResult",
    "IntervalStatus",
    "ModifiedWinklerResult",
    "NonInvertibleScoreError",
    "RootFinderConfig",
    "ScoreFn",
    "acceptable_coverage",
    "bench_dcp",
    "calibration_scores",
    "cqr_interval",
    "dcp_battery",
    "dcp_fit",
    "dcp_intervals",
    "dcp_online_intervals",
    "dcp_predict",
    "get_score",
    "inner_band",
    "inner_band_hdi",
    "inner_band_quantile",
    "interval_metrics",
    "invert_interval",
    "modified_winkler_score",
    "picp",
    "pinaw",
    "predict_draws",
    "predictive_mean",
    "predictive_median",
    "predictive_std",
    "register_score",
    "score_interval",
    "score_knn",
    "score_residual",
    "score_zscore",
    "split_conformal_interval",
    "undercoverage_penalty_factor",
    "width_cv",
    "winkler_components",
    "winkler_score",
]
