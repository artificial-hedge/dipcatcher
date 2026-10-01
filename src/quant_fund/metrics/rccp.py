"""RCCP: Retrieval-Corrected Conformal Prediction for time series.

Implements Retrieval-Corrected Conformal Prediction (RCCP) of Jin, Kim, Lee &
Lee (2026), "Retrieval-Corrected Conformal Prediction for Time Series",
CIKM '26, arXiv:2608.10553 [cs.LG]. Citation verified against
https://arxiv.org/abs/2608.10553 (title/authors/abstract match the lane spec;
the official reference code lives at https://github.com/jinsaaang/rccp).
Fetched 2026-09-30.

Setting (paper Sec. 3.1): a fixed forecaster f produces point predictions
Yhat_t = f(X_t); observations are split in time order into train, calibration
I_cal, and test I_test periods. Standard split conformal uses the global
absolute-residual quantile, which is poorly matched when the error
distribution is temporally dependent or drifting. RCCP separates LOCAL
residual evidence (retrieval) from the coverage guarantee (a scalar conformal
correction):

1. Retrieval knowledge base (Sec. 3.2). Each evaluated context is encoded as
   a retrieval key z_t = psi_f(X_t, Yhat_t) (any predictable features:
   input-window embeddings, forecaster states, prediction features, or
   concatenations). Once Y_i is observed the base stores the key with the
   realized ONE-SIDED residuals e+_i = (Y_i - Yhat_i)_+ and
   e-_i = (Yhat_i - Y_i)_+, kept separately so the upper and lower tails can
   differ. K_{t-1} = {(z_i, e+_i, e-_i) : i < t} is time-ordered, so an
   interval at time t only ever uses past residuals.

2. Retrieval (Sec. 3.2, eq. 3). N_K(t) = KNN_K(z_t, K_{t-1}, d) returns the K
   stored indices with smallest Euclidean distance to the query. Retrieved
   residuals may be uniform (hard top-K) or distance-weighted,
   w_i = exp(-d_i / tau) / sum_j exp(-d_j / tau), tau > 0.

3. Asymmetric retrieved interval (eq. 4-5). Retrieved one-sided quantiles
   R~+_t = Q^w_{1-alpha/2}{e+_i} and R~-_t = Q^w_{1-alpha/2}{e-_i} define
   C~_t = [Yhat_t - R~-_t, Yhat_t + R~+_t]. The symmetric ablation pools the
   absolute residuals into one Q^w_{1-alpha}{|e_i|} radius.

4. Normalized retrieval error and scalar correction (Sec. 3.3, eq. 6-7).
   On each calibration point, BEFORE Y_j is appended to the knowledge base,
       B_j = max( (Y_j - Yhat_j)_+ / R~+_j, (Yhat_j - Y_j)_+ / R~-_j ),
   a scalar nonconformity score for the asymmetric interval: B_t <= c iff
   Y_t in C_t(c) (paper Lemma 1). The correction is the finite-sample
   conformal quantile c_hat = Q_{1-alpha}({B_j : j in I_cal}) and the final
   interval is C^RCCP_t = [Yhat_t - c_hat R~-_t, Yhat_t + c_hat R~+_t].
   Direct retrieval is c_hat = 1 (the "w/o Corr." ablation); retrieval sets
   the local asymmetric SHAPE, the scalar correction sets the coverage SCALE.

5. Coverage-gap bound (Sec. 3.4, Theorem 1). Let F_cal, F_test be the CDFs of
   B on calibration and test, q = 1 - alpha, c* the target multiplier with
   F_cal(c*) = q, e_n = sup_c |Fhat_cal(c) - F_cal(c)| the empirical
   calibration error, and r_n = |q_n - q| the finite-sample quantile-level
   error. Under calibration-test score stability |F_test(c*) - F_cal(c*)| <=
   rho_n (Assumption 1), local identifiability slope m > 0 of F_cal near c*
   (Assumption 2), local Lipschitz constant L of F_test near c* (Assumption
   3), and e_n + r_n <= m delta0 / 2 (Assumption 4, delta0 the local radius):

       |P_test{Y in C(c_hat)} - q| <= rho_n + (2L/m)(e_n + r_n).

   ``coverage_gap_bound`` is a PLUG-IN DIAGNOSTIC of this bound: rho_n is
   replaced by the realized |Fhat_test(c_hat) - Fhat_cal(c_hat)|, e_n by a
   DKWM-form sqrt(log(2/dkwm_delta) / (2 n_cal)) deviation term (honest: a
   distributional bound, not a point estimate), r_n is exact, and m, L, delta0
   are local-slope/radius plug-in estimates of the empirical score CDFs
   unless supplied. When the slope plug-ins are non-positive or Assumption
   4 fails, the bound is reported vacuous rather than fabricated.

Paper reporting conventions (Sec. 4.1): empirical coverage Cov-hat =
mean 1{Y_t in [L_t, U_t]}, signed gap Delta Cov = 100 (Cov-hat - (1-alpha))
in percentage points ("severe undercoverage" when < -2), PI-Width =
mean(U_t - L_t), and the Winkler interval score (reused from
``metrics.calibration2.winkler_interval_score``). "Severe misses" are the
heavy tail of the miss-distance distribution (paper Fig. 4); we
operationalize the documented lane definition ``severe_miss_rate``: fraction
of test points whose distance to the nearest interval boundary exceeds
``factor`` times the interval's own width (default 1.0). The width
adaptivity ratio (paper Table 3) is Decile-10 / Decile-1 interval width.

Composition notes (reuse, never reimplement):

* ``metrics.conformal.conformal_quantile`` — the finite-sample order-statistic
  convention of Sec. 3.3 (c_hat and the SCP baseline share one
  implementation); ``set_metrics`` supplies the coverage/width diagnostics.
* ``metrics.calibration2.winkler_interval_score`` — the paper's Winkler
  score (eq. 12), not reimplemented.
* The lane spec names ``metrics/coverage_cs.py`` for coverage diagnostics;
  that module actually lives at ``quant_fund.research.coverage_cs``, which
  sits ABOVE metrics in ``configs/arch_boundaries.toml`` and cannot be
  imported from this layer. Coverage diagnostics here are the
  ``metrics.conformal`` vocabulary instead (documented spec deviation).
* ``weighted_conformal_quantile`` mirrors the Tibshirani-weighted
  finite-sample convention of ``models.localized_conformal.localized_
  conformal_quantile`` — also an upward edge for metrics and hence
  reimplemented in 6 lines; the test suite pins exact agreement with it and
  its uniform-weight reduction to ``conformal_quantile``.

Deviations from the reference implementation (deliberate, documented): the
upstream repo's online machinery (FAISS memory stores, EWMA residual-scale
normalization, Hopfield/context encoders, per-side "separate" corrections) is
an implementation superset around paper Algorithm 1, not the algorithm
itself. This module implements Algorithm 1 + Theorem 1 faithfully in pure
numpy: exact deterministic k-NN (ties broken by time index), the eq. 3
softmax/uniform weighting, one-sided retrieved quantiles, the max-score
scalar correction, and the plug-in bound diagnostic.

Honesty: all outputs are coverage/width/Winkler calibration diagnostics —
no Sharpe/Sortino/Calmar/P&L content anywhere. Every synthetic DGP in the
bench helper and the test battery is labeled SYNTHETIC correctness material,
never market evidence. Degenerate inputs fail closed (ValueError): bad
alpha, non-finite or misaligned arrays, a retrieval base that never reaches
``min_memory`` evaluated residuals, or too few scored calibration errors for
a meaningful quantile. A retrieved radius of zero is NOT a crash: it is
floored at ``radius_floor`` (mirroring the reference ``+eps`` denominator) so
a zero-width retrieved interval yields an honestly huge B score — flagged in
``degenerate_radius_count``, never silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.calibration2 import winkler_interval_score
from quant_fund.metrics.conformal import conformal_quantile, set_metrics
from quant_fund.utils.series import as_named_1d, require_same_length

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "RCCPConfig",
    "RCCPResult",
    "RetrievalIndex",
    "bench_rccp",
    "build_lagged_context_keys",
    "coverage_gap_bound",
    "knn_indices",
    "miss_distances",
    "normalized_retrieval_error",
    "one_sided_residuals",
    "rccp_evaluate",
    "recency_weighted_intervals",
    "retrieval_weights",
    "severe_miss_rate",
    "split_conformal_intervals",
    "weighted_conformal_quantile",
    "weighted_quantile",
    "width_adaptivity_ratio",
]

#: Reference-implementation divisor convention (proposal + eps).
DEFAULT_RADIUS_FLOOR = 1e-12
#: Minimum scored calibration errors before a conformal quantile is reported.
DEFAULT_MIN_CAL_SCORES = 20
#: Delta for the DKWM-form empirical-CDF deviation term in the bound.
DEFAULT_DKWM_DELTA = 0.05


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _finite_1d(values: object, name: str) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def _as_2d(values: object, name: str) -> Array:
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    return a


@dataclass(frozen=True)
class RCCPConfig:
    """RCCP configuration (paper Algorithm 1 + Table 4 ablation knobs).

    ``k_neighbors`` is K of KNN_K. ``tau`` selects the retrieval weighting of
    eq. 3: ``None`` is the uniform hard top-K case; a finite ``tau > 0`` uses
    w_i proportional to exp(-d_i / tau). ``min_memory`` is the number of
    evaluated residuals the knowledge base must hold before retrieval-based
    scores are computed (paper: "at least K evaluated residuals"; defaults to
    ``k_neighbors``). ``asymmetric=False`` is the "w/o Asym." ablation (one
    pooled |e| radius at level 1 - alpha); ``corrected=False`` is "w/o Corr."
    (c_hat = 1, direct retrieval). ``radius_floor`` is the reference
    implementation's ``+eps`` divisor guard.
    """

    alpha: float = 0.10
    k_neighbors: int = 20
    tau: float | None = None
    min_memory: int | None = None
    asymmetric: bool = True
    corrected: bool = True
    radius_floor: float = DEFAULT_RADIUS_FLOOR
    min_calibration_scores: int = DEFAULT_MIN_CAL_SCORES

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        k = int(self.k_neighbors)
        if k < 1:
            raise ValueError("k_neighbors must be >= 1")
        object.__setattr__(self, "k_neighbors", k)
        if self.tau is not None:
            t = float(self.tau)
            if not np.isfinite(t) or t <= 0.0:
                raise ValueError("tau must be finite and > 0")
            object.__setattr__(self, "tau", t)
        mm = self.k_neighbors if self.min_memory is None else int(self.min_memory)
        if mm < 1:
            raise ValueError("min_memory must be >= 1")
        object.__setattr__(self, "min_memory", mm)
        rf = float(self.radius_floor)
        if not np.isfinite(rf) or rf <= 0.0:
            raise ValueError("radius_floor must be finite and > 0")
        object.__setattr__(self, "radius_floor", rf)
        mcs = int(self.min_calibration_scores)
        if mcs < 1:
            raise ValueError("min_calibration_scores must be >= 1")
        object.__setattr__(self, "min_calibration_scores", mcs)


def one_sided_residuals(y: Array, y_hat: Array) -> tuple[Array, Array]:
    """e+ = (y - y_hat)_+ and e- = (y_hat - y)_+ (paper Sec. 3.2)."""
    yy = as_named_1d("y", np.asarray(y, dtype=float))
    yh = as_named_1d("y_hat", np.asarray(y_hat, dtype=float))
    require_same_length(("y", yy), ("y_hat", yh))
    resid = yy - yh
    if not bool(np.all(np.isfinite(resid))):
        raise ValueError("residuals must be finite")
    return np.maximum(resid, 0.0), np.maximum(-resid, 0.0)


def knn_indices(query: Array, keys: Array, k: int) -> tuple[IntArray, Array]:
    """Exact Euclidean k-NN; deterministic tie-break by stored (time) index.

    ``query`` is one (d,) key and ``keys`` the (n, d) knowledge-base matrix.
    Returns ``(indices, distances)`` in ascending distance order. Ties are
    resolved to the lower stored index — the older evaluated residual — which
    is the deterministic counterpart of the paper's time-ordered base.
    """
    q = np.asarray(query, dtype=float).reshape(-1)
    kb = _as_2d(keys, "keys")
    if q.size != kb.shape[1]:
        raise ValueError("query key dimension must match the knowledge base")
    kk = int(k)
    if kk < 1 or kk > kb.shape[0]:
        raise ValueError("k must be in [1, n_stored]")
    dist = np.linalg.norm(kb - q[None, :], axis=1)
    order = np.lexsort((np.arange(kb.shape[0]), dist))
    take = order[:kk]
    return np.asarray(take, dtype=np.int64), np.asarray(dist[take], dtype=float)


class RetrievalIndex:
    """Time-ordered retrieval knowledge base K_t = {(z_i, e+_i, e-_i)}.

    Pure numpy exact k-NN (the spec's deterministic index; FAISS in the
    reference code is an implementation detail, not the algorithm). Appends
    preserve evaluation order; ``query`` returns the K nearest stored rows.
    """

    def __init__(self, key_dim: int) -> None:
        d = int(key_dim)
        if d < 1:
            raise ValueError("key_dim must be >= 1")
        self._dim = d
        self._keys = np.empty((0, d), dtype=float)
        self._e_pos = np.empty((0,), dtype=float)
        self._e_neg = np.empty((0,), dtype=float)

    @property
    def key_dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return int(self._keys.shape[0])

    @property
    def keys(self) -> Array:
        return self._keys.copy()

    def append(self, key: Array, e_pos: float, e_neg: float) -> None:
        k = np.asarray(key, dtype=float).reshape(-1)
        if k.size != self._dim or not bool(np.all(np.isfinite(k))):
            raise ValueError("key must be finite with the index dimension")
        ep, en = float(e_pos), float(e_neg)
        if not np.isfinite(ep) or not np.isfinite(en) or ep < 0.0 or en < 0.0:
            raise ValueError("one-sided residuals must be finite and >= 0")
        self._keys = np.vstack([self._keys, k[None, :]])
        self._e_pos = np.concatenate([self._e_pos, [ep]])
        self._e_neg = np.concatenate([self._e_neg, [en]])

    def query(self, key: Array, k: int) -> tuple[IntArray, Array]:
        """(indices, distances) of the K nearest stored keys."""
        if self.size == 0:
            raise ValueError("retrieval knowledge base is empty")
        return knn_indices(key, self._keys, k)

    def residuals(self, indices: IntArray) -> tuple[Array, Array]:
        """One-sided residuals of stored rows ``indices`` (e+, e-)."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= self.size:
            raise ValueError("indices out of range for the knowledge base")
        return self._e_pos[idx], self._e_neg[idx]


def retrieval_weights(distances: Array, tau: float | None) -> Array:
    """Retrieval weights of paper eq. 3.

    ``tau=None`` is the uniform hard top-K case. ``tau > 0`` gives
    w_i = exp(-d_i / tau) / sum_j exp(-d_j / tau), computed in max-shifted
    log space so large distances cannot underflow the whole numerator.
    """
    d = _finite_1d(distances, "distances")
    if bool(np.any(d < 0.0)):
        raise ValueError("distances must be >= 0")
    if tau is None:
        return np.full(d.size, 1.0 / d.size)
    t = float(tau)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("tau must be finite and > 0")
    u = -d / t
    u -= float(u.max())
    w = np.exp(u)
    return np.asarray(w / float(w.sum()), dtype=float)


def weighted_quantile(values: Array, weights: Array, level: float) -> float:
    """Smallest v with cumulative weight >= ``level`` of total (paper Q^w).

    The PLAIN weighted empirical quantile of eq. 4 — the finite-sample
    correction lives in the scalar correction step, not here. ``level`` in
    (0, 1]; weights non-negative with positive total mass.
    """
    s = _finite_1d(values, "values")
    w = _finite_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("values and weights must have the same length")
    if bool(np.any(w < 0.0)):
        raise ValueError("weights must be non-negative")
    lv = float(level)
    if not np.isfinite(lv) or not 0.0 < lv <= 1.0:
        raise ValueError("level must be in (0, 1]")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must carry positive total mass")
    order = np.argsort(s, kind="mergesort")
    cum = np.cumsum(w[order])
    idx = int(np.searchsorted(cum, lv * total, side="left"))
    if idx >= s.size:
        return float(s[order][-1])
    return float(s[order][idx])


def weighted_conformal_quantile(scores: Array, weights: Array, alpha: float) -> float:
    """Tibshirani-weighted finite-sample conformal quantile (comparator only).

    Same convention as ``models.localized_conformal.localized_conformal_
    quantile`` (reimplemented: models sits above metrics in
    ``configs/arch_boundaries.toml``): weights normalized to sum n, then the
    smallest score whose cumulative weight reaches ``(1 - alpha) * (n + 1)``-
    style mass — equivalently an implicit +inf atom of mean weight. Uniform
    weights reduce EXACTLY to ``conformal_quantile`` (test-pinned).
    """
    s = _finite_1d(scores, "scores")
    w = _finite_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    a = _validate_alpha(alpha)
    if bool(np.any(w < 0.0)):
        raise ValueError("weights must be non-negative")
    n = int(s.size)
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0.0:
        raise ValueError("weights must carry positive total mass")
    w = w * (n / wsum)  # weights now sum to n; the +inf atom carries weight 1
    order = np.argsort(s, kind="mergesort")
    cum = np.cumsum(w[order]) / (n + 1.0)
    idx = int(np.searchsorted(cum, 1.0 - a, side="left"))
    if idx >= n:
        return float(s[order][-1])
    return float(s[order][idx])


def normalized_retrieval_error(
    y: float,
    y_hat: float,
    r_pos: float,
    r_neg: float,
    radius_floor: float = DEFAULT_RADIUS_FLOOR,
) -> float:
    """B = max((y - y_hat)_+ / R~+, (y_hat - y)_+ / R~-) (paper eq. 6).

    The retrieved radii are floored at ``radius_floor`` — the reference
    implementation's ``proposal + eps`` denominator. A zero retrieved radius
    with a realized residual on that side therefore yields a huge-but-finite
    score (an honest undercoverage signal), never an exception or a NaN.
    """
    yy, yh = float(y), float(y_hat)
    rp, rn = float(r_pos), float(r_neg)
    fl = float(radius_floor)
    if not np.isfinite(yy) or not np.isfinite(yh):
        raise ValueError("y and y_hat must be finite")
    if not np.isfinite(rp) or not np.isfinite(rn) or rp < 0.0 or rn < 0.0:
        raise ValueError("retrieved radii must be finite and >= 0")
    if not np.isfinite(fl) or fl <= 0.0:
        raise ValueError("radius_floor must be finite and > 0")
    e_pos = max(yy - yh, 0.0)
    e_neg = max(yh - yy, 0.0)
    return float(max(e_pos / max(rp, fl), e_neg / max(rn, fl)))


def _retrieved_radii(
    e_pos: Array, e_neg: Array, weights: Array, cfg: RCCPConfig
) -> tuple[float, float]:
    """Paper eq. 4: retrieved quantiles, asymmetric or symmetric ablation."""
    if cfg.asymmetric:
        level = 1.0 - cfg.alpha / 2.0
        return (
            weighted_quantile(e_pos, weights, level),
            weighted_quantile(e_neg, weights, level),
        )
    pooled = np.maximum(e_pos, e_neg)
    r = weighted_quantile(pooled, weights, 1.0 - cfg.alpha)
    return r, r


@dataclass(frozen=True)
class RCCPResult:
    """Output of ``rccp_evaluate`` (paper Algorithm 1).

    ``lo``/``hi`` are the per-test-point corrected intervals;
    ``r_pos``/``r_neg`` the retrieved (pre-correction) radii; ``b_cal`` the
    scored normalized retrieval errors on calibration; ``b_test`` the same
    score evaluated on test points (Lemma 1: covered iff B <= c_hat).
    ``degenerate_radius_count`` counts points where BOTH retrieved radii hit
    ``radius_floor`` (zero-width local evidence, surfaced not hidden).
    """

    lo: Array
    hi: Array
    r_pos: Array
    r_neg: Array
    c_hat: float
    b_cal: Array
    b_test: Array
    n_neighbors: IntArray
    n_warmup: int
    n_cal: int
    n_cal_scored: int
    degenerate_radius_count: int
    alpha: float

    def diagnostics(self, y_test: Array, severe_factor: float = 1.0) -> dict[str, float]:
        """Paper Sec. 4.1 metrics: coverage, gap, width, Winkler, severe misses."""
        yy = _finite_1d(y_test, "y_test")
        if yy.size != self.lo.size:
            raise ValueError("y_test must have one value per test row")
        sm = set_metrics(yy, self.lo, self.hi)
        win = winkler_interval_score(self.lo, self.hi, yy, self.alpha)
        out = {
            "coverage": sm.coverage,
            "target_coverage": 1.0 - self.alpha,
            "coverage_gap_pct": 100.0 * (sm.coverage - (1.0 - self.alpha)),
            "mean_width": sm.mean_width,
            "median_width": sm.median_width,
            "winkler": float(win["mean_score"][0]),
            "severe_miss_rate": severe_miss_rate(yy, self.lo, self.hi, factor=severe_factor),
            "miss_rate": 1.0 - sm.coverage,
            "c_hat": float(self.c_hat),
            "n_test": float(sm.n),
            "n_cal_scored": float(self.n_cal_scored),
            "degenerate_radius_count": float(self.degenerate_radius_count),
        }
        return out


def rccp_evaluate(
    y: Array,
    y_hat: Array,
    keys: Array,
    *,
    n_warmup: int,
    n_cal: int,
    config: RCCPConfig | None = None,
    observe_test: bool = True,
) -> RCCPResult:
    """Paper Algorithm 1 end to end: warmup -> calibration -> test.

    Indices ``[0, n_warmup)`` populate the retrieval knowledge base without
    scoring (the train-period evaluated residuals). Calibration indices
    ``[n_warmup, n_warmup + n_cal)`` are scored BEFORE being appended — a
    point's own residual is never in its retrieval pool. Test indices
    ``[n_warmup + n_cal, n)`` get the corrected interval, then are appended
    after ``observe_test`` (the paper's streaming update, Alg. 1 line 19);
    pass ``observe_test=False`` for a frozen-base evaluation.

    Fail-closed: the calibration loop must produce at least
    ``min_calibration_scores`` scored B_j (i.e. the base must reach
    ``min_memory`` evaluated residuals in time), else ValueError — a short or
    degenerate calibration never returns silently.
    """
    cfg = config if config is not None else RCCPConfig()
    yy = as_named_1d("y", np.asarray(y, dtype=float))
    yh = as_named_1d("y_hat", np.asarray(y_hat, dtype=float))
    kk = _as_2d(keys, "keys")
    n = yy.shape[0]
    require_same_length(("y", yy), ("y_hat", yh), ("keys", kk))
    if not bool(np.all(np.isfinite(yy))) or not bool(np.all(np.isfinite(yh))):
        raise ValueError("y and y_hat must be finite")
    n_warm = int(n_warmup)
    n_c = int(n_cal)
    if n_warm < 0 or n_c < 1:
        raise ValueError("n_warmup must be >= 0 and n_cal >= 1")
    if n_warm + n_c >= n:
        raise ValueError("need at least one test index after warmup + calibration")
    e_pos, e_neg = one_sided_residuals(yy, yh)

    kb = RetrievalIndex(kk.shape[1])
    min_mem = int(cfg.min_memory if cfg.min_memory is not None else cfg.k_neighbors)
    floor = float(cfg.radius_floor)
    degenerate = 0

    # Phase 1 — warmup + calibration (Alg. 1 lines 4-11): score j BEFORE
    # appending its residual to K (causality), then append.
    b_cal: list[float] = []
    for t in range(n_warm + n_c):
        if t >= n_warm and kb.size >= min_mem:
            sel, dist = kb.query(kk[t], min(cfg.k_neighbors, kb.size))
            ep, en = kb.residuals(sel)
            rp, rn = _retrieved_radii(ep, en, retrieval_weights(dist, cfg.tau), cfg)
            if rp <= floor and rn <= floor:
                degenerate += 1
            b_cal.append(normalized_retrieval_error(yy[t], yh[t], rp, rn, floor))
        kb.append(kk[t], e_pos[t], e_neg[t])

    # Phase 2 — the scalar correction (Alg. 1 line 12).
    if not cfg.corrected:
        c_hat = 1.0
    else:
        if len(b_cal) < cfg.min_calibration_scores:
            raise ValueError(
                f"too few scored calibration errors ({len(b_cal)} < "
                f"{cfg.min_calibration_scores}): the knowledge base reached "
                f"{kb.size} evaluated residuals by end of calibration; "
                "increase n_warmup/n_cal or lower min_memory"
            )
        c_hat = conformal_quantile(np.asarray(b_cal, dtype=float), cfg.alpha)

    # Phase 3 — test loop (Alg. 1 lines 13-19): retrieve, correct, append.
    n_test = n - n_warm - n_c
    lo = np.empty(n_test, dtype=float)
    hi = np.empty(n_test, dtype=float)
    rp_out = np.empty(n_test, dtype=float)
    rn_out = np.empty(n_test, dtype=float)
    nn_out = np.empty(n_test, dtype=np.int64)
    b_test = np.empty(n_test, dtype=float)
    for j, t in enumerate(range(n_warm + n_c, n)):
        if kb.size < min_mem:
            raise ValueError(
                "knowledge base has fewer than min_memory evaluated residuals at test time"
            )
        sel, dist = kb.query(kk[t], min(cfg.k_neighbors, kb.size))
        ep, en = kb.residuals(sel)
        rp, rn = _retrieved_radii(ep, en, retrieval_weights(dist, cfg.tau), cfg)
        if rp <= floor and rn <= floor:
            degenerate += 1
        rp_out[j], rn_out[j], nn_out[j] = rp, rn, int(sel.size)
        lo[j] = yh[t] - c_hat * rn
        hi[j] = yh[t] + c_hat * rp
        b_test[j] = normalized_retrieval_error(yy[t], yh[t], rp, rn, floor)
        if observe_test:
            kb.append(kk[t], e_pos[t], e_neg[t])

    return RCCPResult(
        lo=lo,
        hi=hi,
        r_pos=rp_out,
        r_neg=rn_out,
        c_hat=float(c_hat),
        b_cal=np.asarray(b_cal, dtype=float),
        b_test=b_test,
        n_neighbors=nn_out,
        n_warmup=n_warm,
        n_cal=n_c,
        n_cal_scored=len(b_cal),
        degenerate_radius_count=degenerate,
        alpha=float(cfg.alpha),
    )


def build_lagged_context_keys(series: Array, window: int) -> Array:
    """Lagged-window context keys: row j = series[j : j + window].

    Returns an ``(T - window + 1, window)`` matrix whose row ``j`` is the
    lookback window ENDING at index ``j + window - 1`` — i.e. it is a key for
    prediction index ``j + window`` (causal: only past observations). The
    caller aligns ``y``/``y_hat`` as ``y[window:]`` for indices
    ``window .. T-1`` when using this for retrieval on the target itself; any
    other deterministic context embedding (forecaster states, exogenous
    features, concatenations — the paper's psi_f) can be supplied directly to
    ``rccp_evaluate`` instead.
    """
    s = _finite_1d(series, "series")
    w = int(window)
    if w < 1 or w >= s.size:
        raise ValueError("window must be in [1, len(series))")
    out = np.lib.stride_tricks.sliding_window_view(s, w)
    return np.asarray(out, dtype=float)


def split_conformal_intervals(
    y_cal: Array, y_hat_cal: Array, y_hat_test: Array, alpha: float
) -> tuple[Array, Array]:
    """SCP baseline (paper eq. 2): symmetric global-residual intervals."""
    a = _validate_alpha(alpha)
    yc = _finite_1d(y_cal, "y_cal")
    yh_c = _finite_1d(y_hat_cal, "y_hat_cal")
    require_same_length(("y_cal", yc), ("y_hat_cal", yh_c))
    yh_t = _finite_1d(y_hat_test, "y_hat_test")
    scores = np.abs(yc - yh_c)
    q = conformal_quantile(scores, a)
    return yh_t - q, yh_t + q


def recency_weighted_intervals(
    y_cal: Array,
    y_hat_cal: Array,
    y_hat_test: Array,
    alpha: float,
    half_life: float,
) -> tuple[Array, Array]:
    """Eval comparator: recency-weighted split conformal (NexCP-flavored).

    Exponential time-decay weights w_i = 2^(-(n-1-i)/half_life) on the
    calibration absolute residuals through ``weighted_conformal_quantile``
    (Tibshirani-weighted finite-sample quantile). ``half_life -> inf``
    recovers ``split_conformal_intervals`` exactly. Comparator for the eval
    battery only — no coverage certificate is claimed for it.
    """
    a = _validate_alpha(alpha)
    hl = float(half_life)
    if not np.isfinite(hl) or hl <= 0.0:
        raise ValueError("half_life must be finite and > 0")
    yc = _finite_1d(y_cal, "y_cal")
    yh_c = _finite_1d(y_hat_cal, "y_hat_cal")
    require_same_length(("y_cal", yc), ("y_hat_cal", yh_c))
    yh_t = _finite_1d(y_hat_test, "y_hat_test")
    n = yc.size
    scores = np.abs(yc - yh_c)
    w = np.exp2(-(np.arange(n - 1, -1, -1, dtype=float) / hl))
    q = weighted_conformal_quantile(scores, w, a)
    return yh_t - q, yh_t + q


def miss_distances(y: Array, lo: Array, hi: Array) -> Array:
    """Distance of each realized y to the nearest interval boundary (0 if covered).

    The quantity whose CDF the paper's Fig. 4 plots — the violation size of a
    miscovered point.
    """
    yy = _finite_1d(y, "y")
    ll = _finite_1d(lo, "lo")
    uu = _finite_1d(hi, "hi")
    require_same_length(("y", yy), ("lo", ll), ("hi", uu))
    if bool(np.any(uu < ll)):
        raise ValueError("upper must exceed lower")
    return np.asarray(np.maximum(np.maximum(ll - yy, yy - uu), 0.0), dtype=float)


def severe_miss_rate(y: Array, lo: Array, hi: Array, factor: float = 1.0) -> float:
    """Fraction of test points whose miss distance exceeds ``factor`` x width.

    The lane's operationalization of the paper's "severe misses" (the heavy
    tail of the Fig. 4 miss-distance CDF; the paper reports no hard
    threshold). ``factor=1.0`` counts violations deeper than one interval
    width.
    """
    f = float(factor)
    if not np.isfinite(f) or f < 0.0:
        raise ValueError("factor must be finite and >= 0")
    md = miss_distances(y, lo, hi)
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    return float(np.mean(md > f * width))


def _ecdf(x: Array, grid: Array) -> Array:
    """Empirical CDF of ``x`` at ``grid`` (right-continuous)."""
    xs = np.sort(_finite_1d(x, "x"))
    return np.searchsorted(xs, grid, side="right") / xs.size


def coverage_gap_bound(
    b_cal: Array,
    b_test: Array,
    alpha: float,
    *,
    m: float | None = None,
    lipschitz: float | None = None,
    delta0: float | None = None,
    e_n: float | None = None,
    dkwm_delta: float = DEFAULT_DKWM_DELTA,
) -> dict[str, float]:
    """Plug-in diagnostic of paper Theorem 1 (coverage-gap bound).

    |P_test{Y in C(c_hat)} - q| <= rho_n + (2L/m)(e_n + r_n), where
    Lemma 1 reduces coverage of C(c_hat) to P(B <= c_hat). Components:

    * ``rho_n`` -> |Fhat_test(c_hat) - Fhat_cal(c_hat)|, the realized
      calibration-test mismatch at the deployed multiplier (Assumption 1
      plug-in).
    * ``r_n`` = |q_n - q| with q_n = ceil((n+1) q)/n — EXACT (the
      finite-sample level ``conformal_quantile`` actually uses).
    * ``e_n`` -> sqrt(log(2/dkwm_delta)/(2 n_cal)) (DKWM-form empirical-CDF
      deviation at level ``dkwm_delta``), unless supplied.
    * ``m``/``lipschitz`` plug-ins: secant slopes of the calibration/test
      empirical CDFs across +/-delta0 around c_hat (Assumptions 2/3).
      ``delta0`` defaults to ``std(b_cal)`` — a scale-aware local radius
      covering the effective support of the score distribution.
    * ``u_n`` = 2(e_n + r_n)/m must be < delta0 (Assumption 4) — else the
      bound is vacuous: reported ``bound = inf`` with ``vacuous = 1.0``
      (honest degradation, never clipped).
    * ``realized_gap`` = |Fhat_test(c_hat) - q| — what the bound must
      dominate; ``bound_holds`` = realized_gap <= bound + 1e-12.

    A diagnostic, not a certificate: the assumptions are on population CDFs
    while only empirical plug-ins are observable.
    """
    a = _validate_alpha(alpha)
    bc = _finite_1d(b_cal, "b_cal")
    bt = _finite_1d(b_test, "b_test")
    if bc.size < 5 or bt.size < 5:
        raise ValueError("b_cal and b_test need at least 5 scores each")
    q = 1.0 - a
    n_cal = int(bc.size)
    c_hat = conformal_quantile(bc, a)
    q_n = float(np.ceil((n_cal + 1) * q) / n_cal)
    q_n = min(q_n, 1.0)
    r_n = abs(q_n - q)
    en = float(e_n) if e_n is not None else float(np.sqrt(np.log(2.0 / dkwm_delta) / (2.0 * n_cal)))
    if not np.isfinite(en) or en < 0.0:
        raise ValueError("e_n must be finite and >= 0")

    f_cal_at_c = float(_ecdf(bc, np.array([c_hat]))[0])
    f_test_at_c = float(_ecdf(bt, np.array([c_hat]))[0])
    rho_hat = abs(f_test_at_c - f_cal_at_c)
    realized_gap = abs(f_test_at_c - q)

    d0 = float(delta0) if delta0 is not None else float(np.std(bc))
    if not np.isfinite(d0) or d0 < 0.0:
        raise ValueError("delta0 must be finite and >= 0")
    if d0 == 0.0:
        raise ValueError(
            "degenerate score distribution (zero spread): the local radius "
            "delta0 is 0 — pass an explicit delta0 or richer calibration scores"
        )

    def _slope(x: Array, center: float, radius: float) -> float:
        lo_v = float(_ecdf(x, np.array([center - radius]))[0])
        hi_v = float(_ecdf(x, np.array([center + radius]))[0])
        return (hi_v - lo_v) / (2.0 * radius)

    m_hat = float(m) if m is not None else _slope(bc, c_hat, d0)
    l_hat = float(lipschitz) if lipschitz is not None else _slope(bt, c_hat, d0)
    if m is not None and (not np.isfinite(m_hat) or m_hat <= 0.0):
        raise ValueError("m must be finite and > 0 (Assumption 2)")
    if lipschitz is not None and (not np.isfinite(l_hat) or l_hat < 0.0):
        raise ValueError("lipschitz must be finite and >= 0 (Assumption 3)")

    u_n = float("inf") if m_hat <= 0.0 else 2.0 * (en + r_n) / m_hat
    assumption4 = bool(u_n < d0)
    vacuous = m_hat <= 0.0 or not assumption4
    bound = float("inf") if vacuous else rho_hat + (2.0 * l_hat / m_hat) * (en + r_n)
    return {
        "c_hat": float(c_hat),
        "q": float(q),
        "q_n": float(q_n),
        "r_n": float(r_n),
        "e_n": float(en),
        "rho_n": float(rho_hat),
        "m": float(m_hat),
        "lipschitz": float(l_hat),
        "delta0": float(d0),
        "u_n": float(u_n),
        "assumption4": float(assumption4),
        "f_hat_test_at_c": f_test_at_c,
        "f_hat_cal_at_c": f_cal_at_c,
        "realized_gap": float(realized_gap),
        "bound": float(bound),
        "bound_holds": float(realized_gap <= bound + 1e-12),
        "vacuous": float(vacuous),
        "n_cal": float(n_cal),
        "n_test": float(bt.size),
    }


def width_adaptivity_ratio(lo: Array, hi: Array, error_magnitude: Array) -> float:
    """Paper Table 3: Decile-10 width / Decile-1 width, stratified by |error|.

    ``error_magnitude`` is a realized difficulty score per test point (the
    paper stratifies by realized |Y - Yhat| deciles); decile bins are the
    empirical deciles of the supplied magnitudes. Degenerate cases (too few
    points, zero-width deciles) raise rather than fabricate a ratio.
    """
    ll = _finite_1d(lo, "lo")
    uu = _finite_1d(hi, "hi")
    em = _finite_1d(error_magnitude, "error_magnitude")
    require_same_length(("lo", ll), ("hi", uu), ("error_magnitude", em))
    if em.size < 20:
        raise ValueError("need at least 20 test points for decile strata")
    if bool(np.any(uu < ll)):
        raise ValueError("upper must exceed lower")
    edges = np.quantile(em, np.linspace(0.0, 1.0, 11))
    width = uu - ll
    d1 = width[em <= edges[1]]
    d10 = width[em >= edges[9]]
    if d1.size == 0 or d10.size == 0 or float(d1.mean()) <= 0.0:
        raise ValueError("degenerate decile strata — adaptivity ratio undefined")
    return float(d10.mean() / d1.mean())


def _simulate_dgp(kind: str, n: int, rng: np.random.Generator) -> tuple[Array, Array, Array]:
    """SYNTHETIC planted DGPs (correctness material only — never market data).

    Returns ``(y, y_hat, keys)`` with ``keys[t]`` built from trailing residual
    magnitudes + the forecast value (a causal, deterministic retrieval
    embedding). ``kind``:

    * ``"iid"`` — homoskedastic AR(1) residuals (exchangeable; SCP is
      correctly specified, RCCP must not lose coverage).
    * ``"heteroskedastic"`` — two-volatility-regime Markov switching; the
      retrieval key identifies the regime (retrieval earns its keep).
    * ``"regime_shift"`` — a single mid-test scale break (temporal
      dependence; global quantiles lag, local retrieval tracks).
    """
    y = np.empty(n, dtype=float)
    yh = np.empty(n, dtype=float)
    phi = 0.55
    if kind == "iid":
        eps = rng.standard_normal(n)
    elif kind == "heteroskedastic":
        state = np.zeros(n, dtype=np.int64)
        for t in range(1, n):
            state[t] = state[t - 1] if rng.random() < 0.97 else 1 - state[t - 1]
        sigma = np.where(state == 0, 0.4, 1.8)
        eps = sigma * rng.standard_normal(n)
    elif kind == "regime_shift":
        eps = np.concatenate(
            [0.4 * rng.standard_normal(n // 2), 1.8 * rng.standard_normal(n - n // 2)]
        )
    else:
        raise ValueError("dgp must be one of iid / heteroskedastic / regime_shift")
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    yh[0] = 0.0
    yh[1:] = phi * y[:-1]  # perfect in-model forecaster; residuals = eps
    resid = y - yh
    # Retrieval keys: trailing |residual| window + forecast — causal context.
    w = 8
    keys = np.empty((n, w + 1), dtype=float)
    keys[:, w] = yh
    abs_resid = np.abs(resid)
    for t in range(n):
        lo_i = max(0, t - w)
        seg = abs_resid[lo_i:t]
        keys[t, : seg.size] = seg
        keys[t, seg.size : w] = seg[-1] if seg.size else 0.0
    return y, yh, keys


def bench_rccp(
    *,
    seed: int = 7,
    n_warmup: int = 300,
    n_cal: int = 400,
    n_test: int = 300,
    alpha: float = 0.10,
    k_neighbors: int = 20,
    tau: float | None = 0.5,
    dgp: str = "heteroskedastic",
) -> dict[str, float]:
    """Seeded SYNTHETIC bench (~seconds), flat ``dict[str, float]``.

    Runs RCCP plus the SCP and recency-weighted comparators on one planted
    DGP and reports the paper's metrics for each (all prefixed SYNTHETIC —
    correctness material, never market evidence).
    """
    rng = np.random.default_rng(int(seed))
    n = int(n_warmup) + int(n_cal) + int(n_test)
    y, yh, keys = _simulate_dgp(dgp, n, rng)
    cfg = RCCPConfig(alpha=alpha, k_neighbors=k_neighbors, tau=tau)
    res = rccp_evaluate(y, yh, keys, n_warmup=n_warmup, n_cal=n_cal, config=cfg)
    y_test = y[n_warmup + n_cal :]
    diag = res.diagnostics(y_test)
    diag["synthetic"] = 1.0  # labeled SYNTHETIC per the honesty contract

    cal = slice(n_warmup, n_warmup + n_cal)
    yh_test = yh[n_warmup + n_cal :]
    lo_s, hi_s = split_conformal_intervals(y[cal], yh[cal], yh_test, alpha)
    sm_s = set_metrics(y_test, lo_s, hi_s)
    win_s = winkler_interval_score(lo_s, hi_s, y_test, alpha)
    lo_w, hi_w = recency_weighted_intervals(
        y[cal], yh[cal], yh_test, alpha, half_life=max(n_cal // 4, 8)
    )
    sm_w = set_metrics(y_test, lo_w, hi_w)
    win_w = winkler_interval_score(lo_w, hi_w, y_test, alpha)

    # Retrieval-only ablation (c_hat = 1) and the bound diagnostic.
    res_nc = rccp_evaluate(
        y,
        yh,
        keys,
        n_warmup=n_warmup,
        n_cal=n_cal,
        config=RCCPConfig(alpha=alpha, k_neighbors=k_neighbors, tau=tau, corrected=False),
    )
    sm_nc = set_metrics(y_test, res_nc.lo, res_nc.hi)
    bound = coverage_gap_bound(res.b_cal, res.b_test, alpha)

    out = {
        "coverage": diag["coverage"],
        "target_coverage": 1.0 - alpha,
        "coverage_gap_pct": diag["coverage_gap_pct"],
        "mean_width": diag["mean_width"],
        "winkler": diag["winkler"],
        "severe_miss_rate": diag["severe_miss_rate"],
        "c_hat": diag["c_hat"],
        "scp_coverage": sm_s.coverage,
        "scp_mean_width": sm_s.mean_width,
        "scp_winkler": float(win_s["mean_score"][0]),
        "recency_coverage": sm_w.coverage,
        "recency_mean_width": sm_w.mean_width,
        "recency_winkler": float(win_w["mean_score"][0]),
        "uncorrected_coverage": sm_nc.coverage,
        "uncorrected_mean_width": sm_nc.mean_width,
        "bound": bound["bound"],
        "bound_holds": bound["bound_holds"],
        "bound_vacuous": bound["vacuous"],
        "realized_gap": bound["realized_gap"],
        "rho_n": bound["rho_n"],
        "n_test": float(n_test),
        "n_cal_scored": diag["n_cal_scored"],
        "degenerate_radius_count": diag["degenerate_radius_count"],
        "synthetic": 1.0,  # labeled SYNTHETIC per the honesty contract
    }
    return out
