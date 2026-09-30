"""Rolling conformal prediction (rolling-CP) for sequential model training.

Cheng, Liang & Barber (2026), "Rolling Conformal Prediction in Sequential
Model Training", arXiv:2609.26951 [math.ST], v1, 42 pp. Distribution-free
predictive inference for a data stream Z_i = (X_i, Y_i) in which the model at
time n may depend on the whole observed history Z_<n — one-pass training over
massive data, continual fine-tuning, or test-time adaptation:

1. Calibrate-then-roll. Each incoming Z_i is first scored by the *current*
   score function s_i(. ; Z_<i) — the model trained on Z_1..Z_{i-1} — and is
   then rolled into the training data for future score functions. No data
   splitting; every point is a fresh calibration point first and training
   data afterwards.
2. Prediction set (their Eq. (4)): with arrival scores S_i = s_i(Z_i; Z_<i),

       C_n = {z : sum_{i=1}^n 1{s_i(z; Z_<i) > S_i} < (1 - alpha)(n + 1)},

   equivalently, via the rolling conformal p-value (their Sec. 2.2),

       p_rolling = (1 + sum_{i=1}^n 1{S_i >= s_i(Z_{n+1}; Z_<i)}) / (n + 1),
       Z_{n+1} in C_n  <=>  p_rolling > alpha.

Guarantees (verified against the fetched paper text):

- Theorem 1 (their Eq. (5)): if Z_1..Z_{n+1} are exchangeable, then
  P(Z_{n+1} in C_n) >= 1 - 2*alpha for ANY sequence of score functions — no
  stability, consistency, or restriction on the training process. Remark 1
  sharpens the floor to 1 - (2*alpha - 1/(n+1))_+ (``marginal_coverage_floor``).
- Theorem 2: p_rolling is dominated in decreasing-convex order by
  Unif({1/(n+1),...,n/(n+1),1}); it is a p*-value in the sense of Wang (2024,
  "Testing with p*-values", Bernoulli 30(2), 1313-1346), hence
  P(p_rolling <= alpha) <= 2*alpha (their Eq. (6)).
- Theorem 3: for i.i.d. streams, with probability >= 1 - delta the
  training-conditional miscoverage is
  <= 2*alpha + sqrt(2*log(1/delta) / (alpha^2*(n+1))), and their Eq. (7)
  extends the bound uniformly over all n
  (``training_conditional_miscoverage_bound``).
- Assumption 1 (their Eq. (9), score-comparison stability) + Theorem 4
  (Eq. (10a)): if the score functions stabilize after m points with
  pairwise-comparison disagreement rate nu, coverage sharpens to
  >= 1 - alpha*(n+1)/(n-m+2) - 2*sqrt(nu) (``stability_coverage_floor``).
- Proposition 1 (tightness): the factor two cannot be improved without
  stability — for every nu in (0, 2*alpha] there are score functions whose
  limiting coverage is exactly 1 - 2*alpha + nu. Their Appendix A.4
  construction on Unif(0,1) labels alternates s_i(z) = z (i odd) with the
  fold s_i(z) = z*1{z not in [l,r]} + (l+r-z)*1{z in [l,r]} (i even);
  ``AlternatingFoldScorer`` implements it verbatim.

Contrast with split-CP (their Eq. (3)): split conformal scores every
calibration point AND the test point with ONE frozen score function
s(. ; (Z_j)_{j in I}), giving exact P(cover) >= 1 - alpha — but only from a
held-out calibration set and a model frozen at the split. In sequential
training the model keeps moving. ``SplitConformal`` reproduces the paper's
frozen-at-m baseline (their Sec. 4.1, Fig. 2, setting 1: rolling-CP sets are
substantially narrower because they exploit models trained on all n points,
while coverage is similar). ``RollingConformal.naive_pvalue`` implements the
guarantee-free practitioner shortcut — calibrate each point at arrival time,
test the candidate with the final model — which violates the fixed-score
premise of Eq. (3) and can undercover arbitrarily badly; rolling-CP's
comparisons are same-state (s_i(z) vs s_i(Z_i)), which is exactly what buys
the universal 1 - 2*alpha floor. Unlike ACI-style online conformal (their
Sec. 1.4), the target here is per-time marginal coverage, not long-run
average coverage.

Regression intervals: for residual-type scores s_i((x,y)) = g(|y - mu_i(x)|)
with g nondecreasing — g = id ('abs') or g(t) = t^2/2 ('sq'; the squared
residual score of their Sec. 4.1 OLS experiment, where mu_i(x) = x^T theta_{i-1}
runs over the min-norm least-squares trajectory — ``OnlineRidge`` with small
``lam`` approximates it) — the level set {y : p_rolling(y) > alpha} is the
depth level set of the intervals [mu_i(x) - g^{-1}(S_i), mu_i(x) + g^{-1}(S_i)]
(a union of closed intervals). ``RollingConformal.predict_interval`` returns
the connected component containing the current point forecast; membership is
always decided by the exact p-value criterion. ``OnlineQuantileTracker``
shows the same machinery for an asymmetric score (pinball loss at level tau:
sublevel sets [q_i - S/(1-tau), q_i + S/tau]).

Relevance: this is the conformal layer appropriate for continually
fine-tuned / one-pass-trained models — the deployment regime fx-1 is built
for. NO claim about fx-1 itself is made or implied here.

Honesty (AGENTS.md): only coverage / width / p-value diagnostics are exposed.
1 - 2*alpha is a WORST-CASE floor and it is tight (Proposition 1), not
typical behaviour: under stability, empirical coverage approaches 1 - alpha.
All tests are seeded SYNTHETIC correctness checks, never market evidence.

Fail-closed: alpha outside (0, 1), non-finite inputs, feature-dimension
changes, intervals requested when alpha*(n+1) <= 1 (the set is all of R) or
from scorers lacking the interval protocol, and calibration samples too
small for the requested alpha all raise. Conventions: numpy core; the paper's
'>=' convention in p_rolling fixes tie handling deterministically.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "AlternatingFoldScorer",
    "IntervalScorer",
    "LastPointOverfitter",
    "OnlineMeanTracker",
    "OnlineQuantileTracker",
    "OnlineRidge",
    "RollingConformal",
    "RollingCPResult",
    "SequentialScorer",
    "SplitConformal",
    "SplitCPResult",
    "marginal_coverage_floor",
    "rolling_pvalue",
    "split_pvalue",
    "stability_coverage_floor",
    "training_conditional_miscoverage_bound",
]

# Float-noise guard for the integer threshold alpha*(n+1) - 1 (exact-integer
# cases such as alpha=0.7, n+1=10 must floor to the mathematical integer).
_KMIN_EPS = 1e-9


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _as_1d_x(x: Array) -> Array:
    xa = np.asarray(x, dtype=float)
    if xa.ndim != 1:
        raise ValueError("x must be a 1-D feature vector")
    if xa.size == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(xa)):
        raise ValueError("x must be finite")
    return xa


def _check_y(y: float) -> float:
    ya = np.asarray(y, dtype=float)
    if ya.size != 1:
        raise ValueError("y must be a scalar")
    yv = float(ya.ravel()[0])
    if not math.isfinite(yv):
        raise ValueError("y must be finite")
    return yv


def _check_stream(X: Array, y: Array) -> tuple[Array, Array]:
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.ndim != 2:
        raise ValueError("X must be 2-D (n, d)")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError("X and y length mismatch")
    if not (np.all(np.isfinite(Xa)) and np.all(np.isfinite(ya))):
        raise ValueError("X and y must be finite")
    return Xa, ya


def _k_min(alpha: float, denom: int) -> int:
    """Smallest integer count with (1 + count)/denom > alpha (Eq. (4) threshold).

    ``count >= k_min  <=>  count > alpha*denom - 1  <=>  p > alpha`` for the
    conformal p-value ``(1 + count)/denom``. Raises when ``alpha*denom <= 1``:
    the prediction set is then all of R and no finite interval exists.
    """
    t = alpha * denom - 1.0
    k = math.floor(t + _KMIN_EPS) + 1
    if k < 1:
        raise ValueError(
            f"alpha * (n+1) = {alpha * denom:.6g} <= 1: the conformal set is all of R "
            "(or there are no calibration scores); no finite interval exists"
        )
    return int(k)


def rolling_pvalue(calibration_scores: Array, candidate_scores: Array) -> float:
    """Rolling conformal p-value (Cheng et al. 2026, Sec. 2.2).

    ``p_rolling = (1 + #{i : S_i >= s_i(z)}) / (n + 1)`` where ``S_i`` are the
    arrival-time scores ``s_i(Z_i; Z_<i)`` and ``candidate_scores[i]`` is
    ``s_i(z; Z_<i)`` — the SAME model state per pair, which is the rolling
    construction. Covered at level alpha iff ``p_rolling > alpha``. Not
    necessarily a valid p-value; it is a p*-value (their Theorem 2), so
    ``P(p <= alpha) <= 2*alpha`` (their Eq. (6)). Empty inputs give p = 1
    (C_0 = Z by their Eq. (4), since 0 < (1 - alpha) * 1).
    """
    S = np.asarray(calibration_scores, dtype=float).ravel()
    c = np.asarray(candidate_scores, dtype=float).ravel()
    if S.shape != c.shape:
        raise ValueError("calibration and candidate score trajectories must have equal length")
    if S.size and not (np.all(np.isfinite(S)) and np.all(np.isfinite(c))):
        raise ValueError("scores must be finite")
    return float((1.0 + np.count_nonzero(c <= S)) / (S.size + 1.0))  # #{i : S_i >= s_i(z)}


def split_pvalue(calibration_scores: Array, candidate_score: float) -> float:
    """Split-conformal p-value with ONE fixed score function (their App. C.2).

    ``p = (1 + #{i : s(z) <= S_i}) / (n1 + 1)`` where every ``S_i`` and the
    candidate score come from the same frozen score function. Under
    exchangeability p stochastically dominates Unif(0,1) — a valid p-value,
    giving exact ``P(cover) >= 1 - alpha``. Validity lives in the fixed-score
    premise, not the formula: feeding arrival-time scores of an evolving
    scorer (``RollingConformal.naive_pvalue``) has NO guarantee.
    """
    S = np.asarray(calibration_scores, dtype=float).ravel()
    if S.size and not np.all(np.isfinite(S)):
        raise ValueError("calibration scores must be finite")
    c = float(candidate_score)
    if not math.isfinite(c):
        raise ValueError("candidate score must be finite")
    return float((1.0 + np.count_nonzero(c <= S)) / (S.size + 1.0))


def marginal_coverage_floor(alpha: float, n: int) -> float:
    """Marginal coverage floor of rolling-CP after n observations.

    Theorem 1 gives ``1 - 2*alpha``; Remark 1 sharpens it to
    ``1 - (2*alpha - 1/(n+1))_+`` (the bound returned here). Worst case only
    — tight by Proposition 1, NOT typical (stable scorers approach 1 - alpha).
    """
    a = _check_alpha(alpha)
    nn = int(n)
    if nn < 0:
        raise ValueError("n must be >= 0")
    return float(1.0 - max(2.0 * a - 1.0 / (nn + 1.0), 0.0))


def stability_coverage_floor(alpha: float, n: int, m: int, nu: float) -> float:
    """Theorem 4 (Eq. (10a)) floor under score-comparison stability.

    ``1 - alpha*(n+1)/(n-m+2) - 2*sqrt(nu)`` when Assumption 1 holds with
    stabilization time ``m`` and disagreement rate ``nu`` — the guarantee
    sharpening towards 1 - alpha for stable predictors.
    """
    a = _check_alpha(alpha)
    nn, mm = int(n), int(m)
    if nn < 0 or mm < 0 or mm > nn:
        raise ValueError("require 0 <= m <= n")
    v = float(nu)
    if not np.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError("nu must be in [0, 1]")
    return float(1.0 - a * (nn + 1.0) / (nn - mm + 2.0) - 2.0 * math.sqrt(v))


def training_conditional_miscoverage_bound(
    n: int, alpha: float, delta: float, *, uniform_over_time: bool = False
) -> float:
    """Theorem 3 high-probability training-conditional miscoverage bound (i.i.d.).

    With probability >= 1 - delta, ``alpha_P(Z_<n+1) <=`` the returned bound:
    fixed-n ``2*alpha + sqrt(2*log(1/delta) / (alpha^2*(n+1)))``; their Eq. (7)
    uniform-over-time version replaces ``log(1/delta)`` by ``log((n+1)^2/delta)``
    and then holds simultaneously for all n >= 0. Conservative at moderate n
    (can exceed 1, i.e. vacuous) — an asymptotic guarantee; reported raw.
    """
    a = _check_alpha(alpha)
    nn = int(n)
    if nn < 0:
        raise ValueError("n must be >= 0")
    d = float(delta)
    if not np.isfinite(d) or not 0.0 < d < 1.0:
        raise ValueError("delta must be in (0, 1)")
    log_term = math.log((nn + 1.0) ** 2 / d) if uniform_over_time else math.log(1.0 / d)
    return float(2.0 * a + math.sqrt(2.0 * log_term / (a * a * (nn + 1.0))))


class SequentialScorer(Protocol):
    """Evolving score function s_i(·; Z_<i) (Cheng et al. 2026, Sec. 1.2).

    ``score(x, y)`` evaluates s_i((x, y); Z_<i) at the CURRENT state (i = n+1
    after n absorbed points); ``fit_one(x, y)`` rolls one observation into
    training, advancing s_i to s_{i+1}. The trained model is encoded entirely
    in the score function; scores may depend on the history arbitrarily —
    Theorem 1 quantifies over ANY such sequence. Point forecasts are optional
    (see ``IntervalScorer``).
    """

    def fit_one(self, x: Array, y: float) -> None: ...

    def score(self, x: Array, y: float) -> float: ...


class IntervalScorer(Protocol):
    """Optional regression extension used by ``RollingConformal.predict_interval``.

    Contract: each historical state i has a center ``mu_i(x)``
    (``predict_trajectory``), and the sublevel set
    ``{y : s_i((x, y); Z_<i) <= S}`` is the closed interval
    ``[mu_i(x) - a(S), mu_i(x) + b(S)]`` with ``(a, b) = score_radius(S)``.
    ``predict`` returns the current-state center (the point forecast).
    """

    def predict(self, x: Array) -> float: ...

    def predict_trajectory(self, x: Array) -> Array: ...

    def score_radius(self, s: float) -> tuple[float, float]: ...


def _require_scorer(obj: object) -> None:
    if not (callable(getattr(obj, "fit_one", None)) and callable(getattr(obj, "score", None))):
        raise TypeError("scorer must implement fit_one(x, y) and score(x, y)")


def _check_score_kind(score: str) -> str:
    if score not in ("abs", "sq"):
        raise ValueError("score must be 'abs' (|y - mu|) or 'sq' ((y - mu)^2 / 2)")
    return score


def _g_abs(r: Array | float) -> Array | float:
    return r


def _g_sq(r: Array | float) -> Array | float:
    return 0.5 * np.square(r) if isinstance(r, np.ndarray) else 0.5 * float(r) * float(r)


def _depth_components(
    centers: Array, rad_lo: Array, rad_hi: Array, k_min: int
) -> list[tuple[float, float, int]]:
    """Maximal closed regions where at least ``k_min`` of the intervals cover.

    Interval i is ``[centers[i] - rad_lo[i], centers[i] + rad_hi[i]]``; the
    depth of y is the number of intervals containing it. Returns the connected
    components of ``{y : depth(y) >= k_min}`` as ``(lo, hi, max_depth)``,
    merged and sorted left to right (deterministic). Degenerate point
    components (zero-radius scores) are handled via breakpoint depths.
    """
    n = int(centers.size)
    if n == 0:
        return []
    lo = np.asarray(centers, dtype=float) - np.asarray(rad_lo, dtype=float)
    hi = np.asarray(centers, dtype=float) + np.asarray(rad_hi, dtype=float)
    lo_s = np.sort(lo)
    hi_s = np.sort(hi)
    pts = np.unique(np.concatenate([lo_s, hi_s]))
    pt_depth = np.searchsorted(lo_s, pts, side="right") - np.searchsorted(hi_s, pts, side="left")
    if pts.size == 1:
        d = int(pt_depth[0])
        return [(float(pts[0]), float(pts[0]), d)] if d >= k_min else []
    mids = 0.5 * (pts[:-1] + pts[1:])
    gap_depth = np.searchsorted(lo_s, mids, side="right") - np.searchsorted(hi_s, mids, side="left")
    gap_ok = gap_depth >= k_min
    pt_ok = pt_depth >= k_min
    comps: list[tuple[float, float, int]] = []
    if np.any(gap_ok):
        idx = np.flatnonzero(gap_ok)
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([idx[:1], idx[splits + 1]])
        ends = np.concatenate([idx[splits], idx[-1:]])
        for s_k, e_k in zip(starts, ends, strict=True):
            s_i, e_i = int(s_k), int(e_k)
            comps.append(
                (float(pts[s_i]), float(pts[e_i + 1]), int(gap_depth[s_i : e_i + 1].max()))
            )
    left_ok = np.concatenate([np.zeros(1, dtype=bool), gap_ok])
    right_ok = np.concatenate([gap_ok, np.zeros(1, dtype=bool)])
    for k in np.flatnonzero(pt_ok & ~left_ok & ~right_ok):
        comps.append((float(pts[k]), float(pts[k]), int(pt_depth[k])))
    comps.sort(key=lambda c: (c[0], c[1]))
    merged: list[tuple[float, float, int]] = []
    for c_lo, c_hi, c_d in comps:
        if merged and c_lo <= merged[-1][1]:
            p_lo, p_hi, p_d = merged[-1]
            merged[-1] = (p_lo, max(p_hi, c_hi), max(p_d, c_d))
        else:
            merged.append((c_lo, c_hi, c_d))
    return merged


def _select_component(comps: list[tuple[float, float, int]], point: float) -> tuple[float, float]:
    """Component containing ``point``; else deepest, widest, leftmost; else empty.

    The empty set is encoded as ``(inf, -inf)`` (lo > hi): mathematically
    legitimate when no y reaches depth ``k_min``.
    """
    for c_lo, c_hi, _ in comps:
        if c_lo <= point <= c_hi:
            return c_lo, c_hi
    if not comps:
        return math.inf, -math.inf
    best = max(comps, key=lambda c: (c[2], c[1] - c[0], -c[0]))
    return best[0], best[1]


@dataclass(frozen=True)
class RollingCPResult:
    """Batch output of ``RollingConformal.run_stream`` (coverage/width only)."""

    lower: Array
    upper: Array
    point: Array
    pvalues: Array
    covered: NDArray[np.bool_]
    coverage: float
    mean_width: float


@dataclass(frozen=True)
class SplitCPResult:
    """Batch output of ``SplitConformal.run_stream`` (coverage/width only)."""

    lower: Array
    upper: Array
    point: Array
    pvalues: Array
    covered: NDArray[np.bool_]
    coverage: float
    mean_width: float


class RollingConformal:
    """Rolling-CP driver: calibrate-then-roll over a sequential scorer.

    Parameters
    ----------
    scorer_factory:
        Zero-arg callable returning a fresh ``SequentialScorer`` (used both
        for the live state and, for scorers without ``score_trajectory``, to
        replay the stored stream when scoring candidates against every
        historical model state).
    alpha:
        Target miscoverage in (0, 1). Guarantee: marginal coverage
        >= 1 - 2*alpha under exchangeability alone (Theorem 1); approaches
        1 - alpha under stability (Theorem 4).

    Notes
    -----
    ``update(x, y)`` scores the incoming point with the current model state
    (calibrate), then rolls it into training (``fit_one``) — the paper's order.
    ``pvalue`` / ``covered`` implement p_rolling exactly; ``predict_interval``
    additionally requires the ``IntervalScorer`` protocol. ``run_stream``
    rolls a whole stream and evaluates a test batch against the final state
    (marginal coverage across independent test draws, as in the paper's
    hold-out evaluations). State is streaming: ``run_stream`` continues from
    whatever has already been absorbed (construct a fresh driver per stream).
    """

    def __init__(self, scorer_factory: Callable[[], SequentialScorer], alpha: float = 0.1) -> None:
        if not callable(scorer_factory):
            raise TypeError("scorer_factory must be callable")
        self.alpha = _check_alpha(alpha)
        self._factory = scorer_factory
        self._scorer: SequentialScorer = scorer_factory()
        _require_scorer(self._scorer)
        self._xs: list[Array] = []
        self._ys: list[float] = []
        self._scores: list[float] = []
        self._d: int | None = None

    @property
    def n_obs(self) -> int:
        return len(self._scores)

    @property
    def scores(self) -> Array:
        """Arrival-time calibration scores S_i = s_i(Z_i; Z_<i), shape (n,)."""
        return np.asarray(self._scores, dtype=float)

    def update(self, x: Array, y: float) -> float:
        """Calibrate one incoming point against the current state, then roll it in.

        Returns the recorded arrival score S_i. The scorer sees the point
        strictly AFTER it is scored (calibrate-then-roll).
        """
        xv = _as_1d_x(x)
        yv = _check_y(y)
        if self._d is None:
            self._d = int(xv.shape[0])
        elif xv.shape[0] != self._d:
            raise ValueError(f"feature dimension changed: {self._d} -> {xv.shape[0]}")
        s = float(self._scorer.score(xv, yv))
        if not math.isfinite(s):
            raise ValueError("score function returned a non-finite score")
        self._scorer.fit_one(xv, yv)
        self._xs.append(xv)
        self._ys.append(yv)
        self._scores.append(s)
        return s

    def fit_stream(self, X: Array, y: Array) -> RollingConformal:
        Xa, ya = _check_stream(X, y)
        for j in range(Xa.shape[0]):
            self.update(Xa[j], float(ya[j]))
        return self

    def _candidate_scores(self, x: Array, y: float) -> Array:
        """s_i((x, y); Z_<i) for i = 1..n — trajectory fast path, else replay."""
        n = len(self._scores)
        traj_fn = getattr(self._scorer, "score_trajectory", None)
        if callable(traj_fn):
            out = np.asarray(traj_fn(x, y), dtype=float).ravel()
            if out.shape != (n,):
                raise ValueError("score_trajectory must return one score per observed point")
            if n and not np.all(np.isfinite(out)):
                raise ValueError("score_trajectory returned non-finite scores")
            return out
        fresh = self._factory()
        _require_scorer(fresh)
        out = np.empty(n, dtype=float)
        for j in range(n):
            s = float(fresh.score(x, y))
            if not math.isfinite(s):
                raise ValueError("score function returned a non-finite score")
            out[j] = s
            fresh.fit_one(self._xs[j], self._ys[j])
        return out

    def pvalue(self, x_test: Array, y_test: float) -> float:
        """p_rolling for candidate z = (x_test, y_test) (their Sec. 2.2)."""
        xv = _as_1d_x(x_test)
        yv = _check_y(y_test)
        cand = self._candidate_scores(xv, yv)
        return rolling_pvalue(np.asarray(self._scores, dtype=float), cand)

    def covered(self, x_test: Array, y_test: float) -> bool:
        """Z in C_n iff p_rolling > alpha (equivalent to their Eq. (4))."""
        return self.pvalue(x_test, y_test) > self.alpha

    def evaluate(self, X_test: Array, y_test: Array) -> tuple[Array, NDArray[np.bool_]]:
        """Batch p-values and coverage indicators against the current state."""
        Xa, ya = _check_stream(X_test, y_test)
        if Xa.shape[0] == 0:
            raise ValueError("test batch must be non-empty")
        ps = np.array([self.pvalue(Xa[j], float(ya[j])) for j in range(Xa.shape[0])], dtype=float)
        return ps, ps > self.alpha

    def predict_interval(self, x_test: Array) -> tuple[float, float, float]:
        """(lower, upper, point) for the rolling-CP set at feature ``x_test``.

        The returned interval is the connected component of
        ``{y : p_rolling((x, y)) > alpha}`` containing the current point
        forecast (the full level set can in principle be disconnected;
        ``covered``/``pvalue`` remain the exact membership criteria). The
        empty set is encoded as ``(inf, -inf)``. Requires ``IntervalScorer``;
        requires ``alpha * (n + 1) > 1`` (else the set is all of R).
        """
        xv = _as_1d_x(x_test)
        pred_fn = getattr(self._scorer, "predict", None)
        traj_fn = getattr(self._scorer, "predict_trajectory", None)
        rad_fn = getattr(self._scorer, "score_radius", None)
        if not (callable(pred_fn) and callable(traj_fn) and callable(rad_fn)):
            raise TypeError(
                "interval construction requires the scorer to implement predict, "
                "predict_trajectory and score_radius (IntervalScorer protocol)"
            )
        point = float(pred_fn(xv))
        if not math.isfinite(point):
            raise ValueError("point forecast must be finite")
        n = len(self._scores)
        k_min = _k_min(self.alpha, n + 1)
        centers = np.asarray(traj_fn(xv), dtype=float).ravel()
        if centers.shape != (n,):
            raise ValueError("predict_trajectory must return one center per observed point")
        if n and not np.all(np.isfinite(centers)):
            raise ValueError("predict_trajectory returned non-finite centers")
        radii = np.empty((n, 2), dtype=float)
        for j in range(n):
            pair = rad_fn(float(self._scores[j]))
            a, b = float(pair[0]), float(pair[1])
            if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b < 0.0:
                raise ValueError("score_radius must return finite non-negative offsets")
            radii[j, 0] = a
            radii[j, 1] = b
        comps = _depth_components(centers, radii[:, 0], radii[:, 1], k_min)
        lo, hi = _select_component(comps, point)
        return lo, hi, point

    def naive_pvalue(self, x_test: Array, y_test: float) -> float:
        """Guarantee-FREE contrast: arrival-time calibration, final-model test.

        Candidate scored with the CURRENT state s_{n+1}(.; Z_<=n) but compared
        against arrival scores S_i computed under the different states
        s_i(.; Z_<i). This is the practitioner shortcut a continually
        fine-tuned pipeline falls into; it violates the single-fixed-score
        premise of split-CP (their Eq. (3)), so there is NO coverage floor —
        not even 1 - 2*alpha. Diagnostic only; never a validity claim.
        """
        xv = _as_1d_x(x_test)
        yv = _check_y(y_test)
        cand = float(self._scorer.score(xv, yv))
        if not math.isfinite(cand):
            raise ValueError("score function returned a non-finite score")
        return split_pvalue(np.asarray(self._scores, dtype=float), cand)

    def naive_evaluate(self, X_test: Array, y_test: Array) -> tuple[Array, NDArray[np.bool_]]:
        Xa, ya = _check_stream(X_test, y_test)
        if Xa.shape[0] == 0:
            raise ValueError("test batch must be non-empty")
        ps = np.array(
            [self.naive_pvalue(Xa[j], float(ya[j])) for j in range(Xa.shape[0])], dtype=float
        )
        return ps, ps > self.alpha

    def run_stream(self, X: Array, y: Array, X_test: Array, y_test: Array) -> RollingCPResult:
        """Roll (X, y) into the driver, then evaluate the test batch."""
        self.fit_stream(X, y)
        pv, cov = self.evaluate(X_test, y_test)
        Xa, _ = _check_stream(X_test, y_test)
        h = Xa.shape[0]
        lower = np.empty(h, dtype=float)
        upper = np.empty(h, dtype=float)
        point = np.empty(h, dtype=float)
        for j in range(h):
            lower[j], upper[j], point[j] = self.predict_interval(Xa[j])
        widths = np.maximum(upper - lower, 0.0)
        return RollingCPResult(
            lower=lower,
            upper=upper,
            point=point,
            pvalues=pv,
            covered=cov,
            coverage=float(np.mean(cov)),
            mean_width=float(np.mean(widths)),
        )


class SplitConformal:
    """Frozen-model split-CP baseline (their Eq. (3); Sec. 4.1, Fig. 2, setting 1).

    Train the scorer on the first ``n_train`` points, freeze it, then score
    the remaining stream (calibration) and all test candidates with that ONE
    fixed score function — the exchangeability argument applies and coverage
    is exactly >= 1 - alpha. The price is data splitting: the frozen model
    never sees the calibration stream, so intervals are wider than rolling-CP
    at the same coverage (the paper's Fig. 2 finding).

    ``run_stream(X, y, X_test, y_test)`` refits from scratch (frozen at
    ``n_train``) and returns a ``SplitCPResult``.
    """

    def __init__(
        self,
        scorer_factory: Callable[[], SequentialScorer],
        alpha: float = 0.1,
        n_train: int = 100,
    ) -> None:
        if not callable(scorer_factory):
            raise TypeError("scorer_factory must be callable")
        self.alpha = _check_alpha(alpha)
        n = int(n_train)
        if n < 1:
            raise ValueError("n_train must be >= 1")
        self.n_train = n
        self._factory = scorer_factory
        self._scorer: SequentialScorer | None = None
        self._calib: Array = np.empty(0, dtype=float)

    @property
    def n_calib(self) -> int:
        return int(self._calib.size)

    def fit_stream(self, X: Array, y: Array) -> SplitConformal:
        Xa, ya = _check_stream(X, y)
        n = Xa.shape[0]
        if n <= self.n_train:
            raise ValueError(
                f"stream must exceed n_train={self.n_train} (need >= 1 calibration point)"
            )
        scorer = self._factory()
        _require_scorer(scorer)
        for j in range(self.n_train):
            scorer.fit_one(Xa[j], float(ya[j]))
        calib = np.empty(n - self.n_train, dtype=float)
        for j in range(self.n_train, n):
            s = float(scorer.score(Xa[j], float(ya[j])))
            if not math.isfinite(s):
                raise ValueError("score function returned a non-finite score")
            calib[j - self.n_train] = s
        self._scorer = scorer
        self._calib = calib
        return self

    def _frozen(self) -> SequentialScorer:
        if self._scorer is None:
            raise RuntimeError("SplitConformal is not fitted")
        return self._scorer

    def pvalue(self, x_test: Array, y_test: float) -> float:
        xv = _as_1d_x(x_test)
        yv = _check_y(y_test)
        cand = float(self._frozen().score(xv, yv))
        if not math.isfinite(cand):
            raise ValueError("score function returned a non-finite score")
        return split_pvalue(self._calib, cand)

    def covered(self, x_test: Array, y_test: float) -> bool:
        return self.pvalue(x_test, y_test) > self.alpha

    def predict_interval(self, x_test: Array) -> tuple[float, float, float]:
        """Frozen-model interval: component of {y : p_split(y) > alpha} at x_test."""
        xv = _as_1d_x(x_test)
        scorer = self._frozen()
        pred_fn = getattr(scorer, "predict", None)
        rad_fn = getattr(scorer, "score_radius", None)
        if not (callable(pred_fn) and callable(rad_fn)):
            raise TypeError(
                "interval construction requires the scorer to implement predict "
                "and score_radius (IntervalScorer protocol)"
            )
        point = float(pred_fn(xv))
        if not math.isfinite(point):
            raise ValueError("point forecast must be finite")
        k_min = _k_min(self.alpha, self.n_calib + 1)
        centers = np.full(self.n_calib, point, dtype=float)
        radii = np.empty((self.n_calib, 2), dtype=float)
        for j in range(self.n_calib):
            pair = rad_fn(float(self._calib[j]))
            a, b = float(pair[0]), float(pair[1])
            if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b < 0.0:
                raise ValueError("score_radius must return finite non-negative offsets")
            radii[j, 0] = a
            radii[j, 1] = b
        comps = _depth_components(centers, radii[:, 0], radii[:, 1], k_min)
        lo, hi = _select_component(comps, point)
        return lo, hi, point

    def run_stream(self, X: Array, y: Array, X_test: Array, y_test: Array) -> SplitCPResult:
        self.fit_stream(X, y)
        Xa, ya = _check_stream(X_test, y_test)
        h = Xa.shape[0]
        if h == 0:
            raise ValueError("test batch must be non-empty")
        pv = np.array([self.pvalue(Xa[j], float(ya[j])) for j in range(h)], dtype=float)
        cov = pv > self.alpha
        lower = np.empty(h, dtype=float)
        upper = np.empty(h, dtype=float)
        point = np.empty(h, dtype=float)
        for j in range(h):
            lower[j], upper[j], point[j] = self.predict_interval(Xa[j])
        widths = np.maximum(upper - lower, 0.0)
        return SplitCPResult(
            lower=lower,
            upper=upper,
            point=point,
            pvalues=pv,
            covered=cov,
            coverage=float(np.mean(cov)),
            mean_width=float(np.mean(widths)),
        )


class OnlineRidge:
    """Sequential ridge regression with closed-form recursive updates.

    ``theta_i = argmin (1/2) sum_{j<=i} (y_j - x_j^T theta)^2 + (lam/2)||theta||^2``
    maintained via Sherman-Morrison on ``P_i = (sum_j x_j x_j^T + lam I)^{-1}``:

        P_x = P x;  denom = 1 + x^T P_x
        theta <- theta + P_x (y - x^T theta) / denom
        P     <- P - P_x P_x^T / denom

    O(d^2) per point, no refits. The full estimator trajectory
    ``theta_0..theta_n`` (theta_0 = 0, the paper's initialization convention)
    is stored, matching their Sec. 4.1 procedure of computing "the full
    min-norm least-squares trajectory"; ``lam -> 0+`` approximates min-norm
    OLS while ``lam > 0`` keeps every update well-posed (fail-closed:
    ``lam`` must be positive). Score: 'abs' = |y - x^T theta_{i-1}| or
    'sq' = (y - x^T theta_{i-1})^2 / 2 (the squared residual score of their
    OLS experiment).
    """

    def __init__(self, n_features: int, lam: float = 1.0, score: str = "abs") -> None:
        d = int(n_features)
        if d < 1:
            raise ValueError("n_features must be >= 1")
        lam_f = float(lam)
        if not np.isfinite(lam_f) or lam_f <= 0.0:
            raise ValueError("lam must be positive and finite (lam -> 0 approximates min-norm OLS)")
        self.n_features = d
        self.lam = lam_f
        self.score_kind = _check_score_kind(score)
        self._theta = np.zeros(d, dtype=float)
        self._P = np.eye(d, dtype=float) / lam_f
        self._hist: list[Array] = [self._theta.copy()]
        self._n = 0

    @property
    def n_obs(self) -> int:
        return self._n

    def fit_one(self, x: Array, y: float) -> None:
        xv = _as_1d_x(x)
        yv = _check_y(y)
        if xv.shape[0] != self.n_features:
            raise ValueError(f"x has {xv.shape[0]} features, expected {self.n_features}")
        Px = self._P @ xv
        denom = 1.0 + float(xv @ Px)
        if not math.isfinite(denom) or denom <= 0.0:
            raise FloatingPointError("ridge recursion lost positive definiteness")
        resid = yv - float(xv @ self._theta)
        self._theta = self._theta + Px * (resid / denom)
        self._P -= np.outer(Px, Px) / denom
        self._P = 0.5 * (self._P + self._P.T)  # numerical hygiene
        self._hist.append(self._theta.copy())
        self._n += 1

    @property
    def theta(self) -> Array:
        """Current estimator theta_n (copy)."""
        return self._theta.copy()

    def predict(self, x: Array) -> float:
        xv = _as_1d_x(x)
        if xv.shape[0] != self.n_features:
            raise ValueError(f"x has {xv.shape[0]} features, expected {self.n_features}")
        return float(xv @ self._theta)

    def predict_trajectory(self, x: Array) -> Array:
        """mu_i(x) = x^T theta_{i-1} for i = 1..n, shape (n,)."""
        xv = _as_1d_x(x)
        if xv.shape[0] != self.n_features:
            raise ValueError(f"x has {xv.shape[0]} features, expected {self.n_features}")
        H = np.asarray(self._hist[: self._n], dtype=float)
        return np.asarray(H @ xv, dtype=float)

    def score(self, x: Array, y: float) -> float:
        r = abs(_check_y(y) - self.predict(x))
        return float(0.5 * r * r) if self.score_kind == "sq" else float(r)

    def score_trajectory(self, x: Array, y: float) -> Array:
        r = np.abs(_check_y(y) - self.predict_trajectory(x))
        return np.asarray(0.5 * r * r if self.score_kind == "sq" else r, dtype=float)

    def score_radius(self, s: float) -> tuple[float, float]:
        sv = float(s)
        if not math.isfinite(sv) or sv < 0.0:
            raise ValueError("s must be a finite non-negative score")
        rad = math.sqrt(2.0 * sv) if self.score_kind == "sq" else sv
        return rad, rad


class OnlineMeanTracker:
    """Sequential running-mean location tracker (x validated, then ignored).

    mu_1 = ``prior_mean`` (the paper's theta_0 = 0 convention) and mu_{i+1} is
    the mean of y_1..y_i, updated in O(1). A stable predictor: the score
    functions converge, so this sits in the Theorem 4 regime where rolling-CP
    coverage sharpens towards 1 - alpha. Score: 'abs' or 'sq' residual.
    """

    def __init__(self, prior_mean: float = 0.0, score: str = "abs") -> None:
        pm = float(prior_mean)
        if not math.isfinite(pm):
            raise ValueError("prior_mean must be finite")
        self.prior_mean = pm
        self.score_kind = _check_score_kind(score)
        self._mean = pm
        self._hist: list[float] = [pm]
        self._n = 0

    @property
    def n_obs(self) -> int:
        return self._n

    def fit_one(self, x: Array, y: float) -> None:
        _as_1d_x(x)
        yv = _check_y(y)
        self._mean = (self._mean * self._n + yv) / (self._n + 1)
        self._hist.append(self._mean)
        self._n += 1

    def predict(self, x: Array) -> float:
        _as_1d_x(x)
        return float(self._mean)

    def predict_trajectory(self, x: Array) -> Array:
        _as_1d_x(x)
        return np.asarray(self._hist[: self._n], dtype=float)

    def score(self, x: Array, y: float) -> float:
        r = abs(_check_y(y) - self.predict(x))
        return float(0.5 * r * r) if self.score_kind == "sq" else float(r)

    def score_trajectory(self, x: Array, y: float) -> Array:
        r = np.abs(_check_y(y) - self.predict_trajectory(x))
        return np.asarray(0.5 * r * r if self.score_kind == "sq" else r, dtype=float)

    def score_radius(self, s: float) -> tuple[float, float]:
        sv = float(s)
        if not math.isfinite(sv) or sv < 0.0:
            raise ValueError("s must be a finite non-negative score")
        rad = math.sqrt(2.0 * sv) if self.score_kind == "sq" else sv
        return rad, rad


def _pinball(y: float | Array, q: float | Array, tau: float) -> float | Array:
    """Pinball (quantile) loss: tau*(y-q) if y >= q else (1-tau)*(q-y)."""
    diff = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    out = np.where(diff >= 0.0, tau * diff, (tau - 1.0) * diff)
    return float(out) if out.ndim == 0 else np.asarray(out, dtype=float)


class OnlineQuantileTracker:
    """Sequential running-tau-quantile tracker with pinball scores.

    q_1 = ``prior`` (initialization convention), q_{i+1} = tau-quantile
    (linear interpolation, as ``numpy.quantile``) of y_1..y_i, maintained by
    sorted insertion — O(n) per point, no refits. Score is the pinball loss
    s_i((x, y)) = pinball_tau(y - q_i), the standard quantile-regression
    nonconformity; its sublevel sets are the ASYMMETRIC intervals
    [q_i - S/(1-tau), q_i + S/tau] (``score_radius``), demonstrating the
    driver's interval machinery beyond symmetric residuals. x is validated,
    then ignored. Stable predictor: Theorem 4 regime.
    """

    def __init__(self, tau: float = 0.5, prior: float = 0.0) -> None:
        t = float(tau)
        if not np.isfinite(t) or not 0.0 < t < 1.0:
            raise ValueError("tau must be in (0, 1)")
        pr = float(prior)
        if not math.isfinite(pr):
            raise ValueError("prior must be finite")
        self.tau = t
        self.prior = pr
        self._sorted = np.empty(0, dtype=float)
        self._hist: list[float] = [pr]
        self._n = 0

    @property
    def n_obs(self) -> int:
        return self._n

    def fit_one(self, x: Array, y: float) -> None:
        _as_1d_x(x)
        yv = _check_y(y)
        pos = int(np.searchsorted(self._sorted, yv, side="left"))
        self._sorted = np.insert(self._sorted, pos, yv)
        self._hist.append(float(np.quantile(self._sorted, self.tau)))
        self._n += 1

    def predict(self, x: Array) -> float:
        _as_1d_x(x)
        return float(self._hist[-1])

    def predict_trajectory(self, x: Array) -> Array:
        _as_1d_x(x)
        return np.asarray(self._hist[: self._n], dtype=float)

    def score(self, x: Array, y: float) -> float:
        out = _pinball(_check_y(y), self.predict(x), self.tau)
        return float(out)

    def score_trajectory(self, x: Array, y: float) -> Array:
        out = _pinball(_check_y(y), self.predict_trajectory(x), self.tau)
        return np.asarray(out, dtype=float).ravel()

    def score_radius(self, s: float) -> tuple[float, float]:
        sv = float(s)
        if not math.isfinite(sv) or sv < 0.0:
            raise ValueError("s must be a finite non-negative score")
        return sv / (1.0 - self.tau), sv / self.tau


class AlternatingFoldScorer:
    """DELIBERATELY UNSTABLE: the adversarial scores of Proposition 1, App. A.4.

    Data-independent scores on the label z (x validated, then ignored):
    ``s_i(z) = z`` for odd i, and the fold
    ``s_i(z) = z*1{z not in [lo, hi]} + (lo+hi-z)*1{z in [lo, hi]}`` for even
    i. ``fit_one`` only advances the step counter — Theorem 1 quantifies over
    ANY score functions, so this is a legal rolling-CP scorer. With
    Unif(0, 1) labels, ``lo < 1 - alpha < hi`` and ``(lo+hi)/2 > 1 - alpha``
    (take ``lo = 1 - 2*alpha + nu``, ``hi = 1``, ``nu`` in (0, alpha)), their
    DKW argument gives ``C_n -> [0, lo)`` almost surely, i.e. limiting
    coverage exactly ``1 - 2*alpha + nu``: the factor-two floor is ATTAINED.
    This is the worst case, not typical behaviour — stable scorers approach
    1 - alpha. Has no point forecast, so interval construction fail-closes
    (TypeError); use p-values / coverage.
    """

    def __init__(self, lo: float, hi: float) -> None:
        a, b = float(lo), float(hi)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("lo and hi must be finite")
        if not a < b:
            raise ValueError("require lo < hi (paper: lo < 1 - alpha < hi)")
        self.lo = a
        self.hi = b
        self._n = 0

    @property
    def n_obs(self) -> int:
        return self._n

    def fit_one(self, x: Array, y: float) -> None:
        _as_1d_x(x)
        _check_y(y)
        self._n += 1

    def score(self, x: Array, y: float) -> float:
        _as_1d_x(x)
        yv = _check_y(y)
        i = self._n + 1
        if i % 2 == 1:
            return yv
        if self.lo <= yv <= self.hi:
            return self.lo + self.hi - yv
        return yv

    def score_trajectory(self, x: Array, y: float) -> Array:
        _as_1d_x(x)
        yv = _check_y(y)
        n = self._n
        i = np.arange(1, n + 1, dtype=np.int64)
        folded = self.lo + self.hi - yv if self.lo <= yv <= self.hi else yv
        out = np.where(i % 2 == 0, folded, yv).astype(float)
        return np.asarray(out, dtype=float)


class LastPointOverfitter:
    """DELIBERATELY UNSTABLE: memorizes only the most recent label, growing gain.

    ``mu_i(x) = g_i * y_{i-1}`` (x validated, then ignored; ``mu_1 = 0`` by
    the initialization convention) with a gain that grows with the training
    step: ``g_i = 1 + (i-1)/gain_scale`` ('linear') or
    ``g_i = exp((i-1)/gain_scale)`` ('exp'). Score: ``|y - mu_i(x)|``. Think
    continual fine-tuning that increasingly overfits the latest batch while
    ignoring features.

    Purpose (tests document the regime): arrival-time scores
    ``S_i = |y_i - g_i y_{i-1}|`` grow with i, so the naive 'calibrate at
    arrival, test with the final model' split has calibration quantiles that
    systematically lag the final-model test score — its coverage collapses
    far below 1 - 2*alpha (no floor exists once Eq. (3)'s fixed-score premise
    is violated). Rolling-CP is immune by Theorem 1 (every comparison is
    same-state). Honesty: instability ALONE does not attain the worst case —
    here rolling-CP overcovers relative to 1 - 2*alpha; the adversarial fold
    of Proposition 1 (``AlternatingFoldScorer``) is what attains it.
    """

    def __init__(self, gain_scale: float = 40.0, growth: str = "linear") -> None:
        gs = float(gain_scale)
        if not np.isfinite(gs) or gs <= 0.0:
            raise ValueError("gain_scale must be positive and finite")
        if growth not in ("linear", "exp"):
            raise ValueError("growth must be 'linear' or 'exp'")
        self.gain_scale = gs
        self.growth = growth
        self._ys: list[float] = []
        self._n = 0

    @property
    def n_obs(self) -> int:
        return self._n

    def _gain(self, i: float | Array) -> float | Array:
        t = (np.asarray(i, dtype=float) - 1.0) / self.gain_scale
        if self.growth == "exp":
            out = np.exp(t)
        else:
            out = 1.0 + t
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def fit_one(self, x: Array, y: float) -> None:
        _as_1d_x(x)
        yv = _check_y(y)
        self._ys.append(yv)
        self._n += 1

    def predict(self, x: Array) -> float:
        _as_1d_x(x)
        if self._n == 0:
            return 0.0
        g = self._gain(float(self._n + 1))
        return float(np.asarray(g, dtype=float) * self._ys[-1])

    def predict_trajectory(self, x: Array) -> Array:
        """mu_i = g_i * y_{i-1} for i = 1..n (mu_1 = 0: nothing to memorize)."""
        _as_1d_x(x)
        n = self._n
        if n == 0:
            return np.empty(0, dtype=float)
        i = np.arange(1, n + 1, dtype=float)
        g = np.asarray(self._gain(i), dtype=float)
        prev = np.concatenate(
            [np.zeros(1, dtype=float), np.asarray(self._ys[: n - 1], dtype=float)]
        )
        return np.asarray(g * prev, dtype=float)

    def score(self, x: Array, y: float) -> float:
        return abs(_check_y(y) - self.predict(x))

    def score_trajectory(self, x: Array, y: float) -> Array:
        out = np.abs(_check_y(y) - self.predict_trajectory(x))
        return np.asarray(out, dtype=float)

    def score_radius(self, s: float) -> tuple[float, float]:
        sv = float(s)
        if not math.isfinite(sv) or sv < 0.0:
            raise ValueError("s must be a finite non-negative score")
        return sv, sv
