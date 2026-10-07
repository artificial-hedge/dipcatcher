"""Exponential-tilt reweighted conformal under joint (covariate+label) shift.

Implements the generic ExTRA (Exponential Tilt Reweighting Alignment) tilt-
reweighted conformal surface of Choi, "Conformal Prediction under
Exponential-Tilt Joint Shift", arXiv:2609.30886 (stat.ML, submitted
2026-09-25; fetched and verified). The tilt estimator is the generic
marginal-matching form of the paper's Eq. 6 objective; the tilt itself is
the ExTRA reweighting of Maity et al., "Understanding new tasks through the
lens of training data via exponential tilting", ICLR 2023. The weighted
order statistic is Tibshirani, Barber, Candes & Ramdas, "Conformal
prediction under covariate shift", NeurIPS 2019, generalized to arbitrary
weights by Barber, Candes, Ramdas & Tibshirani, "Conformal prediction
beyond exchangeability", Ann. Statist. 51(2), 2023.

Setting (paper Section 2): labeled source pairs (X_i, Y_i) ~ P and unlabeled
target inputs X_j ~ Q_X, where the joint Q may shift BOTH the input marginal
and the input-response relationship — a joint shift that covariate-shift
weighting (:mod:`quant_fund.models.weighted_conformal`) and label-shift
adjustment each cover only half of. The tilt model is

    w(x, y) = exp(theta . g(x, y)) / Z_P(theta),
    Z_P(theta) = E_P[exp(theta . g(X, Y))],

with ``g`` a user feature map decomposed into X-observable features
``phi(x)`` (identified from unlabeled target inputs) and label features
``psi(x, y)`` (NOT identified without target responses — the analyst
supplies that block of theta, and this module's harm surface exists because
that block can be misspecified).

Components:

1. :func:`fit_marginal_tilt` — estimates the X-block ``theta_phi`` by the
   generic analog of paper Eq. 6: maximize

       L(t) = mean_j t . phi(X_j^Q) - log mean_i exp(t . phi(X_i^P))
              - (l2 / 2) ||t||^2,

   the empirical KL projection of the observed target inputs onto the
   tilt-induced input marginals (convex in t: a linear term minus a
   log-sum-exp plus ridge; solved by L-BFGS-B with the analytic gradient,
   deterministic fixed start). Label features must NOT enter this fit —
   they are unidentified from target inputs alone (paper Prop. 1-2).
2. :func:`tilt_weights` — stabilized, mean-1-normalized tilt weights
   ``w_i = exp(theta . g_i - max_l l_l)`` with an optional bounded-weights
   clip at a weight quantile (house-style bound, cf. ``WEIGHT_CLIP`` in
   :mod:`quant_fund.models.weighted_conformal`). Log-sum-exp stabilization
   means arbitrarily large theta underflows cleanly to exact zero weights —
   the degenerate one-sided regime of the exact reductions.
3. :func:`effective_sample_size` — Kish n_eff, imported unchanged from
   :mod:`quant_fund.models.localized_conformal` (NOT reimplemented).
4. :func:`tilted_pvalues` — tilt-adjusted weighted conformal p-values

       p(q) = [w_test + sum_i w_i 1{s_i >= q}] / [w_test + sum_i w_i],

   the Barber et al. (2023) arbitrary-weights rank with a fixed test atom
   ``w_test`` (the +inf atom convention of
   :func:`quant_fund.models.weighted_conformal.weighted_conformal_quantile`;
   the paper's candidate-dependent atom ``H_y`` lives in the sibling
   :func:`quant_fund.models.extra_tilt_conformal.extra_weighted_rank`).
5. :func:`extra_conformal_quantile` — the tilt-weighted conformal quantile
   with an ESS fail-closed guard: when Kish ESS collapses below
   ``ess_floor * n`` the weight vector is degenerate and the estimator
   falls back to the UNWEIGHTED split-conformal quantile (never emits a
   silently-degenerate set; ``on_collapse="raise"`` makes the collapse an
   error instead of a fallback). Exact reductions:

   * ``theta = 0`` gives w_i = 1 for all i — bitwise-equal to
     :func:`quant_fund.metrics.conformal.conformal_quantile` and to
     ``weighted_conformal_quantile`` under uniform weights;
   * a one-sided tilt with |theta| so large that ``exp`` underflows on one
     side gives an exact subset quantile — the weighted statistic masks
     zero-weight rows, so the result is bitwise the unweighted quantile of
     the surviving subset.
6. :func:`predictive_tilt_scores` — the optional predictive-tilting step of
   paper Eq. 10, ``S_theta = S_0 - log h_theta + log Z^pred``: the score
   shift under a tilted predictive law. Generic score-level form only; the
   closed-form conditional law it derives from is the sibling's
   :class:`quant_fund.models.extra_tilt_conformal.TiltedPredictive`.
7. :func:`tilt_path_diagnostics`, :func:`bench_extra_conformal`,
   :func:`bench_extra_harm` — diagnostics and the harm-regime suite. The
   harm suite is a first-class RED-TEAM surface in the sense of
   ``quant_fund.validation.leakage_redteam`` / the anti-gaming contract of
   :mod:`quant_fund.models.replicable_conformal`: a misspecified tilt
   direction (wrong sign on the label block) degrades coverage WORSE than
   doing nothing, on well-posed weights (high ESS), and the suite reports
   that degradation rather than hiding it — consistent with the paper's
   own finding that tilting "can instead cause substantial coverage
   losses" when the shift is misspecified or uninformative.

Composition notes (what is reused and why it is not reimplemented):
``weighted_conformal_quantile`` (weighted order statistic) and
``conformal_quantile`` (unweighted fallback) come from
:mod:`quant_fund.models.weighted_conformal` /
:mod:`quant_fund.metrics.conformal`; ``cqr_scores`` / ``expand_interval`` /
``set_metrics`` are the repo's CQR and evaluation primitives;
``effective_sample_size`` is imported from
:mod:`quant_fund.models.localized_conformal`; the planted benchmark DGP is
:func:`quant_fund.models.extra_tilt_conformal.synthetic_bimodal_shift`
(the verified paper Section 5.2 fixture). The sibling
``extra_tilt_conformal`` owns the paper's model-family-specific machinery
(closed-form conditional moments, candidate-atom ranks, half-line sets);
this module is the model-AGNOSTIC feature-map surface that precedes and
complements it, so neither duplicates the other.

References:
    Choi, S. (2026). Conformal Prediction under Exponential-Tilt Joint
        Shift. arXiv:2609.30886 [stat.ML].
    Maity, S., Dutta, D., Terhorst, J., Sun, Y., & Banerjee, M. (2023).
        Understanding new tasks through the lens of training data via
        exponential tilting. ICLR 2023. arXiv:2212.11577.
    Tibshirani, R. J., Barber, R. F., Candes, E. J., & Ramdas, A. (2019).
        Conformal prediction under covariate shift. NeurIPS 32.
    Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023).
        Conformal prediction beyond exchangeability. Ann. Statist. 51(2).

Honesty: every number this module produces is validated on SYNTHETIC
fixtures labeled as such (``synthetic_bimodal_shift`` and the module's own
``synthetic_gaussian_x_shift``) — correctness tests, never market evidence.
Research outputs are proper-score quantities only (coverage, set width,
p-values, ESS diagnostics); no P&L / Sharpe-family metrics appear anywhere.
``bench_extra_harm`` exists to DOCUMENT a failure mode, not to tune it
away. Determinism: all randomness lives inside seeded numpy Generators;
the optimizer path is deterministic from a fixed start. Fail-closed:
degenerate / empty / non-finite / shape-mismatched inputs raise
``ValueError`` or ``TypeError``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

from quant_fund.metrics.conformal import (
    conformal_quantile,
    cqr_scores,
    expand_interval,
    set_metrics,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.extra_tilt_conformal import synthetic_bimodal_shift
from quant_fund.models.localized_conformal import effective_sample_size
from quant_fund.models.weighted_conformal import weighted_conformal_quantile

Array = NDArray[np.float64]

#: Joint tilt feature map ``g(x, y) -> (n, p)``; first ``p_x`` columns must
#: be X-observable (identifiable from unlabeled target inputs).
FeatureMap = Callable[[Array, Array], Array]

#: Default ESS collapse floor, as a fraction of the calibration size: below
#: ``ess_floor * n`` the tilt weight vector is treated as degenerate.
DEFAULT_ESS_FLOOR: float = 0.05

#: Ridge on the generic marginal-matching fit (paper Eq. 6 penalty analog).
DEFAULT_TILT_L2: float = 1e-3

#: Default coefficient bounds for the X-block fit (keeps the optimizer
#: bounded when target features separate source features).
DEFAULT_TILT_BOUND: float = 10.0

WeightsMode = Literal["tilt", "uniform_fallback"]


def _finite_1d(x: object, name: str) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-d array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have all finite entries")
    return np.asarray(arr, dtype=float)


def _finite_2d(x: object, name: str) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-d array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have all finite entries")
    return np.asarray(arr, dtype=float)


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_theta(theta: object, p: int) -> Array:
    t = np.asarray(theta, dtype=float)
    if t.ndim != 1 or t.size != int(p) or not np.all(np.isfinite(t)):
        raise ValueError(f"theta must be a finite vector of length {int(p)}")
    return np.asarray(t, dtype=float)


def _nonnegative_1d(w: object, name: str) -> Array:
    arr = np.asarray(w, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-d array")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must have finite entries >= 0")
    return np.asarray(arr, dtype=float)


def sign_joint_features(x: Array, y: Array) -> Array:
    """Separable sign tilt family ``g(x, y) = [x_1..x_d, sign(y)]``.

    Generic version of the paper's Eq. 23 family ``(u, sign(y))`` extended
    to every covariate column. The response feature is BOUNDED — a tilt
    linear in ``y`` instead of ``sign(y)`` drives candidate weights past
    the forced-inclusion threshold and yields prediction sets of infinite
    length on one response tail (paper Eq. 21); the sibling module
    documents the same constraint for its closed-form family. Columns
    ``0..d-1`` are X-observable; column ``d`` is the label block.
    """
    xp = _finite_2d(x, "x")
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size != xp.shape[0]:
        raise ValueError("x and y must have one value per row")
    if not np.all(np.isfinite(yy)):
        raise ValueError("y must have all finite entries")
    z = np.where(yy >= 0.0, 1.0, -1.0)
    return np.asarray(np.column_stack([xp, z]), dtype=float)


def log_tilt_weights(features: Array, theta: Array) -> Array:
    """Raw log tilt ``ell_i = theta . g_i`` on a joint feature matrix."""
    g = _finite_2d(features, "features")
    t = _check_theta(theta, g.shape[1])
    return np.asarray(g @ t, dtype=float)


def tilt_weights(
    features: Array,
    theta: Array,
    *,
    clip_quantile: float | None = None,
) -> Array:
    """Normalized exponential-tilt weights ``w_i = exp(theta . g_i)``.

    Log-sum-exp stabilized (``exp(ell_i - max ell)``), normalized to mean
    1. ``theta = 0`` returns exact ones — the bitwise gate of the
    theta=0 reduction. ``clip_quantile`` in (0, 1] bounds the weights at
    that empirical quantile (bounded-weights option; the clip cannot push
    a weight below its current value, so clipping never creates zeros) and
    renormalizes to mean 1. Degenerate tilts that underflow to all-zero
    weights are fail-closed (``ValueError``).
    """
    g = _finite_2d(features, "features")
    t = _check_theta(theta, g.shape[1])
    if clip_quantile is not None:
        cq = float(clip_quantile)
        if not 0.0 < cq <= 1.0:
            raise ValueError("clip_quantile must be in (0, 1]")
    if float(np.max(np.abs(t))) == 0.0:
        return np.ones(g.shape[0], dtype=float)
    ell = g @ t
    w = np.exp(ell - float(np.max(ell)))
    if float(np.sum(w)) <= 0.0:
        raise ValueError("tilt weights underflowed to zero on every row")
    w = w / float(np.mean(w))
    if clip_quantile is not None:
        cap = float(np.quantile(w, clip_quantile))
        w = np.minimum(w, cap)
        w = w / float(np.mean(w))
    return np.asarray(w, dtype=float)


@dataclass(frozen=True)
class TiltDiagnostics:
    """Weight health report for a tilt vector (all proper diagnostics)."""

    n: int
    ess: float
    ess_fraction: float
    max_weight_share: float
    n_at_clip: int


def tilt_diagnostics(weights: Array) -> TiltDiagnostics:
    """ESS and concentration diagnostics on a (normalized) weight vector."""
    w = _nonnegative_1d(weights, "weights")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("weights must have positive total mass")
    ess = effective_sample_size(w)
    n = int(w.size)
    return TiltDiagnostics(
        n=n,
        ess=ess,
        ess_fraction=ess / float(n),
        max_weight_share=float(np.max(w)) / float(np.sum(w)),
        n_at_clip=int(np.sum(w >= np.max(w) - 1e-12)),
    )


@dataclass(frozen=True)
class MarginalTiltFit:
    """Generic marginal-matching fit of the X-block tilt coefficient.

    ``theta`` solves the convex program of :func:`fit_marginal_tilt`;
    ``ess_source`` is the Kish ESS of the implied source weights
    ``exp(theta . phi_i)`` on the fit sample — a weight-degeneracy signal
    independent of the downstream calibration ESS.
    """

    theta: Array
    objective: float
    converged: bool
    n_iter: int
    l2: float
    ess_source: float
    n_source: int
    m_target: int


def fit_marginal_tilt(
    source_features: Array,
    target_features: Array,
    *,
    l2: float = DEFAULT_TILT_L2,
    bound: float = DEFAULT_TILT_BOUND,
    theta0: Array | None = None,
    maxiter: int = 500,
) -> MarginalTiltFit:
    """Estimate the X-observable tilt block from source + unlabeled target.

    Maximizes the generic analog of paper Eq. 6,

    ``L(t) = mean_j t . phi(X_j^Q) - log mean_i exp(t . phi(X_i^P))
            - (l2 / 2) ||t||^2``,

    over X-observable feature matrices only — the empirical KL projection
    of the target input marginal onto the tilt-induced input marginals.
    The objective is concave in t (linear term minus log-sum-exp minus
    ridge), so the fixed start ``theta0 = 0`` reaches the global optimum;
    L-BFGS-B under the per-coordinate ``bound`` keeps iterates finite when
    the empirical features separate. Deterministic.

    Fail-closed: empty / non-finite / column-mismatched inputs, negative
    l2, non-positive bound, bad theta0, or a non-converged optimizer all
    raise. NEVER pass label features here — the label block is
    unidentified from unlabeled target inputs (paper Prop. 1-2) and must
    be supplied by the analyst.
    """
    fp = _finite_2d(source_features, "source_features")
    fq = _finite_2d(target_features, "target_features")
    if fp.shape[1] != fq.shape[1]:
        raise ValueError("source and target features must share the column count")
    p = int(fp.shape[1])
    lam = float(l2)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("l2 must be finite and >= 0")
    b = float(bound)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("bound must be finite and > 0")
    if theta0 is None:
        start = np.zeros(p, dtype=float)
    else:
        start = _check_theta(theta0, p)
        if np.any(np.abs(start) > b):
            raise ValueError("theta0 must lie inside the coefficient bound")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be >= 1")

    mean_q = np.mean(fq, axis=0)
    log_n = float(np.log(fp.shape[0]))

    def neg_objective(t: Array) -> tuple[float, Array]:
        expo = fp @ t
        lse = float(logsumexp(expo))
        obj = float(mean_q @ t) - (lse - log_n) - 0.5 * lam * float(t @ t)
        soft = np.exp(expo - lse)
        grad = mean_q - soft @ fp - lam * t
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros(p)
        return -obj, np.asarray(-grad, dtype=float)

    res = minimize(
        neg_objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-b, b)] * p,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    theta_hat = np.asarray(res.x, dtype=float)
    obj_hat, _ = neg_objective(theta_hat)
    obj_hat = -obj_hat if obj_hat < 1e11 else float("-inf")
    if not (bool(res.success) and np.isfinite(obj_hat) and np.all(np.isfinite(theta_hat))):
        raise ValueError("marginal tilt fit failed to converge")
    ell = fp @ theta_hat
    w_src = np.exp(ell - float(np.max(ell)))
    return MarginalTiltFit(
        theta=np.asarray(theta_hat, dtype=float),
        objective=float(obj_hat),
        converged=True,
        n_iter=int(res.nit),
        l2=lam,
        ess_source=effective_sample_size(w_src),
        n_source=int(fp.shape[0]),
        m_target=int(fq.shape[0]),
    )


def tilted_pvalues(
    scores: Array,
    weights: Array,
    query_scores: Array,
    *,
    test_weight: float = 1.0,
) -> Array:
    """Tilt-adjusted weighted conformal p-values for candidate scores.

    ``p(q) = [w_test + sum_i w_i 1{s_i >= q}] / [w_test + sum_i w_i]`` —
    the arbitrary-weights rank of Barber et al. (2023) with a fixed test
    atom. Tied calibration scores count toward the candidate (the
    conservative direction). At uniform weights and ``test_weight = 1``
    this is bitwise the unweighted split-conformal p-value
    ``(1 + #{s_i >= q}) / (n + 1)``. Candidates are retained iff
    ``p(q) > alpha``; for distinct scores the retained region is exactly
    ``(-inf, qhat]`` under :func:`extra_conformal_quantile`'s statistic.
    """
    s = _finite_1d(scores, "scores")
    w = _nonnegative_1d(weights, "weights")
    q = _finite_1d(query_scores, "query_scores")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("weights must have positive total mass")
    wt = float(test_weight)
    if not np.isfinite(wt) or wt <= 0.0:
        raise ValueError("test_weight must be finite and > 0")
    order = np.argsort(s, kind="mergesort")
    s_sorted, w_sorted = s[order], w[order]
    cumw = np.cumsum(w_sorted)
    total = float(cumw[-1])
    idx = np.searchsorted(s_sorted, q, side="left")
    above = total - np.concatenate([[0.0], cumw])[idx]
    return np.asarray((wt + above) / (wt + total), dtype=float)


@dataclass(frozen=True)
class ExTRAQuantile:
    """Tilt-weighted conformal quantile plus the ESS-guard record.

    ``weights_mode`` is ``"tilt"`` when the supplied weights survived the
    ESS floor and ``"uniform_fallback"`` when they collapsed and the
    unweighted split-conformal statistic was used instead — the mode is
    reported, never hidden.
    """

    qhat: float
    alpha: float
    ess: float
    ess_fraction: float
    n: int
    weights_mode: WeightsMode


def extra_conformal_quantile(
    scores: Array,
    weights: Array,
    alpha: float,
    *,
    ess_floor: float = DEFAULT_ESS_FLOOR,
    on_collapse: Literal["fallback", "raise"] = "fallback",
) -> ExTRAQuantile:
    """Weighted conformal quantile with an ESS fail-closed guard.

    Computes the Kish ESS of the supplied weights; when
    ``ess < ess_floor * n`` the weight vector is degenerate (a handful of
    rows carry all the mass) and the estimator falls back to the
    unweighted split-conformal quantile — fail-closed degradation rather
    than a silently-vacuous set — or raises ``ValueError`` under
    ``on_collapse="raise"``. ``ess_floor`` in ``[0, 1]``; pass ``0`` to
    disable the guard. Zero-weight rows are dropped by the underlying
    order statistic, which is exactly the subset-quantile reduction.
    """
    s = _finite_1d(scores, "scores")
    w = _nonnegative_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    a = _check_alpha(alpha)
    floor = float(ess_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("ess_floor must be in [0, 1]")
    if on_collapse not in ("fallback", "raise"):
        raise ValueError("on_collapse must be 'fallback' or 'raise'")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("weights must have positive total mass")
    ess = effective_sample_size(w)
    n = int(w.size)
    collapsed = ess < floor * float(n)
    if collapsed and on_collapse == "raise":
        raise ValueError(f"tilt weight ESS {ess:.3f} below floor {floor:.3f} * n={n}")
    if collapsed:
        qhat = conformal_quantile(s, a)
        mode: WeightsMode = "uniform_fallback"
    else:
        qhat = weighted_conformal_quantile(s, w, a)
        mode = "tilt"
    return ExTRAQuantile(
        qhat=float(qhat),
        alpha=a,
        ess=float(ess),
        ess_fraction=float(ess) / float(n),
        n=n,
        weights_mode=mode,
    )


def predictive_tilt_scores(
    base_scores: Array,
    log_tilt: Array,
    log_normalizer: Array | float = 0.0,
) -> Array:
    """Optional predictive-tilting step: ``S_t = S_0 - log h + log Z^pred``.

    Generic score-level form of paper Eq. 10. Under the true tilted
    predictive law the input-only factor of ``log h`` cancels against
    ``log Z^pred(x)`` (paper Section 3.1); with plug-in residual scores
    (CQR distances) this is a heuristic re-centering, so callers must pass
    the normalizer explicitly — the default 0 applies the raw ``-log h``
    shift. Componentwise shift: rows upweighted by the tilt see their
    score reduced, which SHRINKS the calibrated quantile toward the
    tilted law (paper's length-reduction direction) and is exactly the
    mechanism that loses coverage when the tilt is misspecified — the
    documented harm regime.
    """
    s = _finite_1d(base_scores, "base_scores")
    lh = _finite_1d(log_tilt, "log_tilt")
    if s.size != lh.size:
        raise ValueError("base_scores and log_tilt must have the same length")
    lz = np.asarray(log_normalizer, dtype=float)
    if lz.ndim == 0:
        lz = np.full(s.size, float(lz))
    if lz.ndim != 1 or lz.size != s.size or not np.all(np.isfinite(lz)):
        raise ValueError("log_normalizer must be finite, scalar or matching length")
    return np.asarray(s - lh + lz, dtype=float)


def extra_prediction_sets(
    lower: Array,
    upper: Array,
    calibration_scores: Array,
    weights: Array,
    alpha: float,
    *,
    ess_floor: float = DEFAULT_ESS_FLOOR,
    on_collapse: Literal["fallback", "raise"] = "fallback",
    scale: Array | None = None,
) -> tuple[Array, Array, ExTRAQuantile]:
    """Tilt-weighted CQR sets: expand ``[lower, upper]`` by the ESS-guarded qhat."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    res = extra_conformal_quantile(
        calibration_scores, weights, alpha, ess_floor=ess_floor, on_collapse=on_collapse
    )
    elo, ehi = expand_interval(lo, hi, res.qhat, scale)
    return np.asarray(elo, dtype=float), np.asarray(ehi, dtype=float), res


@dataclass(frozen=True)
class TiltPathDiagnostics:
    """ESS / quantile path along a scaled tilt direction (monotone checks)."""

    scales: Array
    ess: Array
    ess_fraction: Array
    qhat: Array
    modes: tuple[str, ...]


def tilt_path_diagnostics(
    scores: Array,
    features: Array,
    theta_direction: Array,
    scales: Array,
    alpha: float,
    *,
    ess_floor: float = DEFAULT_ESS_FLOOR,
    clip_quantile: float | None = None,
) -> TiltPathDiagnostics:
    """Evaluate the ESS-guarded quantile along ``theta = scale * direction``.

    ``scales`` is a 1-d grid of scalars (usually sorted); each row reports
    ESS and the resulting qhat at the same alpha. Width-vs-tilt-strength
    and ESS-collapse-vs-strength are both readable off the arrays —
    along a well-posed direction ESS is non-increasing in |scale| and
    qhat moves monotonically toward the upweighted region.
    """
    s = _finite_1d(scores, "scores")
    g = _finite_2d(features, "features")
    if g.shape[0] != s.size:
        raise ValueError("features must have one row per score")
    d = _check_theta(theta_direction, g.shape[1])
    sc = _finite_1d(scales, "scales")
    a = _check_alpha(alpha)
    esses: list[float] = []
    qhats: list[float] = []
    modes: list[str] = []
    for k in sc:
        w = tilt_weights(g, float(k) * d, clip_quantile=clip_quantile)
        res = extra_conformal_quantile(s, w, a, ess_floor=ess_floor)
        esses.append(res.ess)
        qhats.append(res.qhat)
        modes.append(res.weights_mode)
    return TiltPathDiagnostics(
        scales=np.asarray(sc, dtype=float),
        ess=np.asarray(esses, dtype=float),
        ess_fraction=np.asarray(esses, dtype=float) / float(s.size),
        qhat=np.asarray(qhats, dtype=float),
        modes=tuple(modes),
    )


def synthetic_gaussian_x_shift(
    n_source: int,
    n_target: int,
    a: float,
    seed: int,
) -> tuple[Array, Array]:
    """SYNTHETIC mean-shifted Gaussian covariates for the marginal fit.

    Source ``x ~ N(0, 1)``; target ``x ~ N(a, 1)`` — the exact density
    ratio is ``exp(a x - a^2 / 2)``, so under the feature map
    ``phi(x) = x`` the population marginal-matching solution is
    ``theta = a``. Correctness fixture only, never market evidence.
    """
    if int(n_source) < 1 or int(n_target) < 1:
        raise ValueError("n_source and n_target must be >= 1")
    if not np.isfinite(float(a)):
        raise ValueError("a must be finite")
    rng = np.random.default_rng(int(seed))
    xs = rng.normal(0.0, 1.0, int(n_source))
    xt = rng.normal(float(a), 1.0, int(n_target))
    return np.asarray(xs, dtype=float), np.asarray(xt, dtype=float)


class ExTRASplitCQR(JoblibMixin):
    """Split CQR with exponential-tilt weights and an ESS fail-closed guard.

    Calibration weights come from explicit ``weights``, or from
    ``feature_map(x_cal, y_cal) @ theta`` when ``theta`` is supplied. With
    neither, the class is exactly :class:`quant_fund.models.conformal.SplitCQR`
    behavior. ``predict_sets`` only needs the base bands — the joint-shift
    weights live on the calibration side.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        *,
        theta: Array | None = None,
        feature_map: FeatureMap | None = None,
        ess_floor: float = DEFAULT_ESS_FLOOR,
        clip_quantile: float | None = None,
        on_collapse: Literal["fallback", "raise"] = "fallback",
    ) -> None:
        self.alpha = _check_alpha(alpha)
        floor = float(ess_floor)
        if not 0.0 <= floor <= 1.0:
            raise ValueError("ess_floor must be in [0, 1]")
        if on_collapse not in ("fallback", "raise"):
            raise ValueError("on_collapse must be 'fallback' or 'raise'")
        if theta is not None and feature_map is None:
            raise ValueError("theta requires a feature_map to build joint features")
        self.theta = None if theta is None else np.asarray(theta, dtype=float)
        if self.theta is not None and (self.theta.ndim != 1 or self.theta.size == 0):
            raise ValueError("theta must be a non-empty 1-d array")
        self.feature_map = feature_map
        self.ess_floor = floor
        self.clip_quantile = clip_quantile
        self.on_collapse: Literal["fallback", "raise"] = on_collapse
        self.qhat = 0.0
        self.scores_: Array | None = None
        self.result_: ExTRAQuantile | None = None
        self.diagnostics_: TiltDiagnostics | None = None

    def calibrate(
        self,
        y: Array,
        lower: Array,
        upper: Array,
        *,
        x_cal: Array | None = None,
        weights: Array | None = None,
    ) -> ExTRASplitCQR:
        s = cqr_scores(y, lower, upper)
        if weights is not None:
            w = _nonnegative_1d(weights, "weights")
        elif self.theta is not None and self.feature_map is not None:
            if x_cal is None:
                raise ValueError("x_cal is required to build tilt features")
            feats = self.feature_map(np.asarray(x_cal, dtype=float), np.asarray(y, dtype=float))
            w = tilt_weights(feats, self.theta, clip_quantile=self.clip_quantile)
        else:
            w = np.ones(s.size, dtype=float)
        if w.size != s.size:
            raise ValueError("weights must align with calibration rows")
        self.scores_ = s
        self.diagnostics_ = tilt_diagnostics(w)
        self.result_ = extra_conformal_quantile(
            s, w, self.alpha, ess_floor=self.ess_floor, on_collapse=self.on_collapse
        )
        self.qhat = float(self.result_.qhat)
        return self

    def predict_sets(
        self, lower: Array, upper: Array, scale: Array | None = None
    ) -> tuple[Array, Array]:
        if self.result_ is None:
            raise ValueError("calibrate must be called before predict_sets")
        lo, hi = expand_interval(lower, upper, self.qhat, scale)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "alpha": self.alpha,
            "ess_floor": self.ess_floor,
            "weights_mode": None if self.result_ is None else self.result_.weights_mode,
            "ess": None if self.result_ is None else self.result_.ess,
        }
        return ModelMeta(
            family="conformal",
            name="extra_split_cqr",
            version="v1",
            extra=extra,
        )


@dataclass(frozen=True)
class ExTRABenchResult:
    """Bench output rows; dicts carry proper-score keys only, no Sharpe."""

    coverage_tilt: float
    coverage_unweighted: float
    mean_width_tilt: float
    mean_width_unweighted: float
    ess_fraction: float
    weights_mode: str


def _bimodal_cqr_bench(
    theta_label_block: float,
    *,
    eta: float = 1.0,
    a_star: float = 1.0,
    b_star: float = 1.2,
    alpha: float = 0.10,
    n_cal: int = 1000,
    n_test: int = 1500,
    seed: int = 0,
    ess_floor: float = DEFAULT_ESS_FLOOR,
) -> dict[str, float | str]:
    """Shared fixture run for the planted-tilt and harm benches.

    Base predictor: unconditional absolute-residual score around the
    source train median (the trivially honest base — no learned model, so
    the comparison isolates weighting alone). Feature map ``[u, sign(y)]``
    recovers the paper's true tilt ``(a*, b*)``; the label-block
    coefficient is the bench's misspecification knob. The X-block is held
    at the true ``a*`` so that label-block misspecification is the only
    varied factor.
    """
    data = synthetic_bimodal_shift(
        eta=eta, a_star=a_star, b_star=b_star, n_cal=n_cal, n_test=n_test, seed=seed
    )
    a = _check_alpha(alpha)
    m = float(np.median(data.y_train))
    s_cal = cqr_scores(data.y_cal, np.full(n_cal, m), np.full(n_cal, m))
    feats = np.column_stack([data.x_cal[:, 0], np.sign(data.y_cal)])
    w = tilt_weights(feats, np.array([a_star, theta_label_block]))
    res = extra_conformal_quantile(s_cal, w, a, ess_floor=ess_floor)
    lo_t, hi_t = np.full(n_test, m), np.full(n_test, m)
    tlo, thi = expand_interval(lo_t, hi_t, res.qhat)
    q_u = conformal_quantile(s_cal, a)
    ulo, uhi = expand_interval(lo_t, hi_t, q_u)
    m_t = set_metrics(data.y_test, tlo, thi)
    m_u = set_metrics(data.y_test, ulo, uhi)
    return {
        "synthetic_coverage": m_t.coverage,
        "synthetic_mean_width": m_t.mean_width,
        "synthetic_unweighted_coverage": m_u.coverage,
        "synthetic_unweighted_mean_width": m_u.mean_width,
        "synthetic_coverage_error": abs(m_t.coverage - (1.0 - a)),
        "synthetic_unweighted_coverage_error": abs(m_u.coverage - (1.0 - a)),
        "synthetic_ess_fraction": res.ess_fraction,
        "synthetic_weights_mode": res.weights_mode,
        "synthetic_theta_label_block": float(theta_label_block),
        "synthetic_n": float(m_t.n),
        "synthetic_alpha": float(a),
        "synthetic_dgp": "SYNTHETIC_bimodal_joint_shift",
        "synthetic_claim": "research_metric_only",
        "synthetic_seed": float(seed),
    }


def bench_extra_conformal(
    *,
    eta: float = 1.0,
    alpha: float = 0.10,
    n_cal: int = 1000,
    n_test: int = 1500,
    seed: int = 0,
) -> dict[str, float | str]:
    """Planted-tilt coverage: correct-direction ExTRA weights vs unweighted.

    SYNTHETIC correctness bench on the paper's Section 5.2 DGP at the true
    tilt ``(a*, b*) = (1.0, 1.2)``: the tilt-weighted set must cover at
    least as well as the unweighted set and closer to ``1 - alpha``.
    Proper scores only.
    """
    return _bimodal_cqr_bench(1.2, eta=eta, alpha=alpha, n_cal=n_cal, n_test=n_test, seed=seed)


def bench_extra_harm(
    *,
    eta: float = 1.0,
    alpha: float = 0.10,
    n_cal: int = 1000,
    n_test: int = 1500,
    seed: int = 0,
) -> dict[str, float | str]:
    """Harm regime: flipped label-block tilt vs the unweighted baseline.

    The misspecified direction ``(a*, -b*)`` uses well-posed weights (ESS
    is high — no collapse, no fallback) yet covers WORSE than doing
    nothing: this is the documented failure mode of tilt reweighting
    under label-block misspecification (paper Section 5-6: good weighted
    calibration does not guarantee the tilted direction preserves
    coverage), surfaced as a red-team diagnostic rather than hidden.
    ``harm_coverage_gap`` = unweighted coverage minus tilted coverage;
    positive means the tilt hurt.
    """
    row = _bimodal_cqr_bench(-1.2, eta=eta, alpha=alpha, n_cal=n_cal, n_test=n_test, seed=seed)
    row["synthetic_harm_coverage_gap"] = float(row["synthetic_unweighted_coverage"]) - float(
        row["synthetic_coverage"]
    )
    row["synthetic_harm_mode"] = "misspecified_label_tilt"
    return row
