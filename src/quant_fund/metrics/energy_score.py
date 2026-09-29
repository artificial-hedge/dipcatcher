"""Multivariate energy score: strictly proper scoring rule for ensembles.

The energy score is the energy-distance scoring rule for multivariate
predictive distributions (Gneiting & Raftery 2007, JASA 102(477), 359–378,
doi:10.1198/016214506000001437): for an ensemble x_1..x_n in R^d and an
observation y,

    ES = (1/n) Σ_i ||x_i − y||₂  −  (1/(2n(n−1))) Σ_{i≠j} ||x_i − x_j||₂,

the unbiased U-statistic plug-in of E_P||X − y|| − (1/2) E_{P⊗P}||X − X'||.
Strict propriety follows from the negative definiteness of the Euclidean
distance kernel (Székely 2003, "Statistics on the energy distance",
InterStat; Székely & Rizzo 2013, J. Statist. Plann. Inference 143(8),
1249–1272, arXiv:1210.3927): ES is minimized in expectation iff the forecast
distribution equals the data-generating distribution.

``threshold_energy_score`` implements the threshold-weighted variant in the
spirit of Gneiting & Ranjan (2013, "Combining predictive distributions",
Electron. J. Stat. 7, 1747–1782, doi:10.1214/13-EJS823) — a multiplicative
kernel centered at the observation applied to both terms of the score.

**Propriety caveat (measured, not theoretical hand-waving).** That weighting
is *not* a kernel reweighting in the Gneiting & Raftery (2007, §4) sense: term
1 receives ``w`` once (linear) while term 2 receives ``w`` twice (quadratic),
so the two terms scale at different rates. The expected score behaves like

    E[ES_w] ≈ sigma * a * (w - (sqrt(2)/2) * w**2),   a = E||Z||,

and the leading coefficient vanishes at **w = sqrt(2)** — exactly, and
dimension-free (``a`` cancels). For ``weight < sqrt(2)`` the score is bounded
below and has an interior minimum at the truth; for ``weight >= sqrt(2)`` it is
**improper** and diverges to ``-inf`` as forecast dispersion grows, i.e. it
rewards unbounded variance inflation. ``threshold_energy_score`` therefore
emits a ``RuntimeWarning`` for ``weight >= sqrt(2)`` and its arithmetic is left
untouched so that sealed receipts stay reproducible (AGENTS.md honesty contract
rule #4). Use ``quant_fund.metrics.scoring.crps_threshold_weighted`` for a
proper threshold-weighted score. The exact weighting implemented here
(documented, cf. the Matheson & Winkler 1976, Manag. Sci. 22(10), 1087–1096
weighting idea):

    w(r) = weight   if r >  threshold   (tail error, amplified)
    w(r) = 1.0      if r <= threshold   (central error, full weight)

    ES_w = (1/n) Σ_i w(d_i) d_i
           − (1/(2n(n−1))) Σ_{i≠j} w(d_i) w(d_j) ||x_i − x_j||₂,
    d_i = ||x_i − y||₂,  weight >= 1.

Tail errors are amplified by ``weight`` while central misses keep full
weight, so the threshold-weighted score emphasizes tail errors. ``weight =
1.0`` (default) recovers the plain energy score. Pairwise distances use
``scipy.spatial.distance.cdist``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

__all__ = [
    "EnergyScoreCurve",
    "THRESHOLD_WEIGHT_PROPRIETY_BOUND",
    "energy_score",
    "energy_score_curve",
    "threshold_energy_score",
]

Array = NDArray[np.float64]

#: Exact, dimension-free weight at which ``threshold_energy_score`` stops being
#: a proper scoring rule: term 1 is linear in the weight while term 2 is
#: quadratic, so ``E[ES_w] ~ sigma * a * (w - w**2 / sqrt(2))`` and the leading
#: coefficient vanishes at ``w = sqrt(2)``. Above it the expectation diverges to
#: ``-inf`` as forecast dispersion grows (variance inflation is rewarded). The
#: value is the *boundary*, not a safe setting: measured drift still turns
#: negative at ``w = sqrt(2)`` for large ``sigma``. Use ``weight <
#: THRESHOLD_WEIGHT_PROPRIETY_BOUND`` for a bounded-below score, or
#: ``quant_fund.metrics.scoring.crps_threshold_weighted`` (proper at any
#: weight) when tail emphasis is needed.
THRESHOLD_WEIGHT_PROPRIETY_BOUND: float = math.sqrt(2.0)


@dataclass(frozen=True)
class EnergyScoreCurve:
    """Threshold-weighted energy score evaluated on a threshold grid.

    ``scores[k]`` is ``threshold_energy_score(ensemble, observation,
    thresholds[k], weight=weight)``; with ``weight = 1.0`` the curve is flat
    at the plain energy score (the kernel is identically 1).
    """

    thresholds: Array
    scores: Array
    weight: float

    def as_dict(self) -> dict[str, Array | float]:
        return {
            "thresholds": self.thresholds,
            "scores": self.scores,
            "weight": self.weight,
        }


def _validate_inputs(ensemble: Array, observation: Array) -> tuple[Array, Array]:
    ens = np.asarray(ensemble, dtype=float)
    obs = np.asarray(observation, dtype=float)
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be 2-D (n_samples, dim), got ndim={ens.ndim}")
    n, dim = ens.shape
    if n < 2:
        raise ValueError(f"ensemble needs n_samples >= 2, got {n}")
    if dim < 1:
        raise ValueError("ensemble must have dim >= 1")
    if obs.ndim != 1:
        raise ValueError(f"observation must be 1-D (dim,), got ndim={obs.ndim}")
    if obs.shape[0] != dim:
        raise ValueError(f"dim mismatch: ensemble dim={dim}, observation dim={obs.shape[0]}")
    if not np.all(np.isfinite(ens)):
        raise ValueError("ensemble contains non-finite values")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observation contains non-finite values")
    return ens, obs


def _pairwise_terms(ens: Array, obs: Array) -> tuple[Array, Array]:
    """d_i = ||x_i − y||₂ (n,) and D_{ij} = ||x_i − x_j||₂ (n, n)."""
    d_obs = cdist(ens, obs.reshape(1, -1), metric="euclidean").reshape(-1)
    d_ens = cdist(ens, ens, metric="euclidean")
    return d_obs, d_ens


def energy_score(ensemble: Array, observation: Array) -> float:
    """Unbiased energy score of an ensemble against an observation.

    ES = (1/n) Σ_i ||x_i − y||₂ − (1/(2n(n−1))) Σ_{i≠j} ||x_i − x_j||₂,
    with both distance matrices from ``scipy.spatial.distance.cdist``.
    Strictly proper w.r.t. distributions with finite first moment
    (Gneiting & Raftery 2007; Székely 2003). The U-statistic can dip
    slightly below 0 for a perfectly calibrated finite ensemble; that is
    sampling noise, not a bug.
    """
    ens, obs = _validate_inputs(ensemble, observation)
    d_obs, d_ens = _pairwise_terms(ens, obs)
    n = int(ens.shape[0])
    term1 = float(np.mean(d_obs))
    term2 = float(np.sum(d_ens)) / float(2 * n * (n - 1))
    return term1 - term2


def _validate_weight(weight: float, *, warn: bool = True) -> float:
    w = float(weight)
    if not np.isfinite(w) or w < 1.0:
        raise ValueError(f"weight must be >= 1, got {weight}")
    if warn and w >= THRESHOLD_WEIGHT_PROPRIETY_BOUND:
        warnings.warn(
            f"threshold_energy_score is an IMPROPER scoring rule at weight={w!r} >= "
            f"sqrt(2)={THRESHOLD_WEIGHT_PROPRIETY_BOUND!r}: term 1 is linear in the "
            "weight while term 2 is quadratic, so E[ES_w] ~ sigma*a*(w - w^2/sqrt(2)) "
            "diverges to -inf as forecast dispersion grows (unbounded variance "
            "inflation is rewarded). The arithmetic is unchanged so sealed receipts "
            "stay reproducible; use weight < sqrt(2), or "
            "quant_fund.metrics.scoring.crps_threshold_weighted for a proper "
            "threshold-weighted score.",
            RuntimeWarning,
            stacklevel=3,
        )
    return w


def _validate_threshold(threshold: float) -> float:
    t = float(threshold)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError(f"threshold must be finite and >= 0, got {threshold}")
    return t


def threshold_energy_score(
    ensemble: Array,
    observation: Array,
    threshold: float,
    weight: float = 1.0,
) -> float:
    """Threshold-weighted energy score emphasizing tail errors.

    Weighting kernel (Gneiting & Ranjan 2013, multiplicative kernel centered
    at the observation; Matheson & Winkler 1976 weighting idea):

        w(r) = weight  if r >  threshold,  else 1.0,   weight >= 1.

    ES_w = (1/n) Σ_i w(d_i) d_i
           − (1/(2n(n−1))) Σ_{i≠j} w(d_i) w(d_j) ||x_i − x_j||₂,
    d_i = ||x_i − y||₂. Tail errors (d_i > threshold) are amplified by
    ``weight``; central misses (d_i <= threshold) keep full weight.
    ``weight = 1.0`` recovers ``energy_score`` exactly.

    .. warning::
       **Domain of propriety: ``weight < sqrt(2)`` ONLY.** This is *not* a
       kernel-score reweighting: the product kernel ``w(d_i) w(d_j)`` makes
       term 2 quadratic in the weight while term 1 stays linear, so

           E[ES_w] ≈ sigma · a · (w − w²/√2),   a = E||Z||,

       and the leading coefficient vanishes at ``w* = sqrt(2)`` — exactly, and
       dimension-free (``a`` cancels; verified for d = 1..16). Consequences:

       * ``weight < sqrt(2)`` — bounded below, interior minimum at the truth
         (still not *strictly* proper: the weighting collapses information
         about the tails).
       * ``weight >= sqrt(2)`` — **improper**. The expectation is monotone
         decreasing in forecast dispersion and diverges to ``-inf``, so a
         forecaster that inflates its variance without limit is rewarded
         without limit. The score can legitimately come out *negative*, which
         a proper energy score in the population cannot. ``sqrt(2)`` is the
         boundary, not a safe value: at ``w = sqrt(2)`` the residual
         finite-threshold correction still drifts negative as sigma grows
         (measured +0.160 at sigma=8 → −0.0012 at sigma=128, d=2, 60 members).

       A ``RuntimeWarning`` is emitted for ``weight >= sqrt(2)`` but the
       arithmetic is deliberately unchanged: existing tests pin it and sealed
       receipts may reference it, so altering the numbers would invalidate
       immutable evidence (AGENTS.md honesty contract rule #4). Do not
       headline this score at ``weight >= sqrt(2)``; use
       ``quant_fund.metrics.scoring.crps_threshold_weighted`` (proper at any
       weight, 1-D) instead.
    """
    ens, obs = _validate_inputs(ensemble, observation)
    t = _validate_threshold(threshold)
    w = _validate_weight(weight)
    d_obs, d_ens = _pairwise_terms(ens, obs)
    n = int(ens.shape[0])
    w_obs = np.where(d_obs > t, w, 1.0)
    term1 = float(np.mean(d_obs * w_obs))
    term2 = float(np.sum(d_ens * np.outer(w_obs, w_obs))) / float(2 * n * (n - 1))
    return term1 - term2


def energy_score_curve(
    ensemble: Array,
    observation: Array,
    thresholds: Array,
    *,
    weight: float = 1.0,
) -> EnergyScoreCurve:
    """Threshold-weighted energy score across a threshold grid.

    ``scores[k] = threshold_energy_score(ensemble, observation,
    thresholds[k], weight=weight)``. Thresholds must be 1-D, non-empty, all
    finite and >= 0. Keyword-only ``weight`` (default 1.0) matches the plain
    energy score on every grid point.
    """
    thr = np.asarray(thresholds, dtype=float).reshape(-1)
    if thr.size == 0:
        raise ValueError("thresholds must be non-empty")
    if not np.all(np.isfinite(thr)):
        raise ValueError("thresholds contains non-finite values")
    if np.any(thr < 0.0):
        raise ValueError("thresholds must be >= 0")
    w = _validate_weight(weight)
    scores = np.array(
        [threshold_energy_score(ensemble, observation, float(t), weight=w) for t in thr],
        dtype=float,
    )
    return EnergyScoreCurve(thresholds=thr, scores=scores, weight=w)
