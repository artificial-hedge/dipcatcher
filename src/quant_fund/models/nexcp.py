"""NexCP — nonexchangeable conformal prediction with coverage bounds (SYNTHETIC).

Barber, Candès, Ramdas & Tibshirani (2023, Ann. Statist. 51(2),
doi:10.1214/23-AOS2276, "Conformal prediction beyond exchangeability",
arXiv:2202.13415). The method
is abbreviated **NexCP** in the follow-up literature (Barber & Tibshirani,
2025, "Unifying Different Theories of Conformal Prediction",
arXiv:2504.02292). Citation note: the repo roadmap's "NexCP coverage bounds"
item is *not* the Liu/Wang/Xie line — the construction and the coverage
theorems implemented here are exactly those of arXiv:2202.13415, verified
against the paper.

Construction (split conformal, symmetric algorithm — the paper's Eq. (10)
and (11)):

1. Calibration residuals ``R_i = |y_i - mu(x_i)|``, ``i = 1..n``, from a
   pre-fitted point predictor.
2. Trust weights ``w_i in [0, 1]`` encode *side information* about how close
   each calibration point's distribution is to the test point's: recency
   weights ``w_i = exp(-(n - i)/b)`` for temporal drift, or a covariate
   kernel ``w_i = exp(-||x_i - x_test|| / b)`` for conditional shift (the
   paper's spatial-distance example, Sec. 3.1). Normalized (Eq. (10)):
   ``w~_i = w_i / (sum_j w_j + 1)``, ``w~_{n+1} = 1 / (sum_j w_j + 1)``.
3. The prediction interval is ``mu(x_test) +/- Q`` where ``Q`` is the
   weighted quantile (Eq. (11))
   ``Q = Q_{1-alpha}( sum_i w~_i delta_{R_i} + w~_{n+1} delta_{+inf} )
   = inf{ t : sum_{i: R_i <= t} w~_i >= 1 - alpha }``.
   Uniform weights ``w_i = 1`` recover classical split conformal. If
   ``alpha < w~_{n+1}`` the ``+inf`` atom is needed to reach ``1 - alpha``
   and no finite interval exists; we raise instead of returning ``inf``.

Coverage bounds (the paper's Theorems 2 and 3), with
``gap = sum_i w~_i * d_TV(R(Z), R(Z^i))``:

    1 - alpha - gap  <=  P(y_test in C)  <=  1 - alpha + w~_{n+1} + gap

For independent data points the gap is at most
``2 * sum_i w~_i * d_TV((x_i, y_i), (x_test, y_test))`` (Lemma 1 + Sec. 4.1),
so points whose distributions track the test point (small TV, or small
weight) barely erode coverage. The TV distances are population quantities
and cannot be estimated from the data without assumptions, so
``coverage_bounds`` takes caller-supplied values in ``[0, 1]`` or, with
``tv_distances=None``, the worst case ``d_TV = 1``.

Only the symmetric-algorithm split version is implemented: the full-conformal
variant (Eq. (12), (18)) additionally requires refitting the learner on every
hypothesized ``y`` plus a random tag-swap ``K ~ sum_i w~_i delta_i``, which is
out of scope for an online harness. The Theorem 2/3 bounds apply to the split
version as the paper's special case.

Fail-closed: empty/degenerate calibration sets, invalid weights, unattainable
levels (``alpha < w~_{n+1}``), out-of-range TV inputs all raise ``ValueError``;
querying an uncalibrated instance raises ``RuntimeError``. Honesty: only
coverage and interval-width diagnostics are exposed (AGENTS.md contract); no
live-trading or market-evidence claims. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "NexCPSplit",
    "kernel_weights",
    "normalize_weights",
    "recency_weights",
    "weighted_conformal_quantile",
]

# Tolerance for the weighted-quantile CDF comparison; keeps floating-point
# cumsum noise from shifting the recovered order statistic (matches the
# ceil((n+1)(1-alpha)) rule of quant_fund.metrics.conformal.conformal_quantile
# when all weights are uniform).
_CDF_TOL = 1e-12


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def normalize_weights(weights: Array) -> tuple[Array, float]:
    """Normalized NexCP weights (Barber et al., 2023, Eq. (10)).

    Parameters
    ----------
    weights:
        Unnormalized trust weights ``w_1..w_n >= 0``, finite, with positive
        total mass.

    Returns
    -------
    tuple
        ``(w_tilde, w_tilde_test)`` with ``w_tilde[i] = w_i / (sum w + 1)``
        for the calibration points and ``w_tilde_test = 1 / (sum w + 1)`` for
        the test point. The full vector sums to 1.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    denom = total + 1.0
    return w / denom, 1.0 / denom


def recency_weights(n: int, bandwidth: float) -> Array:
    """Exponential recency weights ``w_i = exp(-(n - i) / bandwidth)``, i = 1..n.

    The most recent calibration point gets weight 1; older points decay with
    half-life ``bandwidth * log(2)``. This is the paper's drift example
    (Sec. 3.1: ``w_1 <= ... <= w_n`` so the interval leans on recent data).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    b = float(bandwidth)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("bandwidth must be positive and finite")
    age = np.arange(n - 1, -1, -1, dtype=float)
    return np.asarray(np.exp(-age / b), dtype=float)


def kernel_weights(X_cal: Array, x_test: Array, bandwidth: float) -> Array:
    """Covariate-kernel trust weights ``w_i = exp(-||x_i - x_test|| / bandwidth)``.

    Side-information weighting for conditional shift (the paper's spatial
    example, Sec. 3.1: the weight is a function of the distance to the test
    location). ``X_cal`` is ``(n, p)`` calibration covariates, ``x_test`` is
    the ``(p,)`` test covariate vector.
    """
    X = np.asarray(X_cal, dtype=float)
    xt = np.asarray(x_test, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X_cal must be 2-D (n, p)")
    if X.shape[0] == 0:
        raise ValueError("X_cal must be non-empty")
    if xt.shape[0] != X.shape[1]:
        raise ValueError("x_test dimension mismatch with X_cal")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(xt))):
        raise ValueError("X_cal and x_test must be finite")
    b = float(bandwidth)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("bandwidth must be positive and finite")
    dist = np.linalg.norm(X - xt[None, :], axis=1)
    return np.asarray(np.exp(-dist / b), dtype=float)


def weighted_conformal_quantile(values: Array, weights: Array, alpha: float) -> float:
    """Weighted conformal quantile Q_{1-alpha} of Eq. (11).

    ``Q = inf{ t : sum_{i: R_i <= t} w~_i >= 1 - alpha }`` over the measure
    ``sum_i w~_i delta_{R_i} + w~_{n+1} delta_{+inf}`` with the normalized
    weights of Eq. (10). Raises ``ValueError`` when the finite atoms cannot
    reach mass ``1 - alpha`` (``alpha < w~_{n+1}``), rather than returning
    ``inf`` — the calibration set is too small / too concentrated for the
    requested level.
    """
    r = np.asarray(values, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(r)):
        raise ValueError("values must be finite")
    a = _check_alpha(alpha)
    if r.size != np.asarray(weights).ravel().size:
        raise ValueError("values and weights length mismatch")
    w_tilde, w_test = normalize_weights(weights)
    if a < w_test:
        raise ValueError(
            f"alpha={a:g} < w~_n+1={w_test:g}: the +inf atom is needed to reach "
            "1 - alpha; no finite weighted conformal quantile exists"
        )
    order = np.argsort(r, kind="stable")
    sorted_r = r[order]
    cum = np.cumsum(w_tilde[order])
    idx = int(np.searchsorted(cum, 1.0 - a - _CDF_TOL, side="left"))
    idx = min(idx, sorted_r.size - 1)
    return float(sorted_r[idx])


@dataclass(frozen=True)
class NexCPBounds:
    """Theorem 2 / Theorem 3 coverage bounds for one NexCP predictor."""

    lower: float
    upper: float
    gap: float
    w_test: float


class NexCPSplit:
    """Nonexchangeable split conformal predictor (Barber et al., 2023, Eq. (11)).

    Parameters
    ----------
    alpha:
        Target miscoverage in (0, 1).

    Notes
    -----
    ``calibrate(residuals, weights)`` stores the calibration residuals
    ``R_i = |y_i - mu(x_i)|`` and the trust weights ``w_i`` (use
    ``recency_weights`` / ``kernel_weights`` to build them from side
    information). ``predict_interval(point)`` centers the weighted-quantile
    band on caller-supplied point predictions ``mu(x_test)``.
    ``coverage_bounds(tv_distances)`` evaluates the Theorem 2/3 interval
    ``[1 - alpha - gap, 1 - alpha + w~_{n+1} + gap]`` with
    ``gap = sum_i w~_i * tv_i``, clipped to ``[0, 1]``.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = _check_alpha(alpha)
        self._residuals: Array = np.empty(0)
        self._w_tilde: Array = np.empty(0)
        self._w_test = 0.0
        self._quantile: float | None = None

    def calibrate(self, residuals: Array, weights: Array) -> NexCPSplit:
        """Store calibration residuals ``R_i >= 0`` and trust weights ``w_i``."""
        r = np.asarray(residuals, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("residuals must be non-empty")
        if r.size != w.size:
            raise ValueError("residuals and weights length mismatch")
        if not np.all(np.isfinite(r)):
            raise ValueError("residuals must be finite")
        if np.any(r < 0.0):
            raise ValueError("residuals must be non-negative (absolute-error scores)")
        w_tilde, w_test = normalize_weights(w)
        if self.alpha < w_test:
            raise ValueError(
                f"alpha={self.alpha:g} < w~_n+1={w_test:g}: effective sample size "
                f"{1.0 / w_test:g} is too small for the requested level; add "
                "calibration points or raise the weight mass"
            )
        self._residuals = r
        self._w_tilde = w_tilde
        self._w_test = w_test
        self._quantile = weighted_conformal_quantile(r, w, self.alpha)
        return self

    def _calibrated_quantile(self) -> float:
        q = self._quantile
        if q is None:
            raise RuntimeError("NexCPSplit is not calibrated")
        return float(q)

    @property
    def quantile_(self) -> float:
        """Weighted conformal radius Q_{1-alpha} of Eq. (11)."""
        return self._calibrated_quantile()

    @property
    def weights_normalized_(self) -> Array:
        """Normalized weights ``[w~_1..w~_n, w~_{n+1}]`` summing to 1."""
        self._calibrated_quantile()
        return np.concatenate([self._w_tilde, [self._w_test]])

    @property
    def effective_sample_size_(self) -> float:
        """``1 / w~_{n+1} = sum_i w_i + 1``, the paper's notion of ESS (Sec. 4.2)."""
        self._calibrated_quantile()
        return 1.0 / self._w_test

    def predict_interval(self, point: Array) -> tuple[Array, Array]:
        """Return ``(lower, upper)`` = ``point -/+ Q`` for point predictions."""
        q = self._calibrated_quantile()
        p = np.asarray(point, dtype=float).ravel()
        if p.size == 0:
            raise ValueError("point must be non-empty")
        if not np.all(np.isfinite(p)):
            raise ValueError("point must be finite")
        return p - q, p + q

    def coverage_bounds(self, tv_distances: Array | None = None) -> NexCPBounds:
        """Theorem 2 (lower) and Theorem 3 (upper) coverage bounds.

        Parameters
        ----------
        tv_distances:
            Optional per-calibration-point total-variation distances
            ``d_TV(R(Z), R(Z^i))`` in ``[0, 1]``, shape ``(n,)``. ``None``
            uses the worst case ``d_TV = 1`` for every point.

        Returns
        -------
        NexCPBounds
            ``lower = max(0, 1 - alpha - gap)``,
            ``upper = min(1, 1 - alpha + w~_{n+1} + gap)`` with
            ``gap = sum_i w~_i * tv_i``. The Theorem 3 bound is strict
            (``<``) when residuals are distinct with probability 1; the
            returned value is its right-hand side.
        """
        self._calibrated_quantile()
        n = self._w_tilde.size
        if tv_distances is None:
            tv = np.ones(n, dtype=float)
        else:
            tv = np.asarray(tv_distances, dtype=float).ravel()
            if tv.size != n:
                raise ValueError("tv_distances must have one entry per calibration point")
            if not np.all(np.isfinite(tv)):
                raise ValueError("tv_distances must be finite")
            if np.any(tv < 0.0) or np.any(tv > 1.0):
                raise ValueError("tv_distances must lie in [0, 1]")
        gap = float(np.dot(self._w_tilde, tv))
        lower = max(0.0, 1.0 - self.alpha - gap)
        upper = min(1.0, 1.0 - self.alpha + self._w_test + gap)
        return NexCPBounds(lower=lower, upper=upper, gap=gap, w_test=self._w_test)
