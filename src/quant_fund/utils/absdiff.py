"""One-dimensional pairwise absolute sums.

The double sum of absolute deviations on the line is an order-statistic
identity, not a different estimator and not an approximation. With
:math:`x_{(1)} \\le \\cdots \\le x_{(n)}` the order statistics (1-based):

.. math::

    \\sum_{i=1}^{n}\\sum_{j=1}^{n}|x_i-x_j|
    = 2\\sum_{i=1}^{n}(2i-n-1)\\,x_{(i)}.

The diagonal is included and contributes nothing. Proof: the strict upper
triangle is :math:`\\sum_{i<j}(x_{(j)}-x_{(i)}) = \\sum_k(2k-n-1)x_{(k)}`,
and the double sum is twice that triangle.

The cross sum, after both samples are sorted, is one prefix-sum pass. For
a fixed :math:`x` and sorted :math:`y_1\\le\\cdots\\le y_m`,

.. math::

    \\sum_{j=1}^{m}|x-y_j|
    = (2c-m)\\,x + \\sum_j y_j - 2\\sum_{y_j<x} y_j,

where :math:`c=\\#\\{y_j<x\\}`. A tie contributes zero, so the side on
which a tie is counted does not change the sum.

These are the sums inside ensemble CRPS and the Székely–Rizzo energy
distance. They are not a pairwise median (Qn, kNN conformal denominators):
that multiset does not collapse to a dot product, and this module must not
be used as a substitute for it.

Fail-closed: an empty or non-finite vector raises ``ValueError``. The
identities are exact in float64 for integer data whose weighted sum fits
in the mantissa; on general floats they are the same mathematical sum with
a different addition order than an :math:`n\\times n` broadcast. Research
arithmetic only — not a live-capital claim.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite_1d(name: str, values: Array) -> Array:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size == 0 or not np.isfinite(series).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return series


def _sum_pairwise_abs_sorted(ordered: Array) -> float:
    """Double-sum identity. ``ordered`` is sorted, finite, and nonempty."""
    n = ordered.shape[0]
    # 1-based index i has coefficient 2*i - n - 1.
    weights = 2.0 * np.arange(1, n + 1, dtype=np.float64) - float(n + 1)
    return float(2.0 * np.dot(weights, ordered))


def sum_pairwise_abs(values: Array) -> float:
    """:math:`\\sum_i\\sum_j|x_i-x_j|`, diagonal included."""
    ordered = np.sort(_finite_1d("values", values))
    return _sum_pairwise_abs_sorted(ordered)


def sum_cross_abs(left: Array, right: Array) -> float:
    """:math:`\\sum_i\\sum_j|x_i-y_j|`."""
    xs = np.sort(_finite_1d("left", left))
    ys = np.sort(_finite_1d("right", right))
    prefix = np.empty(ys.shape[0] + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(ys, out=prefix[1:])
    total = float(prefix[-1])
    m = ys.shape[0]
    count_below = np.searchsorted(ys, xs, side="left")
    sum_below = prefix[count_below]
    per_left = (2.0 * count_below - m) * xs + total - 2.0 * sum_below
    return float(per_left.sum())


def mean_pairwise_abs(values: Array) -> float:
    """:math:`n^{-2}\\sum_i\\sum_j|x_i-x_j|`."""
    series = _finite_1d("values", values)
    n = series.shape[0]
    return _sum_pairwise_abs_sorted(np.sort(series)) / float(n * n)


def mean_cross_abs(left: Array, right: Array) -> float:
    """:math:`(n m)^{-1}\\sum_i\\sum_j|x_i-y_j|`."""
    xs = _finite_1d("left", left)
    ys = _finite_1d("right", right)
    return sum_cross_abs(xs, ys) / float(xs.shape[0] * ys.shape[0])


def ensemble_crps(y: float, sample: Array) -> float:
    """Ensemble CRPS :math:`E|X-y| - \\tfrac12 E|X-X'|` for one observation.

    ``sample`` must be a nonempty finite vector. ``y`` must be finite.
    The second term is the pairwise-absolute identity above, including the
    zero diagonal, divided by :math:`2n^2`.
    """
    obs = float(y)
    if not math.isfinite(obs):
        raise ValueError("y must be finite")
    ordered = np.sort(_finite_1d("sample", sample))
    n = ordered.shape[0]
    term1 = float(np.mean(np.abs(ordered - obs)))
    term2 = _sum_pairwise_abs_sorted(ordered) / (2.0 * float(n * n))
    return term1 - term2


__all__ = [
    "ensemble_crps",
    "mean_cross_abs",
    "mean_pairwise_abs",
    "sum_cross_abs",
    "sum_pairwise_abs",
]
