"""Plug-in Shannon entropy tools for bar-return diagnostics.

Provides binned plug-in entropy (normalised and in nats), a rolling version,
and inverse-entropy weighting. Used by the bar diagnostics to ask whether a
bar construction makes bar returns closer to i.i.d.-looking (less serially
predictable) than time bars — a distributional property, not a performance
claim.

Honesty: entropy estimates are descriptive statistics of the supplied
series; no predictive or market claim is attached.

References:
- Shannon, C. E. (1948). A mathematical theory of communication — entropy.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — entropy as an information-arrival proxy.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def shannon_entropy(x: FloatArray, bins: int = 10) -> dict[str, float]:
    """Plug-in Shannon entropy of ``x`` discretised into ``bins`` bins.

    Returns ``{"nats": H, "normalised": H / ln(bins), "bins": bins}``.
    Normalised entropy is in (0, 1]; a constant series gives 0 and a
    perfectly uniform binning gives ≈ 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    counts, _ = np.histogram(x, bins=bins)
    total = float(counts.sum())
    p = counts[counts > 0].astype(np.float64) / total
    h = float(-np.sum(p * np.log(p)))
    return {"nats": h, "normalised": float(h / np.log(bins)), "bins": float(bins)}


def rolling_shannon_entropy(x: FloatArray, window: int, bins: int = 10) -> FloatArray:
    """Rolling plug-in entropy; NaN until the first full window."""
    x = np.asarray(x, dtype=np.float64)
    if window < 2:
        raise ValueError("window must be >= 2")
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(window - 1, n):
        out[t] = shannon_entropy(x[t - window + 1 : t + 1], bins=bins)["nats"]
    return out


def entropy_weights(avg_uniqueness: FloatArray) -> FloatArray:
    """Weights ∝ uniqueness with a small entropy-style softmax temperature.

    Kept deliberately simple: w_i ∝ u_i² / Σ u_j², so low-uniqueness
    samples are quadratically downweighted. Output sums to 1.
    """
    u = np.asarray(avg_uniqueness, dtype=np.float64)
    if u.ndim != 1 or len(u) == 0:
        raise ValueError("avg_uniqueness must be a non-empty one-dimensional array")
    if np.any(u < 0):
        raise ValueError("uniqueness must be non-negative")
    sq = u * u
    total = float(sq.sum())
    if total <= 0:
        return np.full(len(u), 1.0 / len(u), dtype=np.float64)
    return np.asarray(sq / total, dtype=np.float64)
