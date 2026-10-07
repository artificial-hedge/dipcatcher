"""Scale reliability — Cronbach's alpha, KR-20, split-half.

Cronbach (1951) coefficient alpha from item covariances:

    alpha = k/(k-1) (1 - sum_j sigma_j^2 / sigma_total^2)

KR-20 (Kuder & Richardson 1937) is the binary special case
(computed identically via the covariance form). Split-half uses
odd/even part-scores with the Spearman-Brown correction
r = 2 r_half / (1 + r_half) (Brown 1910, Spearman 1910).
Alpha-if-item-deleted flags items that reduce consistency.

Honesty: alpha is a lower bound on reliability under tau-
equivalence — reported as such; the bench uses a planted single-
factor item pool (alpha high) and independent noise items (alpha
low). Fail-closed on single-item or degenerate-total designs.

References: Cronbach (1951) "Coefficient alpha and the internal
structure of tests"; Kuder & Richardson (1937); Spearman (1910) /
Brown (1910) prophecy formula; McDonald (1999) "Test Theory".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_items(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 10 or a.shape[1] < 2:
        raise ValueError("bad item matrix")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    return a


def cronbach_alpha(x: FloatArray) -> dict[str, float]:
    """Cronbach alpha on an (n_persons, k_items) matrix."""
    a = _check_items(x)
    k = a.shape[1]
    item_var = a.var(axis=0, ddof=1)
    total_var = a.sum(axis=1).var(ddof=1)
    if total_var <= 1e-12:
        raise ValueError("zero total variance")
    alpha = float(k / (k - 1) * (1.0 - item_var.sum() / total_var))
    return {"alpha": alpha, "n_items": float(k)}


def alpha_if_deleted(x: FloatArray) -> FloatArray:
    """Alpha computed with each item dropped in turn (k values)."""
    a = _check_items(x)
    k = a.shape[1]
    out = np.empty(k)
    for j in range(k):
        out[j] = cronbach_alpha(np.delete(a, j, axis=1))["alpha"]
    return np.asarray(out, dtype=np.float64)


def kr20(x: FloatArray) -> dict[str, float]:
    """Kuder-Richardson 20 (binary-item alpha)."""
    a = _check_items(x)
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("non-binary items")
    return cronbach_alpha(a)


def split_half(x: FloatArray, seed: int = 0) -> dict[str, float]:
    """Odd/even split-half correlation with Spearman-Brown
    full-test correction."""
    a = _check_items(x)
    k = a.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(k)
    half1 = a[:, order[: k // 2]].sum(axis=1)
    half2 = a[:, order[k // 2 :]].sum(axis=1)
    if half1.std() <= 1e-12 or half2.std() <= 1e-12:
        raise ValueError("degenerate halves")
    r_half = float(np.corrcoef(half1, half2)[0, 1])
    sb = 2.0 * r_half / (1.0 + r_half)
    return {"r_half": r_half, "spearman_brown": float(sb)}


def bench_cronbach(seed: int = 20261231 + 438) -> dict[str, float]:
    """SYNTHETIC check — single-factor items high alpha, noise low."""
    rng = np.random.default_rng(seed)
    n, k = 400, 10
    theta = rng.standard_normal(n)
    loading = np.linspace(0.5, 1.2, k)
    x = theta[:, None] * loading[None, :] + rng.standard_normal((n, k))
    alpha = cronbach_alpha(x)
    alpha_noise = cronbach_alpha(rng.standard_normal((n, k)))
    x_bin = (x > np.median(x, axis=0)).astype(float)
    kr = kr20(x_bin)
    split = split_half(x, seed=seed)
    deleted = alpha_if_deleted(x)
    if alpha["alpha"] < 0.75 or abs(alpha_noise["alpha"]) > 0.2 or split["spearman_brown"] < 0.7:
        raise ValueError(
            f"cronbach off: a={alpha['alpha']:.3f} noise={alpha_noise['alpha']:.3f} "
            f"sb={split['spearman_brown']:.3f}"
        )
    return {
        "synthetic_alpha": alpha["alpha"],
        "synthetic_alpha_noise": alpha_noise["alpha"],
        "synthetic_kr20": kr["alpha"],
        "synthetic_spearman_brown": split["spearman_brown"],
        "synthetic_alpha_if_deleted_min": float(deleted.min()),
        "synthetic_score": 1.0,
    }
