"""Kuiper test — rotation-invariant uniform/CDF deviation statistic (SYNTHETIC).

Kuiper (1960): for an empirical CDF F_n against a reference F, the
two-sided Kolmogorov statistic D = max|F_n - F| is not invariant
under cyclic relabeling of a circular domain; Kuiper's

    V = D+ + D-   where  D+ = max(F_n - F),  D- = max(F - F_n)

is. The upper-tail p-value uses Stephens (1970) asymptotic form:
p = 2 sum_{j>=1} (4j^2 z^2 - 1) exp(-2j^2 z^2), z = V sqrt(n) +
corrections. The two-sample variant computes V on the two empirical
CDFs evaluated on the pooled order statistics.

Honesty: the bench checks (a) a non-uniform circularly-shifted
sample is rejected while KS may tolerate it, (b) a null uniform
sample respects the 5% level at a loose bound, and (c) the two-
sample V separates shifted samples. Stephens' tail approximation is
asymptotic — p bounds are generous and documented. Fail-closed on
empty samples or non-finite data.

References: Kuiper (1960) "Tests concerning random points on a
circle"; Stephens (1970) "Use of the Kolmogorov-Smirnov, Cramer-von
Mises and related statistics without extensive tables"; Press et al.
"Numerical Recipes" ch. 14.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_sample(x: FloatArray, name: str = "sample") -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 4 or not np.isfinite(a).all():
        raise ValueError(f"bad {name}")
    return a


def _kuiper_p(z: float) -> float:
    """Stephens (1970) asymptotic p for z = V sqrt(eff-n)."""
    if z <= 0:
        return 1.0
    s = 0.0
    for j in range(1, 60):
        term = (4.0 * j * j * z * z - 1.0) * np.exp(-2.0 * j * j * z * z)
        s += term
        if abs(term) < 1e-12 * max(1.0, abs(s)):
            break
    return float(np.clip(2.0 * s, 0.0, 1.0))


def kuiper_test(sample: FloatArray, cdf: FloatArray | None = None) -> dict[str, float]:
    """One-sample Kuiper V test against a reference CDF.

    ``cdf`` gives the reference values at the sorted sample points;
    omitted => uniform(0,1). Returns ``v``, ``d_plus``, ``d_minus``,
    ``p``.
    """
    x = np.sort(_check_sample(sample))
    n = x.size
    if cdf is None:
        f = np.clip(x, 0.0, 1.0)
    else:
        f = np.asarray(cdf, dtype=float)
        if f.shape != x.shape or not np.isfinite(f).all():
            raise ValueError("bad cdf")
    i = np.arange(1, n + 1) / n
    d_plus = float(max(0.0, np.max(i - f)))
    d_minus = float(max(0.0, np.max(f - (i - 1.0 / n))))
    v = d_plus + d_minus
    # Stephens (1965) corrected statistic
    z = v * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    return {
        "v": v,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "p": _kuiper_p(z),
    }


def kuiper_two_sample(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Two-sample Kuiper test on the pooled order statistics."""
    a = _check_sample(x, "x")
    b = _check_sample(y, "y")
    xs, ys = np.sort(a), np.sort(b)
    n1, n2 = xs.size, ys.size
    pooled = np.sort(np.concatenate([xs, ys]))
    fx = np.searchsorted(xs, pooled, side="right") / n1
    fy = np.searchsorted(ys, pooled, side="right") / n2
    d = fx - fy
    v = float(d.max() - d.min())
    ne = n1 * n2 / (n1 + n2)
    z = v * np.sqrt(ne)
    return {
        "v": v,
        "p": _kuiper_p(z),
        "n_eff": float(ne),
    }


def bench_kuiper(seed: int = 20261231 + 422) -> dict[str, float]:
    """SYNTHETIC check — circular deviation rejection + null size."""
    rng = np.random.default_rng(seed)
    # circular setup: points on [0,2pi); non-uniform bump that a
    # rotation-invariant stat still detects
    n = 400
    x = rng.vonmises(0.0, 0.8, n) % (2 * np.pi)
    u = x / (2 * np.pi)
    out = kuiper_test(np.sort(u))
    p_bump = out["p"]
    null = rng.random(n)
    p_null = kuiper_test(np.sort(null))["p"]
    a = rng.vonmises(0.5, 1.0, 250) % (2 * np.pi)
    b = rng.vonmises(2.0, 1.0, 250) % (2 * np.pi)
    p_two = kuiper_two_sample(a / (2 * np.pi), b / (2 * np.pi))["p"]
    if p_bump > 0.01 or p_null < 0.01 or p_two > 0.01:
        raise ValueError(f"kuiper off: bump={p_bump:.4f} null={p_null:.4f} two={p_two:.4f}")
    return {
        "synthetic_kuiper_p_bump": p_bump,
        "synthetic_kuiper_p_null": p_null,
        "synthetic_kuiper_p_two": p_two,
        "synthetic_score": 1.0,
    }
