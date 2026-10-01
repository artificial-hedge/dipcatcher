"""Mantel test — matrix-correlation via row-permutation.

Mantel (1967): the correlation between two symmetric distance
matrices A and B cannot be tested elementwise (the n(n-1)/2 entries
are dependent), so significance comes from permuting the row/column
labels of one matrix and recomputing:

    r = <a, b> / (||a|| ||b||)     (lower-triangle entries)
    p = (1 + #{perm r >= r_obs}) / (n_perm + 1)

The partial Mantel test controls for a third distance matrix C via
the first-order partial correlation of (A~C, B~C).

Honesty: the bench builds B as a noisy transform of A (reject) and
an independent C (respects size), using a seeded permutation count
documented in the result. Empirical p granularity is 1/(n_perm+1) —
bounds match. Fail-closed on non-square or asymmetric matrices.

References: Mantel (1967) "The detection of disease clustering and
a generalized regression approach"; Smouse, Long, Sokal (1986)
"Multiple regression and correlation extensions of the Mantel
test"; Legendre & Legendre (2012) "Numerical Ecology".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_dist(m: FloatArray, name: str = "matrix") -> FloatArray:
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 4:
        raise ValueError(f"bad {name}")
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric finite")
    return a


def _lt(m: FloatArray) -> FloatArray:
    i, j = np.tril_indices(m.shape[0], k=-1)
    return np.asarray(m[i, j], dtype=np.float64)


def _corr(a: FloatArray, b: FloatArray) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den <= 1e-300:
        return 0.0
    return float(aa @ bb / den)


def mantel_test(a: FloatArray, b: FloatArray, n_perm: int = 499, seed: int = 0) -> dict[str, float]:
    """Mantel r correlation between distance matrices with
    permutation p-value (one-sided, positive association)."""
    ma = _check_dist(a, "a")
    mb = _check_dist(b, "b")
    if ma.shape != mb.shape:
        raise ValueError("shape mismatch")
    va, vb = _lt(ma), _lt(mb)
    r_obs = _corr(va, vb)
    rng = np.random.default_rng(seed)
    n = ma.shape[0]
    cnt = 0
    for _ in range(max(1, n_perm)):
        idx = rng.permutation(n)
        rp = _corr(va, _lt(mb[np.ix_(idx, idx)]))
        if rp >= r_obs - 1e-12:
            cnt += 1
    p = (1.0 + cnt) / (n_perm + 1.0)
    return {"r": r_obs, "p": p, "n_perm": float(n_perm)}


def mantel_partial(
    a: FloatArray, b: FloatArray, c: FloatArray, n_perm: int = 499, seed: int = 0
) -> dict[str, float]:
    """Partial Mantel correlation of (A,B) controlling C."""
    ma = _check_dist(a, "a")
    mb = _check_dist(b, "b")
    mc = _check_dist(c, "c")
    if not (ma.shape == mb.shape == mc.shape):
        raise ValueError("shape mismatch")
    va, vb, vc = _lt(ma), _lt(mb), _lt(mc)
    r_ab = _corr(va, vb)
    r_ac = _corr(va, vc)
    r_bc = _corr(vb, vc)
    den = np.sqrt(max(1e-300, (1 - r_ac**2) * (1 - r_bc**2)))
    r_obs = (r_ab - r_ac * r_bc) / den
    rng = np.random.default_rng(seed)
    n = ma.shape[0]
    cnt = 0
    for _ in range(max(1, n_perm)):
        idx = rng.permutation(n)
        vbp = _lt(mb[np.ix_(idx, idx)])
        r_abp = _corr(va, vbp)
        r_bcp = _corr(vbp, vc)
        rp = (r_abp - r_ac * r_bcp) / den
        if rp >= r_obs - 1e-12:
            cnt += 1
    p = (1.0 + cnt) / (n_perm + 1.0)
    return {"r_partial": r_obs, "p": p, "r_marginal": r_ab}


def bench_mantel(seed: int = 20261231 + 427) -> dict[str, float]:
    """SYNTHETIC check — dependent matrices rejected, null respected."""
    rng = np.random.default_rng(seed)
    n = 30
    pts = rng.random((n, 2)) * 10
    d_a = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    # B = noisy version of A
    d_b = d_a + 0.2 * rng.standard_normal((n, n))
    d_b = (d_b + d_b.T) / 2.0
    np.fill_diagonal(d_b, 0.0)
    out_dep = mantel_test(d_a, d_b, n_perm=499, seed=seed)
    # independent C
    pts2 = rng.random((n, 2)) * 10
    d_c = np.linalg.norm(pts2[:, None, :] - pts2[None, :, :], axis=2)
    out_ind = mantel_test(d_a, d_c, n_perm=499, seed=seed + 1)
    if out_dep["p"] > 0.01 or out_ind["p"] < 0.005:
        raise ValueError(f"mantel off: dep={out_dep['p']:.4f} ind={out_ind['p']:.4f}")
    return {
        "synthetic_mantel_r_dep": out_dep["r"],
        "synthetic_mantel_p_dep": out_dep["p"],
        "synthetic_mantel_p_ind": out_ind["p"],
        "score": 1.0,
    }
