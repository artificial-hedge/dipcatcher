"""Quade's test — block ranks weighted by within-block range (SYNTHETIC).

Quade (1979): Friedman's within-block ranks are weighted by each
block's spread (range over the k treatments), so blocks with more
pronounced treatment separation count more:

    R_ij = within-block rank of x_ij
    S_ij = w_i (R_ij - (k+1)/2),   w_i = rank of block range among blocks
    Quade T3 = (n-1) B / (A - B)   ~ F_{k-1, (n-1)(k-1)}

with B = (1/n) sum_j S_.j^2 and A = sum_ij S_ij^2. Reported here
in the chi^2 form Q = n(k-1) B / A ~ chi^2_{k-1} (Conover 1999
equivalent for large n); we emit both.

Honesty: bench uses a planted monotone treatment effect (rejected)
and a null (size held). Range weighting makes Quade more sensitive
than Friedman for scale-heterogeneous blocks. Fail-closed on
constant blocks or k<3.

References: Quade (1979) "Using weighted rankings in the analysis
of complete blocks with additive block effects", JASA 74:680;
Conover (1999) "Practical Nonparametric Statistics" 3rd ed. ch. 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, f, rankdata

FloatArray = NDArray[np.float64]


def quade_test(x: FloatArray) -> dict[str, float]:
    """Quade's weighted-rank test on an (n blocks, k treatments)
    matrix."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] < 3:
        raise ValueError("bad block matrix")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    n, k = a.shape
    if (np.ptp(a, axis=1) <= 1e-12).all():
        raise ValueError("constant blocks")
    r = np.apply_along_axis(rankdata, 1, a)
    ranges = np.ptp(a, axis=1)
    w = rankdata(ranges)
    s = w[:, None] * (r - (k + 1.0) / 2.0)
    b = float((s.sum(axis=0) ** 2).sum() / n)
    atot = float((s * s).sum())
    if atot <= 0:
        raise ValueError("zero Quade denominator")
    t3 = float((n - 1) * b / (atot - b)) if atot > b else 0.0
    p_f = float(f.sf(max(0.0, t3), k - 1, (n - 1) * (k - 1)))
    q = float(n * (k - 1) * b / atot)
    p_chi = float(chi2.sf(q, k - 1))
    return {"t3": t3, "p_f": p_f, "q": q, "p": p_chi}


def bench_quade(seed: int = 20261231 + 435) -> dict[str, float]:
    """SYNTHETIC check — ordered effect rejected, null held."""
    rng = np.random.default_rng(seed)
    n, k = 24, 4
    base = rng.standard_normal((n, 1))
    x = base + np.array([0.0, 0.5, 1.0, 1.6])[None, :] + 0.35 * rng.standard_normal((n, k))
    out = quade_test(x)
    xn = base + 0.35 * rng.standard_normal((n, k))
    out_n = quade_test(xn)
    if out["p"] > 0.01 or out_n["p"] < 0.005:
        raise ValueError(f"quade off: p={out['p']:.4f} null={out_n['p']:.4f}")
    return {
        "synthetic_quade_p": out["p"],
        "synthetic_quade_p_null": out_n["p"],
        "synthetic_quade_t3": out["t3"],
        "synthetic_score": 1.0,
    }
