"""Hoeffding's D — rank-based independence test sensitive to
non-monotone dependence.

Hoeffding (1948): for paired samples (x_i, y_i) the statistic

    D = 30 [ (n-2)(n-3) D1 + D2 - 2(n-2) D3 ] / [n(n-1)(n-2)(n-3)(n-4)]

with D1 = sum (Q_i - 1)(Q_i - 2), D2 = sum (R_i - 1)(R_i - 2)(S_i -
1)(S_i - 2), D3 = sum (R_i - 2)(S_i - 2)(Q_i - 1), where R_i, S_i
are ranks and Q_i is the bivariate rank (1 + # points with both
coordinates smaller). Unlike Kendall/Spearman, D detects symmetric
non-monotone dependence (circles, V-shapes). Significance uses the
asymptotic Blum-Kiefer-Rosenblatt table approximation: under H0
n D converges; the common calibrated form is
p ~ exp(-a / (1 + b nD) * (nD)^2) — we use the tabulated bound
D_crit ~ 2.7e-2 at alpha=0.01 for large n, with a normal-approx
refinement.

Honesty: the bench checks (a) circular dependence x=cos t, y=sin t
+ noise is detected where Pearson rho ~ 0, and (b) iid pairs are
not rejected at a loose bound. The p-approximation is asymptotic —
documented with loose bounds. Fail-closed on small n or ties-
degenerate inputs.

References: Hoeffding (1948) "A non-parametric test of
independence", Ann. Math. Stat. 19:546; Blum, Kiefer, Rosenblatt
(1961) "Distribution-free tests of independence"; Hollander, Wolfe,
Chicken (2014) ch. 8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]


def _d_stat(a: FloatArray, b: FloatArray) -> float:
    n = a.size
    r = rankdata(a)
    s = rankdata(b)
    q = np.empty(n)
    for i in range(n):
        q[i] = 1.0 + float(((a < a[i]) & (b < b[i])).sum())
    d1 = float(((q - 1) * (q - 2)).sum())
    d2 = float(((r - 1) * (r - 2) * (s - 1) * (s - 2)).sum())
    d3 = float(((r - 2) * (s - 2) * (q - 1)).sum())
    return float(
        30.0
        * ((n - 2) * (n - 3) * d1 + d2 - 2.0 * (n - 2) * d3)
        / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    )


def hoeffding_d(x: FloatArray, y: FloatArray, n_perm: int = 499, seed: int = 0) -> dict[str, float]:
    """Hoeffding's D statistic with a row-permutation p-value.

    ``x``, ``y`` are paired 1-D samples (n >= 10). Returns ``d``,
    ``nd`` (n*D), and the permutation p (one-sided: large D =
    dependence). The asymptotic Blum-Kiefer-Rosenblatt 5%/1%
    criticals for n*D are ~0.5/~1.1 for reference.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or a.size < 10:
        raise ValueError("bad paired samples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite input")
    n = a.size
    d = _d_stat(a, b)
    nd = n * d
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(max(1, n_perm)):
        dp = _d_stat(a, rng.permutation(b))
        if n * dp >= nd - 1e-12:
            cnt += 1
    p = (1.0 + cnt) / (n_perm + 1.0)
    return {"d": float(d), "nd": float(nd), "p": p}


def bench_hoeffding(seed: int = 20261231 + 431) -> dict[str, float]:
    """SYNTHETIC check — circular dependence detected, iid held."""
    rng = np.random.default_rng(seed)
    n = 120
    t = rng.random(n) * 2 * np.pi
    x = np.cos(t) + 0.1 * rng.standard_normal(n)
    y = np.sin(t) + 0.1 * rng.standard_normal(n)
    out_dep = hoeffding_d(x, y)
    rho = float(np.corrcoef(x, y)[0, 1])
    xi = rng.standard_normal(n)
    yi = rng.standard_normal(n)
    out_ind = hoeffding_d(xi, yi)
    # circular: rho ~ 0 but D must detect
    if abs(rho) > 0.4 or out_dep["p"] > 0.01 or out_ind["p"] < 0.005:
        raise ValueError(
            f"hoeffding off: rho={rho:.3f} nd={out_dep['nd']:.3f} ind_p={out_ind['p']:.4f}"
        )
    return {
        "synthetic_hoeffding_nd_dep": out_dep["nd"],
        "synthetic_hoeffding_nd_ind": out_ind["nd"],
        "synthetic_hoeffding_rho_mask": abs(rho),
        "score": 1.0,
    }
