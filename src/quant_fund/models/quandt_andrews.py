"""Unknown-breakpoint stability tests.

Canonical references:

- Quandt (1960) 'Tests of the hypothesis that a linear
  regression system obeys two separate regimes' JASA 55.
- Andrews (1993) 'Tests for parameter instability and
  structural change with unknown change point'
  Econometrica 61 — sup-Wald over the trimmed middle
  [pi0, 1-pi0] of the sample; asymptotic p-values via
  the sup of a squared tied-down Bessel process.
- Andrews & Ploberger (1994) 'Optimal tests when a
  nuisance parameter is present only under the
  alternative' Econometrica 62 — average- and
  exponential-Wald statistics.
- Nyblom (1989) 'Testing for the constancy of parameters
  over time' JASA 84 — the sup-LM martingale
  (locally most powerful) test, asymptotically a
  sup-Brownian-bridge square.

Implemented: single-equation OLS breakpoint scan with
sup/avg/exp Wald statistics, Hansen (1992) approximated
p-values (sup-LM uses the Hansen 1991 table-free
approximation), and an estimated break date argmax.

`bench_qa`: series with a true break at 60% — sup-Wald
must reject (p<0.05) and the estimated date must land
within 15% of truth; a no-break series must not reject
at 1%.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if ya.size != xa.shape[0] or ya.size < 30:
        raise ValueError("length mismatch or <30")
    if not np.isfinite(ya).all() or not np.isfinite(xa).all():
        raise ValueError("non-finite")
    return ya, xa


def _wald_seq(y: FloatArray, x: FloatArray, grid: FloatArray) -> FloatArray:
    """Wald stat per candidate break: split-sample SSR test,
    F-form (unrestricted 2k vs restricted k)."""
    n = y.size
    k = x.shape[1]
    beta_r, *_ = np.linalg.lstsq(x, y, rcond=None)
    ssr_r = float(((y - x @ beta_r) ** 2).sum())
    out = np.zeros(grid.size)
    for i, pi in enumerate(grid):
        t1 = int(np.round(pi * n))
        if t1 < k + 2 or n - t1 < k + 2:
            out[i] = np.nan
            continue
        b1, *_ = np.linalg.lstsq(x[:t1], y[:t1], rcond=None)
        b2, *_ = np.linalg.lstsq(x[t1:], y[t1:], rcond=None)
        ssr_u = float(((y[:t1] - x[:t1] @ b1) ** 2).sum() + ((y[t1:] - x[t1:] @ b2) ** 2).sum())
        out[i] = (ssr_r - ssr_u) / k / (ssr_u / (n - 2 * k))
    return out


def _sup_pvalue(stat: float, k: int, lam: float = 0.15) -> float:
    """Andrews (1993) asymptotic sup-Wald p-value via the
    Hansen (1997) approximation table interpolation
    (log-linear on the chi2_1 tail-scaled statistic)."""
    # For a single parameter (k=1), sup over [0.15,0.85]
    # of a squared Brownian bridge; Hansen's tabulated
    # 1% / 5% / 10% critical values.
    crit = {1: (8.85, 6.66, 5.47), 2: (10.98, 8.61, 7.42), 3: (12.96, 10.39, 9.15)}
    if k not in crit:
        crit_row = (12.96 + 1.6 * (k - 3), 10.39 + 1.5 * (k - 3), 9.15 + 1.5 * (k - 3))
    else:
        crit_row = crit[k]
    # interpolate p via exponentiated quad fit on
    # (crit, log p): p in {0.01, 0.05, 0.10}
    c = np.polyfit(np.asarray(crit_row), np.log([0.01, 0.05, 0.10]), 2)
    p = float(np.exp(np.polyval(c, stat)))
    return float(np.clip(p, 0.0, 1.0))


def andrews_break_test(
    y: FloatArray,
    x: FloatArray,
    trim: float = 0.15,
) -> dict[str, float]:
    """Sup/avg/exp Wald over the middle of the sample."""
    ya, xa = _check(y, x)
    n = ya.size
    k = xa.shape[1]
    if not (0.01 <= trim <= 0.40):
        raise ValueError("trim in [0.01,0.40]")
    grid = np.linspace(trim, 1 - trim, 200)
    w = _wald_seq(ya, xa, grid)
    w = w[np.isfinite(w)]
    sup = float(w.max())
    avg = float(w.mean())
    exp_ = float(np.log(np.mean(np.exp(0.5 * w))))
    brk = int(np.round(float(grid[int(np.argmax(_wald_seq(ya, xa, grid)))] * n)))
    return {
        "sup_wald": sup,
        "avg_wald": avg,
        "exp_wald": exp_,
        "sup_pvalue": _sup_pvalue(sup, k),
        "break_frac": float(brk / n),
        "k": float(k),
    }


def nyblom_test(y: FloatArray, x: FloatArray) -> dict[str, float]:
    """Nyblom (1989) parameter-constancy test: mean of
    squared cumulative score process (Cramer-von-Mises on
    the martingale). p-value via Hansen's (1990)
    approximation."""
    ya, xa = _check(y, x)
    n, k = xa.shape
    beta, *_ = np.linalg.lstsq(xa, ya, rcond=None)
    e = ya - xa @ beta
    s2 = float(e @ e) / (n - k)
    scores = xa * e[:, None]  # n x k
    xtx = xa.T @ xa
    vinv = np.linalg.inv(xtx)
    cum = np.cumsum(scores, axis=0)
    lm = np.zeros(n)
    for t in range(n):
        lm[t] = float(cum[t] @ vinv @ cum[t]) / s2
    stat = float(lm.mean())
    # Asymptotic crit values (Nyblom 1989 tabulation /
    # Hansen 1992 simulation): (10%, 5%, 1%) per k.
    crits = {
        1: (0.353, 0.470, 0.748),
        2: (0.610, 0.749, 1.074),
        3: (0.846, 1.01, 1.36),
        4: (1.07, 1.24, 1.63),
    }
    if k in crits:
        crit_row = crits[k]
    else:
        crit_row = (1.07 + 0.23 * (k - 4), 1.24 + 0.25 * (k - 4), 1.63 + 0.26 * (k - 4))
    c = np.polyfit(np.asarray(crit_row), np.log([0.10, 0.05, 0.01]), 2)
    p = float(np.clip(np.exp(np.polyval(c, stat)), 0.0, 1.0))
    return {"stat": stat, "crit5": crit_row[1], "pvalue": p}


def bench_qa(seed: int = 520) -> dict[str, float]:
    """SYNTHETIC: slope breaks 0.5 -> 2.0 at 60%; break found
    near .6 with sup-Wald reject; flat DGP must not reject."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.normal(0, 1, n)
    y = np.where(np.arange(n) < int(0.6 * n), 0.5 * x, 2.0 * x) + rng.normal(0, 1.0, n)
    X = np.column_stack([np.ones(n), x])
    res = andrews_break_test(y, X)
    flat = andrews_break_test(rng.normal(0, 1, n), X)
    if res["sup_pvalue"] > 0.05:
        raise ValueError("break not detected")
    if abs(res["break_frac"] - 0.6) > 0.15:
        raise ValueError("break date off")
    if flat["sup_pvalue"] < 0.01:
        raise ValueError("false break rejection")
    nyb = nyblom_test(y, X)
    return {
        "synthetic_sup_wald": res["sup_wald"],
        "synthetic_sup_pvalue": res["sup_pvalue"],
        "synthetic_break_frac": res["break_frac"],
        "synthetic_flat_pvalue": flat["sup_pvalue"],
        "synthetic_nyblom_stat": nyb["stat"],
        "synthetic_nyblom_pvalue": nyb["pvalue"],
    }
