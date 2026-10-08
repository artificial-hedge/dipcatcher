"""Heteroskedasticity diagnostics for OLS residuals (SYNTHETIC).

Canonical references:

- Goldfeld & Quandt (1965) 'Some tests for
  homoscedasticity' JASA 60 — split-sample residual
  variance F test after dropping d central
  observations.
- Park (1966) 'Estimation with heteroscedastic error
  terms' Econometrica 34 — regression of log e^2 on
  log x; slope and its t.
- Glejser (1969) 'A new test for heteroskedasticity'
  JASA 64 — |e| on forms of x.
- Breusch & Pagan (1979) 'A simple test for
  heteroscedasticity and random coefficient
  variation' Econometrica 47 — LM = RSS_aux/2 where
  the auxiliary regresses e^2/(ee'/n) on the
  regressors (chi2_k).
- White (1980) 'A heteroskedasticity-consistent
  covariance matrix estimator and a direct test for
  heteroskedasticity' Econometrica 48 — nR^2 of the
  auxiliary regression of e^2 on the regressors,
  squares and (optionally) cross products (chi2_p).

`bench_het`: homoskedastic regressions accepted ~95%;
multiplicative heteroskedasticity rejected at high
power.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _ols(x: FloatArray, y: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.shape[0] != ya.size or ya.size < xa.shape[1] + 5:
        raise ValueError("bad regression")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    xd = np.column_stack([np.ones(xa.shape[0]), xa])
    b = np.linalg.lstsq(xd, ya, rcond=None)[0]
    return ya - xd @ b


def goldfeld_quandt(x: FloatArray, y: FloatArray, drop: int = 0) -> dict[str, float]:
    """GQ split-sample F on residual variance."""
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2 or xa.shape[1] != 1:
        raise ValueError("GQ needs a single regressor")
    xa = xa[:, 0]
    order = np.argsort(xa)
    xs, ys = xa[order], np.asarray(y, dtype=np.float64).ravel()[order]
    n = xs.size
    d = drop if drop > 0 else n // 5
    n1 = (n - d) // 2
    lo_x, lo_y = xs[:n1], ys[:n1]
    hi_x, hi_y = xs[n - n1 :], ys[n - n1 :]
    e1 = _ols(lo_x[:, None], lo_y)
    e2 = _ols(hi_x[:, None], hi_y)
    s1 = float((e1**2).sum() / (n1 - 2))
    s2 = float((e2**2).sum() / (n1 - 2))
    f = s2 / s1
    return {
        "stat": float(f),
        "pvalue": float(stats.f.sf(f, n1 - 2, n1 - 2)),
        "ratio": f,
    }


def park(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Park (1966): log e^2 on log |x|, slope t-test."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    e = _ols(xa[:, None], y)
    mask = np.abs(xa) > 1e-8
    lx = np.log(np.abs(xa[mask]))
    le = np.log(e[mask] ** 2)
    xd = np.column_stack([np.ones(lx.size), lx])
    b, res_, rank, _ = np.linalg.lstsq(xd, le, rcond=None)
    k = xd.shape[1]
    s2 = float(((le - xd @ b) ** 2).sum() / (lx.size - k))
    xtx = np.linalg.inv(xd.T @ xd)
    se = np.sqrt(s2 * xtx[1, 1])
    t = b[1] / se
    return {
        "stat": float(t),
        "pvalue": float(2 * stats.t.sf(abs(t), lx.size - k)),
        "slope": float(b[1]),
    }


def glejser(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Glejser (1969): |e| on x (form x^1), slope t."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    e = np.abs(_ols(xa[:, None], y))
    xd = np.column_stack([np.ones(xa.size), xa])
    b, *_ = np.linalg.lstsq(xd, e, rcond=None)
    k = xd.shape[1]
    s2 = float(((e - xd @ b) ** 2).sum() / (xa.size - k))
    xtx = np.linalg.inv(xd.T @ xd)
    se = np.sqrt(s2 * xtx[1, 1])
    t = b[1] / se
    return {
        "stat": float(t),
        "pvalue": float(2 * stats.t.sf(abs(t), xa.size - k)),
        "slope": float(b[1]),
    }


def breusch_pagan(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """BP LM test: aux regression of normalized e^2 on x."""
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    e = _ols(xa, y)
    g = e**2 / float((e**2).mean())
    xd = np.column_stack([np.ones(xa.shape[0]), xa])
    b = np.linalg.lstsq(xd, g, rcond=None)[0]
    rss = float(((g - g.mean()) ** 2 - (g - xd @ b) ** 2).sum())
    lm = 0.5 * rss
    k = xa.shape[1]
    return {
        "stat": float(lm),
        "pvalue": float(stats.chi2.sf(lm, k)),
        "df": float(k),
    }


def white(x: FloatArray, y: FloatArray, cross: bool = True) -> dict[str, float]:
    """White (1980) nR^2 test on squares + cross-products."""
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    n, p = xa.shape
    e = _ols(xa, y)
    cols = [xa]
    cols.append(xa**2)
    if cross and p > 1:
        cross_cols = [xa[:, i] * xa[:, j] for i in range(p) for j in range(i + 1, p)]
        if cross_cols:
            cols.append(np.column_stack(cross_cols))
    z = np.column_stack([c for c in cols])
    zd = np.column_stack([np.ones(n), z])
    b = np.linalg.lstsq(zd, e**2, rcond=None)[0]
    r2 = 1 - float(((e**2 - zd @ b) ** 2).sum() / ((e**2 - (e**2).mean()) ** 2).sum())
    q = zd.shape[1] - 1
    stat = n * r2
    return {
        "stat": float(stat),
        "pvalue": float(stats.chi2.sf(stat, q)),
        "df": float(q),
        "r2": float(r2),
    }


def bench_het(seed: int = 526) -> dict[str, float]:
    """SYNTHETIC: y = a+bx + e, e ~ N(0, (0.5+1.2x^+)^2)
    heteroskedastic; homoskedastic control."""
    rng = np.random.default_rng(seed)
    R = 60
    tests = {
        "gq": goldfeld_quandt,
        "park": park,
        "glej": glejser,
        "bp": breusch_pagan,
        "white": lambda x, y: white(x, y, cross=True),
    }
    rej_null = {k: 0 for k in tests}
    rej_alt = {k: 0 for k in tests}
    for _ in range(R):
        x = rng.uniform(0, 3, 120)
        y_hom = 1 + x + rng.normal(0, 0.8, 120)
        sig = 0.3 + 1.1 * np.clip(x, 0, None)
        y_het = 1 + x + rng.normal(0, 1, 120) * sig
        for name, fn in tests.items():
            rej_null[name] += int(fn(x[:, None], y_hom)["pvalue"] < 0.05)
            rej_alt[name] += int(fn(x[:, None], y_het)["pvalue"] < 0.05)
    out: dict[str, float] = {}
    for name in tests:
        out[f"synthetic_type1_{name}"] = rej_null[name] / R
        out[f"synthetic_power_{name}"] = rej_alt[name] / R
        if rej_alt[name] < R * 0.8:
            raise ValueError(f"{name} misses heteroskedasticity")
        if rej_null[name] > R * 0.18:
            raise ValueError(f"{name} over-rejects")
    return out
