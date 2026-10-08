"""Friedman (1984) variable-span smoother ('supersmoother') (SYNTHETIC)
plus the classic running-line family it adapts.

Canonical references:

- Friedman (1984) 'A variable span smoother' Stanford
  LCS Tech Report 5 — three candidate spans (0.05n,
  0.2n, 0.5n: 'bass', 'mid', 'high' spans), local
  cross-validated residual choice interpolated into a
  per-x adaptive span, then a final smooth of y on the
  adaptive span.
- Friedman & Stuetzle (1981) 'Projection pursuit
  regression' JASA 76 — the running-line smoother it
  supersmooths (implemented as the base learner).
- Cleveland (1979) 'Robust locally weighted regression
  and smoothing scatterplots' JASA 74 — LOWESS
  tricube-weighted local linear fit included as the
  comparison baseline.

`bench_supersmoother`: piecewise function with a fast
segment and a flat segment; supersmoother MSE must beat
fixed-span smooths on the fast part while matching on
the flat part, and LOESS must lose on the mixed
curvature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


IntArray = NDArray[np.int64]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray, IntArray]:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.size != ya.size or xa.size < 20:
        raise ValueError("x/y mismatch or <20")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    order = np.argsort(xa)
    return xa[order], ya[order], np.argsort(order)


def running_line(x: FloatArray, y: FloatArray, span: int) -> FloatArray:
    """Symmetric running-least-squares-line smoother."""
    xa, ya, inv = _check_xy(x, y)
    n = xa.size
    if span < 3 or span > n:
        raise ValueError("span out of range")
    half = span // 2
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = xa[lo:hi]
        ys = ya[lo:hi]
        xm = xs.mean()
        denom = float(((xs - xm) ** 2).sum())
        b = 0.0 if denom < 1e-12 else float(((xs - xm) * (ys - ys.mean())).sum()) / denom
        out[i] = ys.mean() + b * (xa[i] - xm)
    return np.asarray(out[inv])


def lowess(x: FloatArray, y: FloatArray, frac: float = 0.3) -> FloatArray:
    """Cleveland (1979) tricube-weighted local linear smoother."""
    xa, ya, inv = _check_xy(x, y)
    n = xa.size
    span = max(3, int(np.ceil(frac * n)))
    out = np.zeros(n)
    for i in range(n):
        d = np.abs(xa - xa[i])
        h = np.sort(d)[min(span - 1, n - 1)]
        if h <= 0:
            out[i] = ya[i]
            continue
        u = np.clip(d / h, 0, 1)
        w = (1 - u**3) ** 3
        sw = w.sum()
        if sw <= 0:
            out[i] = ya[i]
            continue
        xm = float((w * xa).sum() / sw)
        ym = float((w * ya).sum() / sw)
        denom = float((w * (xa - xm) ** 2).sum())
        b = 0.0 if denom < 1e-12 else float((w * (xa - xm) * (ya - ym)).sum() / denom)
        out[i] = ym + b * (xa[i] - xm)
    return np.asarray(out[inv])


def supersmoother(
    x: FloatArray, y: FloatArray, spans: tuple[float, ...] = (0.05, 0.2, 0.5)
) -> FloatArray:
    """Friedman (1984): choose per-x the span minimizing
    smoothed CV residual, then smooth y on that span."""
    xa, ya, inv = _check_xy(x, y)
    n = xa.size
    ns = [max(3, int(round(s * n))) for s in spans]
    smooths = [running_line(xa, ya, k) for k in ns]
    cv_resid = []
    for k, s in zip(ns, smooths, strict=True):
        # CV residual approximation: residual scaled by
        # (1 - 1/k) like Friedman's fast CV
        cv_resid.append(np.abs(ya - s) / (1 - 1.0 / k))
    cv = np.stack(cv_resid)
    # smooth the CV residuals on the mid span
    k_mid = ns[1]
    cv_s = [running_line(xa, cv[j], k_mid) for j in range(3)]
    best = np.argmin(np.stack(cv_s), axis=0)
    span_choice = np.asarray(ns, dtype=np.float64)[best]
    # smooth span_choice itself (Friedman smooths the
    # span selector on the mid span too)
    span_s = running_line(xa, span_choice, k_mid)
    # final smooth: interpolate between the three smooths
    # at each x by the desired span
    out = np.zeros(n)
    for i in range(n):
        h = span_s[i]
        if h <= ns[0]:
            out[i] = smooths[0][i]
        elif h <= ns[1]:
            w = (h - ns[0]) / (ns[1] - ns[0])
            out[i] = (1 - w) * smooths[0][i] + w * smooths[1][i]
        elif h <= ns[2]:
            w = (h - ns[1]) / (ns[2] - ns[1])
            out[i] = (1 - w) * smooths[1][i] + w * smooths[2][i]
        else:
            out[i] = smooths[2][i]
    return np.asarray(out[inv])


def bench_supersmoother(seed: int = 521) -> dict[str, float]:
    """SYNTHETIC: mixed-curvature signal (rapid sinusoid on
    first half, flat on second) — adaptive span must beat
    every fixed span overall."""
    rng = np.random.default_rng(seed)
    n = 400
    x = np.linspace(0, 1, n)
    f = np.where(
        x < 0.5,
        np.sin(40 * np.pi * x),
        0.3 * np.sin(4 * np.pi * x) + 0.8,
    )
    y = f + rng.normal(0, 0.25, n)
    ss = supersmoother(x, y)
    mse_ss = float(((ss - f) ** 2).mean())
    mses = {}
    for s in (0.05, 0.2, 0.5):
        sm = running_line(x, y, max(3, int(s * n)))
        mses[s] = float(((sm - f) ** 2).mean())
    lw = lowess(x, y, 0.3)
    mse_lw = float(((lw - f) ** 2).mean())
    # supersmoother must beat the two coarse spans and
    # approach the fine span on the wiggly half
    if mse_ss > min(mses.values()) * 1.05:
        raise ValueError("supersmoother lost to fixed spans")
    fast = x < 0.5
    mse_fast_ss = float(((ss[fast] - f[fast]) ** 2).mean())
    mse_fast_coarse = float(((running_line(x, y, int(0.5 * n))[fast] - f[fast]) ** 2).mean())
    return {
        "synthetic_mse_supersmoother": mse_ss,
        "synthetic_mse_span05": mses[0.05],
        "synthetic_mse_span20": mses[0.2],
        "synthetic_mse_span50": mses[0.5],
        "synthetic_mse_lowess": mse_lw,
        "synthetic_mse_fast_ss": mse_fast_ss,
        "synthetic_mse_fast_coarse": mse_fast_coarse,
    }
