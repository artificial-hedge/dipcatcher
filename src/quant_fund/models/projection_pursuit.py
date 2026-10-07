"""Projection pursuit regression (PPR) and indices (SYNTHETIC).

Canonical references:

- Friedman & Tukey (1974) 'A projection pursuit
  algorithm for exploratory data analysis' IEEE TC —
  maximize an index over directions a:
  I(a) = s(z) * d(z) with z = a'x, s a scale of the
  projected cloud and d a local-density term.
- Friedman & Stuetzle (1981) 'Projection pursuit
  regression' JASA 76 — additive ridge model
  y = sum_m g_m(a_m' x); each (a, g) found by
  alternating direction updates (Gauss-Newton on the
  sphere) and local smoothing of g.

`bench_ppr`: y = sin(2 a0'x) + noise with a0 =
(1,-1,0.5)/||.|| must be fit far better than a linear
model, and the recovered direction must align with
a0 (|cos| > 0.9).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.ndim != 2 or xa.shape[0] != ya.size or xa.shape[0] < 30:
        raise ValueError("bad design")
    if not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    return xa, ya


def _smooth(y: FloatArray, span: float = 0.3) -> FloatArray:
    """Running-mean smoother on a 1-D projection index."""
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = ya.size
    k = max(int(np.ceil(span * n)), 5)
    half = k // 2
    csum = np.concatenate([[0.0], np.cumsum(ya)])
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i - half + k)
        lo = max(0, min(lo, hi - k))
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def _eval_smooth(z_train: FloatArray, g_train: FloatArray, z_new: FloatArray) -> FloatArray:
    """Evaluate the fitted ridge function at new projections
    via interpolation on sorted training projections."""
    order = np.argsort(z_train)
    return np.asarray(
        np.interp(
            z_new,
            z_train[order],
            g_train[order],
            left=g_train[order][0],
            right=g_train[order][-1],
        ),
        dtype=np.float64,
    )


def ft_index(x: FloatArray) -> float:
    """Friedman-Tukey index of a projected cloud z = a'x.
    I(a) = s(z) * d(z), s = projection std, d =
    (1/n^2) sum_{ij} (1 - (z_ij/R)^2)_+ — the trimmed
    local density along the projection."""
    z = np.asarray(x, dtype=np.float64).ravel()
    n = z.size
    if n < 10:
        raise ValueError("too few points")
    s = float(z.std())
    if s <= 0:
        return 0.0
    r = 2.0 * s
    diff = z[:, None] - z[None, :]
    u = np.clip(1 - (diff / r) ** 2, 0.0, None)
    d = float(u.sum() / (n * n))
    return s * d


def ppr_fit(
    x: FloatArray,
    y: FloatArray,
    n_terms: int = 3,
    span: float = 0.3,
    n_starts: int = 8,
    max_iter: int = 30,
    seed: int = 0,
) -> dict[str, object]:
    """PPR: sequential ridge-term fit.

    Each term: initialize direction, then alternate
    (a) smooth g(z) = E[resid | a'x] and (b) update a
    by one Gauss-Newton step minimizing ||resid -
    g(a'x)||^2; keep the best of n_starts random
    restarts."""
    xa, ya = _check_xy(x, y)
    n, p = xa.shape
    rng = np.random.default_rng(seed)
    resid = ya - ya.mean()
    mus = float(ya.mean())
    terms: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for _m in range(n_terms):
        best: tuple[float, FloatArray | None, tuple[FloatArray, FloatArray] | None] = (
            np.inf,
            None,
            None,
        )
        for _s in range(n_starts):
            a = rng.normal(0, 1, p)
            a /= np.linalg.norm(a)
            for _it in range(max_iter):
                z = xa @ a
                order = np.argsort(z)
                g = _smooth(resid[order], span)
                g_full = np.zeros(n)
                g_full[order] = g
                # GN step on a: d(resid - g(a'x))/da ~ -g'(z) x
                dz = np.gradient(g_full, z)
                j = -dz[:, None] * xa  # n x p jacobian of g(a'x)
                r = resid - g_full
                # GN: r(a+da) ~ r + J da -> da = -J^+ r
                delta = -np.linalg.lstsq(j, r, rcond=None)[0]
                a_new = a + delta
                nn = np.linalg.norm(a_new)
                if nn < 1e-9 or not np.isfinite(a_new).all():
                    break
                a_new /= nn
                if np.linalg.norm(a_new - a) < 1e-6:
                    a = a_new
                    break
                a = a_new
            z = xa @ a
            order = np.argsort(z)
            g = _smooth(resid[order], span)
            r = resid.copy()
            r[order] -= g
            rss = float(r @ r)
            if rss < best[0]:
                best = (rss, a, (z[order], g))
        if best[1] is None or best[2] is None:
            break
        a_best, gz = best[1], best[2]
        z_sorted, g_sorted = gz
        resid = resid.copy()
        resid[np.argsort(xa @ a_best)] -= g_sorted
        terms.append((a_best, z_sorted, g_sorted))
    return {"terms": terms, "mu": mus, "resid_sd": float(resid.std())}


def ppr_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2:
        raise ValueError("bad design")
    out = np.full(xa.shape[0], float(model["mu"]))  # type: ignore[arg-type]
    terms = model["terms"]
    if not isinstance(terms, list):
        raise ValueError("bad model")
    for a, z_tr, g_tr in terms:
        av = np.asarray(a, dtype=np.float64)
        zt = np.asarray(z_tr, dtype=np.float64)  # sorted ascending
        gt = np.asarray(g_tr, dtype=np.float64)
        out = out + np.interp(xa @ av, zt, gt, left=gt[0], right=gt[-1])
    return np.asarray(out, dtype=np.float64)


def bench_ppr(seed: int = 530) -> dict[str, float]:
    """SYNTHETIC: single ridge + one interaction term.
    PPR must beat linear and align first direction."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(-1, 1, (n, 3))
    a0 = np.array([1.0, -1.0, 0.5])
    a0 /= np.linalg.norm(a0)
    y = np.sin(2.5 * (x @ a0)) + 0.4 * x[:, 2] + rng.normal(0, 0.15, n)
    tr, te = slice(0, 240), slice(240, 300)
    m = ppr_fit(x[tr], y[tr], n_terms=2, n_starts=6)
    pred = ppr_predict(m, x[te])
    rmse = float(np.sqrt(((y[te] - pred) ** 2).mean()))
    xd = np.column_stack([np.ones(240), x[tr]])
    b = np.linalg.lstsq(xd, y[tr], rcond=None)[0]
    xt = np.column_stack([np.ones(60), x[te]])
    rmse_lin = float(np.sqrt(((y[te] - xt @ b) ** 2).mean()))
    terms_b = m["terms"]
    if not isinstance(terms_b, list):
        raise ValueError("bad model")
    a_hat = np.asarray(terms_b[0][0], dtype=np.float64)
    cos = float(abs(a_hat @ a0))
    if rmse > 0.75 * rmse_lin:
        raise ValueError("ppr no better than linear")
    if cos < 0.9:
        raise ValueError(f"direction misaligned cos={cos:.3f}")
    return {
        "synthetic_rmse_ppr": rmse,
        "synthetic_rmse_linear": rmse_lin,
        "synthetic_cos_a0": cos,
        "synthetic_n_terms": float(len(terms_b)),
    }
