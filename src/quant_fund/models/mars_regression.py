"""Multivariate Adaptive Regression Splines (MARS).

Canonical reference:

- Friedman (1991) 'Multivariate adaptive regression
  splines' Annals of Statistics 19 — the forward
  pass adds pairs of hinge basis functions
  (x_j - t)_+ and (t - x_j)_+ chosen to minimize
  residual SS; the backward pass prunes terms one at
  a time under the generalized cross-validation
  criterion
  GCV(M) = RSS/n / (1 - c(M)/n)^2,
  c(M) = M + d*M with d=3 (the standard smoothing
  parameter), matching earth/MARS conventions.

`bench_mars`: recovers a piecewise-linear interaction
y = 3*(x1-0.5)_+ * (x2-0.3)_+ - 2*(0.7-x3)_+ + noise
with test RMSE materially below a linear baseline.
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


def _build_basis(x: FloatArray, terms: list[tuple[int, int, float, int]]) -> FloatArray:
    """Basis columns for term list; column 0 is intercept.

    term = (parent_col, feature, knot, sgn): the new
    column is parent_col's values multiplied by the
    hinge sgn*(x_feat - knot) truncated at 0."""
    n = x.shape[0]
    basis = [np.ones(n)]
    for parent, col, t, sgn in terms:
        h = np.clip(sgn * (x[:, col] - t), 0.0, None)
        basis.append(basis[parent] * h)
    return np.column_stack(basis)


def _hinge(x: FloatArray, col: int, t: float, sgn: int) -> FloatArray:
    return np.clip(sgn * (x[:, col] - t), 0.0, None)


def mars_fit(
    x: FloatArray,
    y: FloatArray,
    max_terms: int = 15,
    n_knots: int = 8,
    d: float = 3.0,
) -> dict[str, object]:
    """Forward-add hinge pairs then GCV-prune.

    ``all_terms``: every forward-pass term (needed to
    rebuild parent columns during prediction);
    ``keep``: indices of retained basis columns
    (0 = intercept, i>0 maps to all_terms[i-1])."""
    xa, ya = _check_xy(x, y)
    n, p = xa.shape
    knots = np.linspace(0.05, 0.95, n_knots)
    all_terms: list[tuple[int, int, float, int]] = []
    resid = ya - ya.mean()
    basis = [np.ones(n)]
    for _ in range(max_terms):
        # each step adds a hinge PAIR (both directions);
        # select best (parent, col, knot) by the pair's
        # joint RSS after orthogonalizing each leg.
        best = (np.inf, -1, -1, 0.0)
        for parent in range(len(basis)):
            depth = 1
            j = parent
            while j > 0:
                depth += 1
                j = all_terms[j - 1][0]
            if depth > 2:
                continue
            for col in range(p):
                for t in knots:
                    tt = float(np.quantile(xa[:, col], t))
                    c1 = basis[parent] * _hinge(xa, col, tt, 1)
                    c2 = basis[parent] * _hinge(xa, col, tt, -1)
                    cand = np.column_stack([c1, c2])
                    # RSS after projecting out the pair
                    q, _ = np.linalg.qr(cand)
                    proj = resid - q @ (q.T @ resid)
                    rss = float(proj @ proj)
                    if rss < best[0]:
                        best = (rss, parent, col, tt)
        rss, parent, col, tt = best
        if parent < 0:
            break
        for sgn in (1, -1):
            all_terms.append((parent, col, tt, sgn))
            basis.append(basis[parent] * _hinge(xa, col, tt, sgn))
        bmat = np.column_stack(basis)
        coef = np.linalg.lstsq(bmat, ya, rcond=None)[0]
        resid = ya - bmat @ coef
    bmat = np.column_stack(basis)

    def gcv(cols: list[int]) -> float:
        bm = bmat[:, cols]
        cf = np.linalg.lstsq(bm, ya, rcond=None)[0]
        rss = float(((ya - bm @ cf) ** 2).sum())
        m = len(cols)
        c = m + d * m
        return rss / n / (1 - c / n) ** 2 if c < n else np.inf

    keep = list(range(bmat.shape[1]))
    g_cur = gcv(keep)
    improved = True
    while improved and len(keep) > 1:
        improved = False
        for i in keep[1:]:  # never drop the intercept
            cols = [j for j in keep if j != i]
            g = gcv(cols)
            if g < g_cur:
                g_cur, keep = g, cols
                improved = True
                break
    coef = np.linalg.lstsq(bmat[:, keep], ya, rcond=None)[0]
    return {
        "all_terms": all_terms,
        "keep": keep,
        "coef": np.asarray(coef),
        "gcv": float(g_cur),
        "rss": float(((ya - bmat[:, keep] @ coef) ** 2).sum()),
    }


def mars_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2:
        raise ValueError("bad design")
    bmat = _build_basis(xa, model["all_terms"])  # type: ignore[arg-type]
    keep = np.asarray(model["keep"], dtype=int)
    coef = np.asarray(model["coef"], dtype=np.float64)
    return np.asarray(bmat[:, keep] @ coef)


def bench_mars(seed: int = 528) -> dict[str, float]:
    """SYNTHETIC: interaction hinge surface — MARS must
    beat linear RMSE by >30% on held-out data."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0, 1, (n, 4))
    f = (
        8.0 * np.clip(x[:, 0] - 0.5, 0, None) * np.clip(x[:, 1] - 0.3, 0, None)
        - 5.0 * np.clip(0.7 - x[:, 2], 0, None)
        + 0.4 * x[:, 3]
    )
    y = f + rng.normal(0, 0.2, n)
    tr, te = slice(0, 300), slice(300, 400)
    m = mars_fit(x[tr], y[tr], max_terms=12, n_knots=20)
    pred = mars_predict(m, x[te])
    rmse_mars = float(np.sqrt(((y[te] - pred) ** 2).mean()))
    xd = np.column_stack([np.ones(300), x[tr]])
    b = np.linalg.lstsq(xd, y[tr], rcond=None)[0]
    xt = np.column_stack([np.ones(100), x[te]])
    rmse_lin = float(np.sqrt(((y[te] - xt @ b) ** 2).mean()))
    if rmse_mars > 0.7 * rmse_lin:
        raise ValueError("mars no better than linear")
    keep_list = m["keep"]
    if not isinstance(keep_list, list) or len(keep_list) < 2:
        raise ValueError("mars kept too few terms")
    return {
        "synthetic_rmse_mars": rmse_mars,
        "synthetic_rmse_linear": rmse_lin,
        "synthetic_ratio": rmse_mars / rmse_lin,
        "synthetic_terms": float(len(keep_list)),
        "synthetic_gcv": float(m["gcv"]),  # type: ignore[arg-type]
    }
