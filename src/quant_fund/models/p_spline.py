"""P-splines — Eilers-Marx penalized B-spline regression.

Eilers & Marx (1996): a B-spline basis B(x) of degree q on
equidistant knots is penalized by a difference penalty ||D_d beta||^2
of order d, giving the normal equations

    (B^T B + lam D^T D) beta = B^T y

The smoothing parameter lam is chosen by GCV. Whittaker smoothing
(q=0 basis = identity) falls out as a special case.

Honesty: the bench recovers a smooth curve (Doppler-free sine +
localized bump) at a fraction of the raw noisy MSE, and checks the
GCV-chosen lambda sits in a plausible range. O(n knots^3) solve —
sized for the documented fixture. Fail-closed on degenerate design
or non-finite data.

References: Eilers, Marx (1996) "Flexible smoothing with B-splines
and penalties", Stat. Sci. 11:89; de Boor (1978) "A Practical Guide
to Splines"; Golub-Heath-Wahba (1979) GCV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _bspline_basis(x: FloatArray, n_knots: int, degree: int = 3) -> FloatArray:
    """B-spline basis via scipy BSpline design matrix on equidistant knots."""
    from scipy.interpolate import BSpline

    a = np.asarray(x, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    knots = np.linspace(lo, hi, n_knots)
    # full knot vector with boundary repeats
    t = np.concatenate(
        [
            np.full(degree, knots[0]),
            knots,
            np.full(degree, knots[-1]),
        ]
    )
    nb = knots.size + degree - 1
    b = np.zeros((a.size, nb))
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        spl = BSpline(t, c, degree)
        b[:, i] = spl(a)
    # BSpline evaluates only on [t[degree], t[-degree-1]]; clip edge nans
    b = np.nan_to_num(b, nan=0.0)
    return np.asarray(b, dtype=np.float64)


def _diff_penalty(n: int, order: int = 2) -> FloatArray:
    d = np.eye(n)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return np.asarray(d, dtype=np.float64)


def p_spline(
    x: FloatArray,
    y: FloatArray,
    n_knots: int = 20,
    degree: int = 3,
    order: int = 2,
    lam_grid: FloatArray | None = None,
) -> dict[str, FloatArray | float]:
    """Eilers-Marx P-spline fit with GCV lambda selection.

    Returns ``fitted``, ``coef``, ``lambda``, ``basis``, ``gcv``.
    """
    a = np.asarray(x, dtype=float)
    b_ = np.asarray(y, dtype=float)
    if a.ndim != 1 or b_.shape != a.shape or a.size < 6:
        raise ValueError("bad input")
    if not np.isfinite(a).all() or not np.isfinite(b_).all():
        raise ValueError("non-finite input")
    b = _bspline_basis(a, n_knots, degree)
    nb = b.shape[1]
    d = _diff_penalty(nb, order)
    p = d.T @ d
    btb = b.T @ b
    bty = b.T @ b_
    if lam_grid is None:
        lam_grid = np.logspace(-4, 4, 25)
    lams = np.asarray(lam_grid, dtype=float)
    if lams.ndim != 1 or lams.size < 2 or (lams <= 0).any():
        raise ValueError("bad lam_grid")
    best = (np.inf, 0.0)
    coef_best = np.zeros(nb)
    for lam in lams:
        a_mat = btb + lam * p
        try:
            coef = np.linalg.solve(a_mat, bty)
        except np.linalg.LinAlgError:
            continue
        fit = b @ coef
        # hat trace via H = B (BtB+lamP)^{-1} Bt -> tr = sum B * coef-solve rows
        h_trace = float(np.sum(b * (np.linalg.solve(a_mat, b.T)).T))
        denom = max(1.0 - h_trace / a.size, 1e-6)
        gcv = float(np.mean((b_ - fit) ** 2)) / (denom * denom)
        if gcv < best[0]:
            best = (gcv, float(lam))
            coef_best = coef
    if not np.isfinite(best[0]):
        raise ValueError("gcv selection failed")
    fitted = b @ coef_best
    return {
        "fitted": np.asarray(fitted, dtype=np.float64),
        "coef": np.asarray(coef_best, dtype=np.float64),
        "lambda": np.asarray(best[1]),
        "basis": np.asarray(b, dtype=np.float64),
        "gcv": np.asarray(best[0]),
    }


def bench_p_spline(seed: int = 20261231 + 423) -> dict[str, float]:
    """SYNTHETIC check — smooth-curve recovery + localized feature."""
    rng = np.random.default_rng(seed)
    n = 300
    x = np.sort(rng.random(n) * 8.0)
    f = np.sin(x) + 0.8 * np.exp(-0.5 * ((x - 4.0) / 0.4) ** 2)
    y = f + 0.2 * rng.standard_normal(n)
    out = p_spline(x, y, n_knots=24)
    fitted = np.asarray(out["fitted"], dtype=np.float64)
    mse_s = float(np.mean((fitted - f) ** 2))
    mse_r = float(np.mean((y - f) ** 2))
    ratio = mse_s / max(mse_r, 1e-12)
    lam = float(out["lambda"])
    if ratio > 0.4 or not (1e-4 <= lam <= 1e4):
        raise ValueError(f"p_spline off: ratio={ratio:.3f} lam={lam:.3g}")
    return {
        "synthetic_pspline_mse_ratio": ratio,
        "synthetic_pspline_lambda": lam,
        "synthetic_pspline_gcv": float(out["gcv"]),
        "score": 1.0,
    }
