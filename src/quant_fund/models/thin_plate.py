"""Thin-plate splines — 2D radial-basis surface smoothing.

Duchon (1977), Wahba (1990): the thin-plate spline interpolant is
the minimizer of the bending energy int |d^2 f|^2 over surfaces
through the data; in R^2 the radial kernel is U(r) = r^2 log r. With
ridge smoothing lam the system is

    [K + lam I  P] [c]   [y]
    [P^T       0 ] [d] = [0]

with P = [1, x1, x2] the polynomial (null-space) part.

Honesty: the bench fits a smooth surface (saddle + localized bump)
with noise and checks (a) held-out RMSE beats the marginal-sd
baseline, and (b) exact interpolation (lam=0) recovers the training
points to solver tolerance. The dense (n+3)x(n+3) solve is O(n^3) —
fixture sized accordingly. Fail-closed on duplicate coordinates or
degenerate designs.

References: Duchon (1977) "Splines minimizing rotation-invariant
semi-norms"; Wahba (1990) "Spline Models for Observational Data";
Wood (2003) thin-plate regression splines.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _u_kernel(d: FloatArray) -> FloatArray:
    """U(r) = r^2 log r (with U(0)=0)."""
    d2 = d * d
    out = np.where(d2 > 0, 0.5 * d2 * np.log(np.maximum(d2, 1e-300)), 0.0)
    return np.asarray(out, dtype=np.float64)


def thin_plate(xy: FloatArray, z: FloatArray, lam: float = 0.0) -> dict[str, FloatArray]:
    """Thin-plate spline solve for scattered 2D data.

    ``xy`` is (n,2) coordinates, ``z`` the (n,) responses. Returns
    ``coef_r`` (radial coefficients), ``coef_p`` (polynomial),
    ``knots``, and a ``predict`` closure evaluated via
    ``thin_plate_predict``.
    """
    a = np.asarray(xy, dtype=float)
    zz = np.asarray(z, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 4:
        raise ValueError("bad xy")
    if zz.shape != (a.shape[0],) or not np.isfinite(a).all() or not np.isfinite(zz).all():
        raise ValueError("bad z")
    # duplicates break the interpolation system
    if np.unique(a, axis=0).shape[0] != a.shape[0]:
        raise ValueError("duplicate coordinates")
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("bad lam")
    n = a.shape[0]
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    k = _u_kernel(d)
    # normalize kernel scale for conditioning
    scale = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    p = np.column_stack([np.ones(n), a])
    a_sys = np.zeros((n + 3, n + 3))
    a_sys[:n, :n] = k + lam * scale * np.eye(n)
    a_sys[:n, n:] = p
    a_sys[n:, :n] = p.T
    rhs = np.concatenate([zz, np.zeros(3)])
    try:
        sol = np.linalg.solve(a_sys, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("thin-plate system singular") from exc
    return {
        "coef_r": np.asarray(sol[:n], dtype=np.float64),
        "coef_p": np.asarray(sol[n:], dtype=np.float64),
        "knots": np.asarray(a, dtype=np.float64),
    }


def thin_plate_predict(fit: dict[str, FloatArray], xy: FloatArray) -> FloatArray:
    """Evaluate a fitted thin-plate spline at new coordinates."""
    a = np.asarray(xy, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("bad xy")
    knots = np.asarray(fit["knots"], dtype=np.float64)
    cr = np.asarray(fit["coef_r"], dtype=np.float64)
    cp = np.asarray(fit["coef_p"], dtype=np.float64)
    d = np.linalg.norm(a[:, None, :] - knots[None, :, :], axis=2)
    p = np.column_stack([np.ones(a.shape[0]), a])
    return np.asarray(_u_kernel(d) @ cr + p @ cp, dtype=np.float64)


def bench_thin_plate(seed: int = 20261231 + 424) -> dict[str, float]:
    """SYNTHETIC check — surface recovery + interpolation exactness."""
    rng = np.random.default_rng(seed)
    n = 120
    xy = rng.random((n, 2)) * 4.0
    z = (xy[:, 0] - 2.0) * (xy[:, 1] - 2.0) + 0.5 * np.exp(
        -(((xy[:, 0] - 2.5) ** 2 + (xy[:, 1] - 1.5) ** 2) / 0.15)
    )
    y = z + 0.05 * rng.standard_normal(n)
    fit = thin_plate(xy, y, lam=1e-4)
    test_xy = rng.random((200, 2)) * 4.0
    zt = (test_xy[:, 0] - 2.0) * (test_xy[:, 1] - 2.0) + 0.5 * np.exp(
        -(((test_xy[:, 0] - 2.5) ** 2 + (test_xy[:, 1] - 1.5) ** 2) / 0.15)
    )
    pred = thin_plate_predict(fit, test_xy)
    rmse = float(np.sqrt(np.mean((pred - zt) ** 2)))
    ratio = rmse / max(float(zt.std()), 1e-9)
    # exact interpolation at lam=0
    fit0 = thin_plate(xy, z, lam=0.0)
    interp_err = float(np.abs(thin_plate_predict(fit0, xy) - z).max())
    if ratio > 0.3 or interp_err > 1e-4:
        raise ValueError(f"tps off: ratio={ratio:.3f} interp={interp_err:.2e}")
    return {
        "synthetic_tps_rmse_ratio": ratio,
        "synthetic_tps_interp_err": interp_err,
        "score": 1.0,
    }
