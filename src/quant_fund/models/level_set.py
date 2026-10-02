"""Level-set canon: signed-distance reinitialization (Sussman–
Smereka–Osher pseudo-time iteration) and curvature-driven level-
set evolution on a 2-D grid — a circle initialized as a noisy
signed distance is re-distanced, and a two-blob interface is
evolved by mean-curvature motion. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sussman_sign(phi0: FloatArray, dx: float) -> FloatArray:
    """Smoothed initial sign sgn(φ0)·|φ0|/sqrt(φ0²+dx²)."""
    return np.asarray(phi0 / np.sqrt(phi0**2 + dx**2), dtype=np.float64)


def grad_mag(phi: FloatArray, dx: float) -> FloatArray:
    """|∇φ| by central differences (for diagnostics)."""
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
    gy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    return np.asarray(np.sqrt(gx**2 + gy**2), dtype=np.float64)


def _godunov_grad(phi: FloatArray, dx: float, sgn: FloatArray) -> FloatArray:
    """Upwind Godunov |∇φ| for the eikonal equation — picks the
    correct one-sided differences by the sign of φ0."""
    p = np.zeros_like(phi)
    p[1:, :] = (phi[1:, :] - phi[:-1, :]) / dx  # backward x
    q = np.zeros_like(phi)
    q[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dx  # forward x
    r = np.zeros_like(phi)
    r[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx  # backward y
    s = np.zeros_like(phi)
    s[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx  # forward y
    g = np.zeros_like(phi)
    pos = sgn >= 0
    gm_pos = np.sqrt(
        np.maximum(np.maximum(p, 0) ** 2, np.minimum(q, 0) ** 2)
        + np.maximum(np.maximum(r, 0) ** 2, np.minimum(s, 0) ** 2)
    )
    # for sgn<0: G = sqrt(max(a⁻², b⁺²) + max(c⁻², d⁺²))
    gm_neg = np.sqrt(
        np.maximum(np.minimum(p, 0) ** 2, np.maximum(q, 0) ** 2)
        + np.maximum(np.minimum(r, 0) ** 2, np.maximum(s, 0) ** 2)
    )
    g[pos] = gm_pos[pos]
    g[~pos] = gm_neg[~pos]
    return np.asarray(g, dtype=np.float64)


def reinitialize(
    phi0: FloatArray,
    dx: float,
    iters: int = 20,
    dtau: float | None = None,
) -> FloatArray:
    """Sussman–Smereka–Osher reinitialization:

    φ_τ + S(φ0)(|∇φ| − 1) = 0 — pseudo-time marching with the
    upwind Godunov gradient pulls |∇φ| toward 1 while freezing
    the zero level set.
    """
    phi = phi0.copy()
    sgn = sussman_sign(phi0, dx)
    dt = dtau if dtau is not None else 0.3 * dx
    for _ in range(iters):
        gm = _godunov_grad(phi, dx, sgn)
        phi -= dt * sgn * (gm - 1.0)
    return np.asarray(phi, dtype=np.float64)


def curvature_flow(
    phi0: FloatArray,
    dx: float,
    dt: float,
    steps: int,
    eps: float = 1e-10,
) -> FloatArray:
    """Mean-curvature motion φ_t = κ |∇φ| (upwind-free explicit
    step on the 4-point curvature stencil, small dt)."""
    phi = phi0.copy()
    for _ in range(steps):
        gx = np.zeros_like(phi)
        gy = np.zeros_like(phi)
        gx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
        gy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
        gxx = np.zeros_like(phi)
        gyy = np.zeros_like(phi)
        gxy = np.zeros_like(phi)
        gxx[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / dx**2
        gyy[:, 1:-1] = (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, :-2]) / dx**2
        gxy[1:-1, 1:-1] = (phi[2:, 2:] - phi[2:, :-2] - phi[:-2, 2:] + phi[:-2, :-2]) / (4 * dx**2)
        denom = gx**2 + gy**2 + eps
        kappa = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / denom**1.5
        phi[1:-1, 1:-1] += dt * kappa[1:-1, 1:-1] * np.sqrt(denom[1:-1, 1:-1])
    return np.asarray(phi, dtype=np.float64)


def bench_level_set(seed: int = 20261231) -> dict[str, float]:
    """Reinitialize a distorted circle, then shrink it by curvature
    flow — signed-distance and area-loss checks."""
    out: dict[str, float] = {}
    n = 64
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    dx = x[1] - x[0]
    r0 = 0.5
    # distorted signed distance: true SDF + perturbation
    sdf = np.sqrt(xx**2 + yy**2) - r0
    rng = np.random.default_rng(seed)
    phi0 = sdf + 0.15 * np.sin(4 * xx) * np.cos(4 * yy) + rng.standard_normal((n, n)) * 0.02
    gm_before = float(np.abs(grad_mag(phi0, dx) - 1).mean())
    phi_r = reinitialize(phi0, dx, iters=30)
    gm_after = float(np.abs(grad_mag(phi_r, dx)[2:-2, 2:-2] - 1).mean())
    out["synthetic_grad_mag_before"] = gm_before
    out["synthetic_grad_mag_after"] = gm_after
    out["synthetic_grad_improvement"] = gm_before / max(gm_after, 1e-30)
    # zero contour should not move much: compare sign patterns
    sign_drift = float(np.mean((phi_r > 0) != (sdf > 0)))
    out["synthetic_zero_contour_drift"] = sign_drift
    # curvature flow on a clean SDF: radius shrinks ~sqrt over time
    phi_c = curvature_flow(sdf, dx, dt=0.2 * dx**2, steps=40)
    out["synthetic_curv_area_start"] = float(np.mean(sdf < 0))
    out["synthetic_curv_area_end"] = float(np.mean(phi_c < 0))
    out["synthetic_curv_area_ratio"] = out["synthetic_curv_area_end"] / max(
        out["synthetic_curv_area_start"], 1e-30
    )
    return out
