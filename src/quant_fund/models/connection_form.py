"""Levi-Civita connection 1-forms on the sphere (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def sphere_metric(theta: float) -> tuple[float, float]:
    """g_theta_theta = 1, g_phi_phi = sin^2(theta) on unit sphere."""
    return 1.0, float(np.sin(theta) ** 2)


def christoffel_sphere(theta: float) -> dict[tuple[int, int, int], float]:
    """Nonzero Christoffels of the unit sphere in (theta, phi) coords.

    Gamma^theta_{phi phi} = -sin cos, Gamma^phi_{theta phi} = cot.
    """
    st, ct = float(np.sin(theta)), float(np.cos(theta))
    return {
        (0, 1, 1): -st * ct,
        (1, 0, 1): ct / st,
        (1, 1, 0): ct / st,
    }


def covariant_derivative_metric(theta: float, eps: float = 1e-6) -> np.ndarray:
    """Numerically verify nabla_k g_ij = 0 via dg_ij/dx_k - Gamma_ki^l g_lj - Gamma_kj^l g_li."""
    res = np.zeros((2, 2, 2))
    g = np.diag(sphere_metric(theta))
    dgd = np.zeros((2, 2, 2))
    gp = np.diag(sphere_metric(theta + eps))
    gm = np.diag(sphere_metric(theta - eps))
    for i in (0, 1):
        for j in (0, 1):
            dgd[0, i, j] = (gp[i, j] - gm[i, j]) / (2 * eps)
    ga = christoffel_sphere(theta)
    for k in (0, 1):
        for i in (0, 1):
            for j in (0, 1):
                val = dgd[k, i, j]
                for m in (0, 1):
                    val -= ga.get((m, k, i), 0.0) * g[m, j] + ga.get((m, k, j), 0.0) * g[i, m]
                res[k, i, j] = val
    return res


def _bench_connection_form(seed: int = 0) -> float:
    checks = []
    ga = christoffel_sphere(0.7)
    st, ct = float(np.sin(0.7)), float(np.cos(0.7))
    checks.append(abs(ga[(0, 1, 1)] - (-st * ct)) < 1e-12)
    checks.append(abs(ga[(1, 0, 1)] - ct / st) < 1e-12)
    checks.append(abs(ga[(1, 1, 0)] - ct / st) < 1e-12)
    # at the equator Gamma^theta_{phiphi} = 0
    checks.append(abs(christoffel_sphere(np.pi / 2)[(0, 1, 1)]) < 1e-12)
    # metric compatibility: nabla g = 0 numerically
    checks.append(float(np.max(np.abs(covariant_derivative_metric(0.9)))) < 1e-4)
    return float(sum(checks) / len(checks))


def bench_connection_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connection_form": _bench_connection_form(seed)}
