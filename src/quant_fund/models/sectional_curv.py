"""Sectional/Gaussian curvature K = det(II)/det(I) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def sphere_first_fundamental(theta: float) -> np.ndarray:
    return np.diag([1.0, float(np.sin(theta)) ** 2])


def sphere_second_fundamental(theta: float) -> np.ndarray:
    return np.diag([-1.0, -(float(np.sin(theta)) ** 2)])


def gauss_curvature(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.det(second) / np.linalg.det(first))


def mean_curvature(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.trace(np.linalg.solve(first, second)))


def _bench_sectional_curv(seed: int = 0) -> float:
    checks = []
    # unit sphere K = 1 everywhere
    for th in (0.3, 0.8, 1.4):
        k = gauss_curvature(sphere_first_fundamental(th), sphere_second_fundamental(th))
        checks.append(abs(k - 1.0) < 1e-9)
    # mean curvature of unit sphere = -1 (inward normal convention)
    checks.append(
        abs(mean_curvature(sphere_first_fundamental(0.9), sphere_second_fundamental(0.9)) + 1.0)
        < 1e-9
    )
    # flat cylinder (r=1): I = diag(1,1), II = diag(-1,0) -> K = 0
    checks.append(abs(gauss_curvature(np.eye(2), np.diag([-1.0, 0.0]))) < 1e-12)
    # hyperbolic paraboloid z = xy at origin: K = -1
    checks.append(abs(gauss_curvature(np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])) + 1.0) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_sectional_curv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sectional_curv": _bench_sectional_curv(seed)}
