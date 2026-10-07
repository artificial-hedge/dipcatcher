"""SE(3) pose interpolation via Lie-group exp/log maps (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """xi = [rho(3), phi(3)] -> 4x4 SE(3) matrix."""
    xi = np.asarray(xi, float)
    rho, phi = xi[:3], xi[3:]
    th = float(np.linalg.norm(phi))
    W = _skew(phi)
    if th < 1e-12:
        R = np.eye(3) + W
        V = np.eye(3)
    else:
        R = np.eye(3) + np.sin(th) / th * W + (1.0 - np.cos(th)) / th**2 * (W @ W)
        V = np.eye(3) + (1.0 - np.cos(th)) / th**2 * W + (th - np.sin(th)) / th**3 * (W @ W)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) matrix -> xi = [rho, phi]."""
    R = T[:3, :3]
    p = T[:3, 3]
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if th < 1e-9:
        return np.asarray(np.r_[p, np.zeros(3)])
    W = th / (2.0 * np.sin(th)) * (R - R.T)
    phi = np.array([W[2, 1], W[0, 2], W[1, 0]])
    Wn = _skew(phi)
    Vinv = (
        np.eye(3)
        - 0.5 * Wn
        + (1.0 / th**2) * (1.0 - th * np.sin(th) / (2.0 * (1.0 - np.cos(th)))) * (Wn @ Wn)
    )
    rho = Vinv @ p
    return np.asarray(np.r_[rho, phi])


def interpolate(T1: np.ndarray, T2: np.ndarray, s: float) -> np.ndarray:
    """Geodesic interpolation: T1 @ exp(s * log(T1^{-1} T2))."""
    xi = log_se3(np.linalg.inv(T1) @ T2)
    return np.asarray(T1 @ exp_se3(s * xi))


def _geodesic_dist(T1: np.ndarray, T2: np.ndarray) -> float:
    return float(np.linalg.norm(log_se3(np.linalg.inv(T1) @ T2)))


def bench_se3_spline(seed: int = 20261231 + 865) -> dict[str, float]:
    """Endpoints exact, midpoint equidistant, exp/log round-trip."""
    rng = np.random.default_rng(seed)
    checks = 0.0
    total = 0
    for _ in range(20):
        xi1 = rng.uniform(-1.0, 1.0, 6)
        xi2 = rng.uniform(-1.0, 1.0, 6)
        T1, T2 = exp_se3(xi1), exp_se3(xi2)
        total += 4
        checks += float(np.allclose(log_se3(exp_se3(xi1)), xi1, atol=1e-8))
        checks += float(np.allclose(interpolate(T1, T2, 0.0), T1, atol=1e-9))
        checks += float(np.allclose(interpolate(T1, T2, 1.0), T2, atol=1e-9))
        Tm = interpolate(T1, T2, 0.5)
        d1 = _geodesic_dist(T1, Tm)
        d2 = _geodesic_dist(Tm, T2)
        checks += float(abs(d1 - d2) < 1e-8 * max(1.0, d1))
    return {"synthetic_se3": checks / total}
