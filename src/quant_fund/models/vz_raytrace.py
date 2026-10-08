"""Ray tracing in a linear-gradient V(z) = v0 + k*z medium (SYNTHETIC).

With ray parameter p = sin(theta)/v constant (theta from vertical), rays are
circular arcs of radius R = 1/(p k):
  x = x0 + R (cos t0 - cos t),  z = z0 + R (sin t - sin t0),
  t_time = (1/k) ln( tan(t/2) / tan(t0/2) ).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 937


def ray_parameter(v0: float, theta0: float) -> float:
    return float(np.sin(theta0) / v0)


def turning_depth(v0: float, k: float, theta0: float) -> float:
    v_turn = v0 / float(np.sin(theta0))
    return (v_turn - v0) / k


def trace_ray(
    v0: float,
    k: float,
    theta0: float,
    z0: float = 0.0,
    x0: float = 0.0,
    n_pts: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, z, t_time, theta) along the down-going arc up to theta=pi/2."""
    p = ray_parameter(v0, theta0)
    r = 1.0 / (p * k)
    th = np.linspace(theta0, np.pi / 2 - 1e-9, n_pts)
    x = x0 + r * (np.cos(theta0) - np.cos(th))
    z = z0 + r * (np.sin(th) - np.sin(theta0))
    t = (1.0 / k) * np.log(np.tan(th / 2.0) / np.tan(theta0 / 2.0))
    return x, z, t, th


def bench_vz_raytrace(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    v0 = 1800.0
    k = 0.6
    theta0 = float(rng.uniform(0.35, 1.1))
    x, z, t, th = trace_ray(v0, k, theta0)
    p = ray_parameter(v0, theta0)
    v_z = v0 + k * z
    snell_err = float(np.max(np.abs(np.sin(th) / v_z - p)))
    z_turn = turning_depth(v0, k, theta0)
    depth_err = abs(float(z[-1]) - z_turn) / z_turn
    r = 1.0 / (p * k)
    xc = x[0] + r * np.cos(theta0)
    zc = -r * np.sin(theta0)
    rad = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    radius_err = float(np.max(np.abs(rad - r)) / r)
    monot = float(np.all(np.diff(t) > 0))
    t_analytic = (1.0 / k) * np.log(np.tan((np.pi / 2 - 1e-9) / 2.0) / np.tan(theta0 / 2.0))
    t_err = abs(t[-1] - t_analytic) / t_analytic
    checks = [
        snell_err < 1e-9,
        depth_err < 1e-6,
        radius_err < 1e-9,
        monot == 1.0,
        t_err < 1e-9,
        z[-1] > z[0],
        x[-1] > x[0],
    ]
    return {"synthetic_vz_raytrace": float(np.mean(checks))}
