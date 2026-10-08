"""Universal-variable Lambert solver (Bate-Mueller-White formulation) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _stumpff(z: float) -> tuple[float, float]:
    if z > 1e-6:
        sz = np.sqrt(z)
        return (1.0 - np.cos(sz)) / z, (sz - np.sin(sz)) / sz**3
    if z < -1e-6:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1.0) / (-z), (np.sinh(sz) - sz) / sz**3
    return 0.5, 1.0 / 6.0


def lambert(
    r1: np.ndarray, r2: np.ndarray, tof: float, mu: float, tm: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (v1, v2) for the transfer from r1 to r2 in `tof` seconds.

    `tm` = +1 short-way transfer, -1 long-way.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1m, r2m = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    cos_dnu = float(np.dot(r1, r2) / (r1m * r2m))
    cross = np.cross(r1, r2)
    sgn = np.sign(cross[2]) if abs(cross[2]) > 1e-12 else 1.0
    sin_dnu = tm * np.sqrt(max(0.0, 1.0 - cos_dnu * cos_dnu)) * sgn
    A = sin_dnu * np.sqrt(r1m * r2m / (1.0 - cos_dnu))

    def _y(z: float) -> float:
        C, S = _stumpff(z)
        return float(r1m + r2m + A * (z * S - 1.0) / np.sqrt(C))

    def _tof_z(z: float) -> float:
        C, S = _stumpff(z)
        y = _y(z)
        x = np.sqrt(y / C)
        return float((x**3 * S + A * np.sqrt(y)) / np.sqrt(mu))

    z = 0.0
    for _ in range(200):
        F = _tof_z(z) - tof
        if abs(F) < 1e-7:
            break
        h = 1e-4
        dz = (_tof_z(z + h) - _tof_z(z)) / h
        if dz == 0.0:
            z += 0.1
            continue
        z -= F / dz
    else:  # pragma: no cover
        raise RuntimeError("lambert did not converge")

    y = _y(z)
    f = 1.0 - y / r1m
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2m
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


def _propagate_rk4(
    r0: np.ndarray, v0: np.ndarray, t: float, mu: float, steps: int = 400
) -> np.ndarray:
    """Two-body propagation oracle via fixed-step RK4."""

    def acc(r: np.ndarray) -> np.ndarray:
        return -mu * r / float(np.linalg.norm(r)) ** 3

    def step(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        k1v, k1r = acc(r), v
        k2v, k2r = acc(r + k1r * dt / 2.0), v + k1v * dt / 2.0
        k3v, k3r = acc(r + k2r * dt / 2.0), v + k2v * dt / 2.0
        k4v, k4r = acc(r + k3r * dt), v + k3v * dt
        return r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r), v + dt / 6.0 * (
            k1v + 2 * k2v + 2 * k3v + k4v
        )

    r, v = np.asarray(r0, float), np.asarray(v0, float)
    dt = t / steps
    for _ in range(steps):
        r, v = step(r, v, dt)
    return r


def bench_lambert_problem(seed: int = 20261231 + 856) -> dict[str, float]:
    """Solve Lambert transfers; oracle = independent RK4 two-body propagation."""
    from quant_fund.models.orbital_elements import elements_to_state

    rng = np.random.default_rng(seed)
    mu = 398600.4418
    checks = 0.0
    total = 0
    for _ in range(12):
        a = float(rng.uniform(8000.0, 15000.0))
        e = float(rng.uniform(0.0, 0.3))
        inc = float(rng.uniform(0.0, 0.6))
        nu1 = float(rng.uniform(0.0, 2.0 * np.pi))
        dnu = float(rng.uniform(0.4, 2.5))
        nu2 = nu1 + dnu
        r1, _ = elements_to_state(a, e, inc, 0.3, 0.2, nu1, mu)
        r2, _ = elements_to_state(a, e, inc, 0.3, 0.2, nu2, mu)
        p = a * (1.0 - e * e)
        tof = float(rng.uniform(0.3, 0.8)) * 2.0 * np.pi * np.sqrt(p**3 / mu) / (1.0 - e * e) ** 1.5
        try:
            v1, _v2 = lambert(r1, r2, tof, mu, tm=1)
        except RuntimeError:  # pragma: no cover
            total += 1
            continue
        r_arr = _propagate_rk4(r1, v1, tof, mu)
        total += 2
        checks += float(np.linalg.norm(r_arr - r2) / np.linalg.norm(r2) < 1e-4)
        energy = np.dot(v1, v1) / 2.0 - mu / np.linalg.norm(r1)
        checks += float(energy < 0.0)
    return {"synthetic_lambert": checks / total}
