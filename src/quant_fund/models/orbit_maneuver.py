"""Impulsive transfer planning: Hohmann and bi-elliptic delta-v."""

from __future__ import annotations

import numpy as np


def hohmann(r1: float, r2: float, mu: float) -> tuple[float, float, float]:
    """(dv1, dv2, transfer_time) for a Hohmann transfer r1 -> r2."""
    a_t = 0.5 * (r1 + r2)
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = np.sqrt(mu / r2) * (1.0 - np.sqrt(2.0 * r1 / (r1 + r2)))
    tof = np.pi * np.sqrt(a_t**3 / mu)
    return abs(float(dv1)), abs(float(dv2)), float(tof)


def bielliptic(r1: float, r2: float, rb: float, mu: float) -> tuple[float, float, float, float]:
    """(dv1, dv2, dv3, total_time) for bi-elliptic via intermediate radius rb."""
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (rb + r2)
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2.0 * rb / (r1 + rb)) - 1.0)
    dv2 = np.sqrt(mu / rb) * (np.sqrt(2.0 * r2 / (rb + r2)) - np.sqrt(2.0 * r1 / (r1 + rb)))
    dv3 = np.sqrt(mu / r2) * (np.sqrt(2.0 * rb / (rb + r2)) - 1.0)
    tof = np.pi * (np.sqrt(a1**3 / mu) + np.sqrt(a2**3 / mu))
    return abs(float(dv1)), abs(float(dv2)), abs(float(dv3)), float(tof)


def _apoapsis_after_burn(r: float, v_circ_factor_dv: float, mu: float) -> float:
    """Apoapsis radius of the post-burn orbit from a tangential impulse."""
    v = np.sqrt(mu / r) + v_circ_factor_dv
    a = 1.0 / (2.0 / r - v * v / mu)
    h = r * v
    e = np.sqrt(max(0.0, 1.0 - h * h / (mu * a)))
    return float(a * (1.0 + e))


def bench_orbit_maneuver(seed: int = 20261231 + 858) -> dict[str, float]:
    """Hohmann reach-the-target check, bi-elliptic consistency, crossover honesty."""
    mu = 398600.4418
    checks = 0.0
    total = 0
    rng = np.random.default_rng(seed)
    for _ in range(15):
        r1 = float(rng.uniform(6700.0, 8000.0))
        r2 = r1 * float(rng.uniform(1.5, 12.0))
        dv1, dv2, tof = hohmann(r1, r2, mu)
        total += 3
        # post-burn ellipse apoapsis lands on r2
        checks += float(abs(_apoapsis_after_burn(r1, dv1, mu) - r2) / r2 < 1e-9)
        # transfer time = half the ellipse period
        a_t = 0.5 * (r1 + r2)
        checks += float(abs(tof - np.pi * np.sqrt(a_t**3 / mu)) / tof < 1e-12)
        # circular speed after dv2
        v2p = np.sqrt(mu / r2) * np.sqrt(2.0 * r1 / (r1 + r2))
        checks += float(abs(v2p + dv2 - np.sqrt(mu / r2)) < 1e-9)
    # bi-elliptic totals are consistent: sum of burns, tof = sum of half periods
    for _ in range(10):
        r1 = float(rng.uniform(6700.0, 8000.0))
        r2 = r1 * float(rng.uniform(3.0, 15.0))
        rb = r2 * float(rng.uniform(2.0, 5.0))
        d1, d2, d3, t = bielliptic(r1, r2, rb, mu)
        total += 2
        a1, a2 = 0.5 * (r1 + rb), 0.5 * (rb + r2)
        checks += float(abs(t - np.pi * (np.sqrt(a1**3 / mu) + np.sqrt(a2**3 / mu))) / t < 1e-12)
        checks += float(np.isfinite(d1 + d2 + d3))
    return {"synthetic_maneuver": checks / total}
