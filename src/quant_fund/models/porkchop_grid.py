"""Porkchop-plot C3 grid via Lambert solutions between two body orbits.

SYNTHETIC bench: two coplanar circular heliocentric orbits (Earth-like,
Mars-like); computes C3 over a departure/arrival grid and verifies the
minimum falls near the Hohmann ideal and C3>=0 everywhere.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from quant_fund.models.lambert_problem import lambert

_SEED = 20261231 + 924
_MU_SUN = 1.32712440018e11  # km^3/s^2
_AU = 1.495978707e8  # km


def _body_state(a: float, theta: float, mu: float = _MU_SUN) -> tuple[np.ndarray, np.ndarray]:
    r = a * np.asarray([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(mu / a) * np.asarray([-np.sin(theta), np.cos(theta), 0.0])
    return r, v


def porkchop(
    a1: float,
    a2: float,
    theta1_0: float,
    theta2_0: float,
    dep_days: np.ndarray,
    tof_days: np.ndarray,
    mu: float = _MU_SUN,
) -> np.ndarray:
    """C3 (km^2/s^2) grid: dep × tof. Bodies on circular coplanar orbits."""
    n1 = np.sqrt(mu / a1**3)
    n2 = np.sqrt(mu / a2**3)
    c3 = np.full((len(dep_days), len(tof_days)), np.inf)
    for i, dep in enumerate(dep_days):
        r1, v1 = _body_state(a1, theta1_0 + n1 * dep * 86400.0, mu)
        for j, tof in enumerate(tof_days):
            r2, _ = _body_state(a2, theta2_0 + n2 * (dep + tof) * 86400.0, mu)
            try:
                vt, _ = lambert(r1, r2, tof * 86400.0, mu)
            except RuntimeError:
                try:
                    vt, _ = lambert(r1, r2, tof * 86400.0, mu, tm=-1)
                except RuntimeError:
                    continue
            vinf = float(norm(vt - v1))
            c3[i, j] = vinf * vinf
    return c3


def bench_porkchop_grid(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    a1, a2 = _AU, 1.524 * _AU
    # phase angle for a nominal transfer at dep=0: place Mars ahead by the
    # Hohmann lead angle  pi(1 - (tof/T2))  with tof ~ 259 d
    n2 = np.sqrt(_MU_SUN / a2**3)
    tof_h = np.pi * np.sqrt(((a1 + a2) / 2) ** 3 / _MU_SUN) / 86400.0
    lead = np.pi - n2 * tof_h * 86400.0
    dep_days = np.linspace(-30.0, 30.0, 13)
    tof_days = np.linspace(150.0, 360.0, 15)
    c3 = porkchop(a1, a2, 0.0, lead, dep_days, tof_days)
    finite = np.isfinite(c3)
    score += 1.0 if finite.mean() > 0.9 else 0.0
    score += 1.0 if float(c3[finite].min()) >= 0.0 else 0.0
    # min near ideal Hohmann C3
    rE = np.asarray([a1, 0.0, 0.0])
    vE = np.asarray([0.0, np.sqrt(_MU_SUN / a1), 0.0])
    # slight offset from exact antiparallel to keep the Lambert solve non-singular
    rM = a2 * np.asarray([np.cos(np.pi - 0.02), np.sin(np.pi - 0.02), 0.0])
    v1t, _ = lambert(rE, rM, tof_h * 86400.0, _MU_SUN)
    c3h = float(norm(v1t - vE) ** 2)
    i, j = np.unravel_index(int(np.nanargmin(np.where(finite, c3, np.inf))), c3.shape)
    score += 1.0 if abs(c3[i, j] - c3h) / max(c3h, 1e-9) < 0.35 else 0.0
    score += 1.0 if abs(tof_days[j] - tof_h) < 40.0 else 0.0
    rng.random()
    return {"synthetic_porkchop_grid": score / 4.0}
