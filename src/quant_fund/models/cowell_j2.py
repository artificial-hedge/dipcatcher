"""Cowell numerical propagation with J2 perturbation.

SYNTHETIC bench: verifies energy conservation without J2, first-order
secular RAAN regression rate against the analytical J2 formula, and
osculating-element sanity over one rev.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_SEED = 20261231 + 921
_MU = 398600.4418
_RE = 6378.137
_J2 = 1.08262668e-3


def accel(
    r: np.ndarray, v: np.ndarray, mu: float = _MU, j2: float = _J2, re: float = _RE
) -> np.ndarray:
    rm = float(norm(r))
    a = -mu * r / rm**3
    if j2 != 0.0:
        z2 = r[2] * r[2]
        a = a + 1.5 * j2 * mu * re * re / rm**5 * np.asarray(
            [
                r[0] * (5.0 * z2 / rm**2 - 1.0),
                r[1] * (5.0 * z2 / rm**2 - 1.0),
                r[2] * (5.0 * z2 / rm**2 - 3.0),
            ]
        )
    return np.asarray(a, dtype=np.float64)


def propagate(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    mu: float = _MU,
    j2: float = _J2,
    nstep: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    h = dt / max(1, nstep)

    def f(y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[3:], accel(y[:3], y[3:], mu, j2)])

    for _ in range(max(1, nstep)):
        y = np.concatenate([r, v])
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        yn = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        r, v = yn[:3], yn[3:]
    return r, v


def bench_cowell_j2(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # slightly eccentric orbit (e ~ 0.03) to keep angles well-posed
    r0 = np.asarray([6800.0, 0.0, 0.0])
    a_true = 6900.0
    inc = 0.5
    vmag = np.sqrt(_MU * (2.0 / 6800.0 - 1.0 / a_true))
    v0 = np.asarray([0.0, vmag * np.cos(inc), vmag * np.sin(inc)])
    T = 2 * np.pi * np.sqrt(a_true**3 / _MU)
    # 1) energy conserved without J2 over 1 rev
    r1, v1 = propagate(r0, v0, T, j2=0.0)
    E0 = 0.5 * float(v0 @ v0) - _MU / 6800.0
    E1 = 0.5 * float(v1 @ v1) - _MU / float(norm(r1))
    score += 1.0 if abs(E1 - E0) / abs(E0) < 1e-8 else 0.0
    # 2) RAAN regresses: compare vs analytical -1.5 J2 (Re/p)^2 n cos i
    from quant_fund.models.orbital_elements import state_to_elements

    _, _, _, raan0, _, _ = state_to_elements(r0, v0, _MU)
    r2, v2 = propagate(r0, v0, 4 * T, j2=_J2)
    _, _, _, raan2, _, _ = state_to_elements(r2, v2, _MU)
    p = a_true
    n = np.sqrt(_MU / a_true**3)
    pred = -1.5 * _J2 * (_RE / p) ** 2 * n * np.cos(inc) * 4 * T
    draan = (raan2 - raan0 + np.pi) % (2 * np.pi) - np.pi
    score += 1.0 if abs(draan - pred) / abs(pred) < 0.15 else 0.0
    # 3) J2 perturbed orbit stays bound at similar semi-major axis
    a2, _, _, _, _, _ = state_to_elements(r2, v2, _MU)
    score += 1.0 if abs(a2 - a_true) / a_true < 0.01 else 0.0
    # 4) z-oscillation: inclination stays put (J2 secular in RAAN only)
    _, _, i2, _, _, _ = state_to_elements(r2, v2, _MU)
    score += 1.0 if abs(i2 - inc) < 0.02 else 0.0
    rng.random()
    return {"synthetic_cowell_j2": score / 4.0}
