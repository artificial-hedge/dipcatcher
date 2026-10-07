"""Cartesian state <-> classical Keplerian elements (two-body) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def state_to_elements(
    r: np.ndarray, v: np.ndarray, mu: float
) -> tuple[float, float, float, float, float, float]:
    """Return (a, e, i, raan, argp, nu); angles in radians."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rm = float(np.linalg.norm(r))
    vm = float(np.linalg.norm(v))
    h = np.cross(r, v)
    hm = float(np.linalg.norm(h))
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nm = float(np.linalg.norm(n))
    ev = np.cross(v, h) / mu - r / rm
    e = float(np.linalg.norm(ev))
    a = 1.0 / (2.0 / rm - vm * vm / mu)
    inc = float(np.arccos(np.clip(h[2] / hm, -1.0, 1.0)))

    def _ang(x: float, y: float, quadrant_ok: bool) -> float:
        base = float(np.arccos(np.clip(x, -1.0, 1.0)))
        return base if quadrant_ok else _TWO_PI - base

    if nm > 1e-9:
        raan = _ang(n[0] / nm, n[1], n[1] >= 0.0)
        argp = _ang(float(np.dot(n, ev)) / (nm * e), ev[2], ev[2] >= 0.0)
        nu = _ang(float(np.dot(ev, r)) / (e * rm), float(np.dot(r, v)), float(np.dot(r, v)) >= 0.0)
    else:  # equatorial: measure angles from +x
        raan = 0.0
        argp = _ang(ev[0] / e, ev[1], ev[1] >= 0.0)
        nu = _ang(float(np.dot(ev, r)) / (e * rm), float(np.dot(r, v)), float(np.dot(r, v)) >= 0.0)
    return a, e, inc, raan, argp, nu


def _rot(axis: int, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    if axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def elements_to_state(
    a: float, e: float, inc: float, raan: float, argp: float, nu: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keplerian elements -> (r, v) in the inertial frame."""
    p = a * (1.0 - e * e)
    den = 1.0 + e * np.cos(nu)
    r_pf = np.array([p * np.cos(nu) / den, p * np.sin(nu) / den, 0.0])
    v_pf = np.array([-np.sqrt(mu / p) * np.sin(nu), np.sqrt(mu / p) * (e + np.cos(nu)), 0.0])
    q = _rot(2, raan) @ _rot(0, inc) @ _rot(2, argp)
    return q @ r_pf, q @ v_pf


def bench_orbital_elements(seed: int = 20261231 + 854) -> dict[str, float]:
    """Random elliptical inclined orbits round-trip through both maps."""
    rng = np.random.default_rng(seed)
    mu = 398600.4418
    checks = 0.0
    total = 0
    for _ in range(30):
        a = float(rng.uniform(8000.0, 30000.0))
        e = float(rng.uniform(0.05, 0.6))
        inc = float(rng.uniform(0.05, 2.5))
        raan = float(rng.uniform(0.0, _TWO_PI))
        argp = float(rng.uniform(0.0, _TWO_PI))
        nu = float(rng.uniform(0.0, _TWO_PI))
        r, v = elements_to_state(a, e, inc, raan, argp, nu, mu)
        a2, e2, i2, o2, w2, n2 = state_to_elements(r, v, mu)
        total += 6
        checks += float(abs(a2 - a) / a < 1e-9)
        checks += float(abs(e2 - e) < 1e-9)
        checks += float(abs(i2 - inc) < 1e-9)
        for got, want in ((o2, raan), (w2, argp), (n2, nu)):
            d = abs((got - want + np.pi) % _TWO_PI - np.pi)
            checks += float(d < 1e-9)
    return {"synthetic_orbital_elements": checks / total}
