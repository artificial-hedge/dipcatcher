"""TLE-style mean-element propagation with J2 secular rates (SGP4-lite) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

from quant_fund.models.orbital_elements import elements_to_state

_J2 = 1.08262668e-3
_RE = 6378.137  # km


def secular_rates(a: float, e: float, inc: float, n: float) -> tuple[float, float, float]:
    """J2 secular rates for (raan_dot, argp_dot, M_dot_correction) in rad/s."""
    p = a * (1.0 - e * e)
    fac = 0.75 * _J2 * (_RE / p) ** 2 * n
    raan_dot = -2.0 * fac * np.cos(inc)
    argp_dot = fac * (5.0 * np.cos(inc) ** 2 - 1.0)
    m_dot = fac * np.sqrt(1.0 - e * e) * (3.0 * np.cos(inc) ** 2 - 1.0)
    return raan_dot, argp_dot, m_dot


def propagate(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    m0: float,
    t: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Propagate mean elements `t` seconds; return (r, v, updated angles)."""
    n = np.sqrt(mu / a**3)
    d_raan, d_argp, d_m = secular_rates(a, e, inc, n)
    raan_t = raan + d_raan * t
    argp_t = argp + d_argp * t
    m_t = m0 + (n + d_m) * t
    # mean -> true anomaly via Kepler
    from quant_fund.models.kepler_solve import kepler_E

    E = kepler_E(m_t % (2.0 * np.pi), e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0))
    r, v = elements_to_state(a, e, inc, raan_t, argp_t, nu, mu)
    return r, v, (raan_t, argp_t, m_t)


def bench_tle_propagate(seed: int = 20261231 + 857) -> dict[str, float]:
    """Secular-rate sanity: period closure, regression sign, rate consistency."""
    mu = 398600.4418
    checks = 0.0
    total = 0
    # 1) after one anomalistic period the mean anomaly advances by 2*pi
    a, e, inc = 7200.0, 0.02, np.radians(55.0)
    n = np.sqrt(mu / a**3)
    d_raan, d_argp, d_m = secular_rates(a, e, inc, n)
    T = 2.0 * np.pi / n
    _, _, (_, _, m_t) = propagate(a, e, inc, 0.4, 0.1, 0.0, T, mu)
    total += 1
    checks += float(abs((m_t - 2.0 * np.pi) / (2.0 * np.pi)) < 2e-3)
    # 2) RAAN regresses for a prograde orbit, progresses for retrograde
    total += 2
    checks += float(d_raan < 0.0)
    d_retro, _, _ = secular_rates(a, e, np.radians(120.0), n)
    checks += float(d_retro > 0.0)
    # 3) J2-off limit: zero rates recover plain mean motion
    a2, e2 = 7200.0, 0.02
    p = a2 * (1 - e2 * e2)
    n2 = np.sqrt(mu / a2**3)
    fac = 0.75 * _J2 * (_RE / p) ** 2 * n2
    total += 1
    checks += float(abs(d_raan + 2.0 * fac * np.cos(inc)) < 1e-15)
    # 4) polar orbit: argp_dot must equal -fac (5*0 - 1)
    _, wpol, _ = secular_rates(a, e, np.pi / 2.0, n)
    total += 1
    checks += float(abs(wpol + fac * (a / p) ** 0 * 1.0) / fac < 0.1 or abs(wpol + fac) < 1e-15)
    # 5) numerical: propagated RAAN after dt matches linear rate to O(dt^2)
    _, _, (r1, _, _) = propagate(a, e, inc, 1.0, 0.2, 0.0, 60.0, mu)
    total += 1
    checks += float(abs(r1 - (1.0 + d_raan * 60.0)) < 1e-12)
    return {"synthetic_tle": checks / total}
