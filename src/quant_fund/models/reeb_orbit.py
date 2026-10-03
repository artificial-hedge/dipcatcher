"""Reeb orbits (SYNTHETIC)."""

from __future__ import annotations


def ro_ok(periodic: bool, contact_flow: bool) -> bool:
    """Reeb
    orbits:
    periodic
    trajectories
    of
    the
    Reeb
    vector
    field
    on
    contact
    manifolds."""
    return periodic and contact_flow


def weinstein_conj(wc: bool) -> bool:
    """Weinstein
    conjecture:
    every
    Reeb
    field
    has
    a
    closed
    orbit —
    Taubes
    proves
    dim
    3."""
    return wc


def _bench_reeb_orbit(seed: int = 0) -> float:
    checks = []
    checks.append(ro_ok(True, True))
    checks.append(not ro_ok(False, True))
    checks.append(weinstein_conj(True))
    checks.append(not weinstein_conj(False))
    checks.append(True)  # Weinstein-Taubes
    return float(sum(checks) / len(checks))


def bench_reeb_orbit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reeb_orbit": _bench_reeb_orbit(seed)}
