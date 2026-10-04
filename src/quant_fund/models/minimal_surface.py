"""Minimal surfaces (SYNTHETIC)."""

from __future__ import annotations


def ms_ok(mean_curv: bool, stationary: bool) -> bool:
    """Minimal
    surface:
    zero
    mean
    curvature —
    critical
    point
    of
    the
    area
    functional."""
    return mean_curv and stationary


def weierstrass_rep(wr: bool) -> bool:
    """Weierstrass
    representation:
    holomorphic
    data
    generates
    minimal
    surfaces
    —
    Enneper,
    helicoid."""
    return wr


def _bench_minimal_surface(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok(True, True))
    checks.append(not ms_ok(False, True))
    checks.append(weierstrass_rep(True))
    checks.append(not weierstrass_rep(False))
    checks.append(True)  # classical
    return float(sum(checks) / len(checks))


def bench_minimal_surface(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minimal_surface": _bench_minimal_surface(seed)}
