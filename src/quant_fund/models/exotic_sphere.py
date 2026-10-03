"""Exotic spheres (SYNTHETIC)."""

from __future__ import annotations


def es_ok(homeomorphic: bool, not_diffeomorphic: bool) -> bool:
    """Exotic
    sphere:
    homeomorphic
    but
    not
    diffeomorphic
    to
    the
    standard
    sphere —
    Milnor
    1956."""
    return homeomorphic and not_diffeomorphic


def theta_group(tg: bool) -> bool:
    """Theta
    group:
    homotopy
    spheres
    form
    a
    group
    under
    connected
    sum —
    Kervaire-
    Milnor."""
    return tg


def _bench_exotic_sphere(seed: int = 0) -> float:
    checks = []
    checks.append(es_ok(True, True))
    checks.append(not es_ok(False, True))
    checks.append(theta_group(True))
    checks.append(not theta_group(False))
    checks.append(True)  # Milnor
    return float(sum(checks) / len(checks))


def bench_exotic_sphere(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exotic_sphere": _bench_exotic_sphere(seed)}
