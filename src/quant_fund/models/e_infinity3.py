"""E-infinity algebras (SYNTHETIC)."""

from __future__ import annotations


def ei3_ok(e_infty: bool, algebra: bool) -> bool:
    """E-
    infinity:
    E-
    infinity
    algebra —
    May
    E-
    infinity."""
    return e_infty and algebra


def e_infty_maps(em: bool) -> bool:
    """E-
    infinity
    maps:
    E-
    infinity
    maps —
    E-
    infinity
    operad."""
    return em


def _bench_e_infinity3(seed: int = 0) -> float:
    checks = []
    checks.append(ei3_ok(True, True))
    checks.append(not ei3_ok(False, True))
    checks.append(e_infty_maps(True))
    checks.append(not e_infty_maps(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_e_infinity3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_e_infinity3": _bench_e_infinity3(seed)}
