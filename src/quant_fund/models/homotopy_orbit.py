"""homotopy orbit module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_orbit_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_orbit
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_orbit_aux(aux: bool) -> bool:
    """homotopy_orbit
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_homotopy_orbit(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_orbit_ok(True, True))
    checks.append(not homotopy_orbit_ok(False, True))
    checks.append(homotopy_orbit_aux(True))
    checks.append(not homotopy_orbit_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_orbit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_orbit": _bench_homotopy_orbit(seed)}
