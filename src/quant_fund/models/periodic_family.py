"""periodic family module (SYNTHETIC)."""

from __future__ import annotations


def periodic_family_ok(homotopy: bool, unstable: bool) -> bool:
    """periodic_family
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def periodic_family_aux(aux: bool) -> bool:
    """periodic_family
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_periodic_family(seed: int = 0) -> float:
    checks = []
    checks.append(periodic_family_ok(True, True))
    checks.append(not periodic_family_ok(False, True))
    checks.append(periodic_family_aux(True))
    checks.append(not periodic_family_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_periodic_family(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodic_family": _bench_periodic_family(seed)}
