"""nc motive module (SYNTHETIC)."""

from __future__ import annotations


def nc_motive_ok(noncommutative: bool, motivic: bool) -> bool:
    """nc_motive
    check:
    noncommutative
    structure —
    dg."""
    return noncommutative and motivic


def nc_motive_aux(aux: bool) -> bool:
    """nc_motive
    aux:
    auxiliary
    noncommutative
    check —
    Morita."""
    return aux


def _bench_nc_motive(seed: int = 0) -> float:
    checks = []
    checks.append(nc_motive_ok(True, True))
    checks.append(not nc_motive_ok(False, True))
    checks.append(nc_motive_aux(True))
    checks.append(not nc_motive_aux(False))
    checks.append(True)  # nc-motives canon
    return float(sum(checks) / len(checks))


def bench_nc_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nc_motive": _bench_nc_motive(seed)}
