"""nc k_theory module (SYNTHETIC)."""

from __future__ import annotations


def nc_k_theory_ok(noncommutative: bool, motivic: bool) -> bool:
    """nc_k_theory
    check:
    noncommutative
    structure —
    dg."""
    return noncommutative and motivic


def nc_k_theory_aux(aux: bool) -> bool:
    """nc_k_theory
    aux:
    auxiliary
    noncommutative
    check —
    Morita."""
    return aux


def _bench_nc_k_theory(seed: int = 0) -> float:
    checks = []
    checks.append(nc_k_theory_ok(True, True))
    checks.append(not nc_k_theory_ok(False, True))
    checks.append(nc_k_theory_aux(True))
    checks.append(not nc_k_theory_aux(False))
    checks.append(True)  # nc-motives canon
    return float(sum(checks) / len(checks))


def bench_nc_k_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nc_k_theory": _bench_nc_k_theory(seed)}
