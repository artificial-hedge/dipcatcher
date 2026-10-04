"""surfaces operad module (SYNTHETIC)."""

from __future__ import annotations


def surfaces_operad_ok(higher: bool, algebra: bool) -> bool:
    """surfaces_operad
    check:
    higher
    algebra —
    operadic."""
    return higher and algebra


def surfaces_operad_aux(aux: bool) -> bool:
    """surfaces_operad
    aux:
    auxiliary
    higher
    check —
    factorization."""
    return aux


def _bench_surfaces_operad(seed: int = 0) -> float:
    checks = []
    checks.append(surfaces_operad_ok(True, True))
    checks.append(not surfaces_operad_ok(False, True))
    checks.append(surfaces_operad_aux(True))
    checks.append(not surfaces_operad_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_surfaces_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_surfaces_operad": _bench_surfaces_operad(seed)}
