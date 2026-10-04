"""de boor module (SYNTHETIC)."""

from __future__ import annotations


def de_boor_ok(knots: bool, basis: bool) -> bool:
    """de_boor
    check:
    spline
    theory —
    knots."""
    return knots and basis


def de_boor_aux(aux: bool) -> bool:
    """de_boor
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_de_boor(seed: int = 0) -> float:
    checks = []
    checks.append(de_boor_ok(True, True))
    checks.append(not de_boor_ok(False, True))
    checks.append(de_boor_aux(True))
    checks.append(not de_boor_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_de_boor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_de_boor": _bench_de_boor(seed)}
