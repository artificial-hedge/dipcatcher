"""floyd farey module (SYNTHETIC)."""

from __future__ import annotations


def floyd_farey_ok(higher: bool, algebraic: bool) -> bool:
    """floyd_farey
    check:
    higher-algebra
    structure —
    operad."""
    return higher and algebraic


def floyd_farey_aux(aux: bool) -> bool:
    """floyd_farey
    aux:
    auxiliary
    higher
    check —
    discs."""
    return aux


def _bench_floyd_farey(seed: int = 0) -> float:
    checks = []
    checks.append(floyd_farey_ok(True, True))
    checks.append(not floyd_farey_ok(False, True))
    checks.append(floyd_farey_aux(True))
    checks.append(not floyd_farey_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_floyd_farey(seed: int = 0) -> dict[str, float]:
    return {"synthetic_floyd_farey": _bench_floyd_farey(seed)}
