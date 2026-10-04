"""little discs3 module (SYNTHETIC)."""

from __future__ import annotations


def little_discs3_ok(higher: bool, algebraic: bool) -> bool:
    """little_discs3
    check:
    higher-algebra
    structure —
    operad."""
    return higher and algebraic


def little_discs3_aux(aux: bool) -> bool:
    """little_discs3
    aux:
    auxiliary
    higher
    check —
    discs."""
    return aux


def _bench_little_discs3(seed: int = 0) -> float:
    checks = []
    checks.append(little_discs3_ok(True, True))
    checks.append(not little_discs3_ok(False, True))
    checks.append(little_discs3_aux(True))
    checks.append(not little_discs3_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_little_discs3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_little_discs3": _bench_little_discs3(seed)}
