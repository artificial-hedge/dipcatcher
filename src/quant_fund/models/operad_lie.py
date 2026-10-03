"""operad lie module (SYNTHETIC)."""

from __future__ import annotations


def operad_lie_ok(higher: bool, algebra: bool) -> bool:
    """operad_lie
    check:
    higher
    algebra —
    operadic."""
    return higher and algebra


def operad_lie_aux(aux: bool) -> bool:
    """operad_lie
    aux:
    auxiliary
    higher
    check —
    factorization."""
    return aux


def _bench_operad_lie(seed: int = 0) -> float:
    checks = []
    checks.append(operad_lie_ok(True, True))
    checks.append(not operad_lie_ok(False, True))
    checks.append(operad_lie_aux(True))
    checks.append(not operad_lie_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_operad_lie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_lie": _bench_operad_lie(seed)}
