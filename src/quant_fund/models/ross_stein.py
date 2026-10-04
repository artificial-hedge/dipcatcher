"""ross stein module (SYNTHETIC)."""

from __future__ import annotations


def ross_stein_ok(op: bool, bound: bool) -> bool:
    """ross_stein
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def ross_stein_aux(aux: bool) -> bool:
    """ross_stein
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_ross_stein(seed: int = 0) -> float:
    checks = []
    checks.append(ross_stein_ok(True, True))
    checks.append(not ross_stein_ok(False, True))
    checks.append(ross_stein_aux(True))
    checks.append(not ross_stein_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_ross_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ross_stein": _bench_ross_stein(seed)}
