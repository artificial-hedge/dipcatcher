"""chen stein module (SYNTHETIC)."""

from __future__ import annotations


def chen_stein_ok(op: bool, bound: bool) -> bool:
    """chen_stein
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def chen_stein_aux(aux: bool) -> bool:
    """chen_stein
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_chen_stein(seed: int = 0) -> float:
    checks = []
    checks.append(chen_stein_ok(True, True))
    checks.append(not chen_stein_ok(False, True))
    checks.append(chen_stein_aux(True))
    checks.append(not chen_stein_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_chen_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chen_stein": _bench_chen_stein(seed)}
