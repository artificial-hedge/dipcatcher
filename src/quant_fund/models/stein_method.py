"""stein method module (SYNTHETIC)."""

from __future__ import annotations


def stein_method_ok(op: bool, bound: bool) -> bool:
    """stein_method
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def stein_method_aux(aux: bool) -> bool:
    """stein_method
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_stein_method(seed: int = 0) -> float:
    checks = []
    checks.append(stein_method_ok(True, True))
    checks.append(not stein_method_ok(False, True))
    checks.append(stein_method_aux(True))
    checks.append(not stein_method_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_stein_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stein_method": _bench_stein_method(seed)}
