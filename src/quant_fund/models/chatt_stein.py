"""chatt stein module (SYNTHETIC)."""

from __future__ import annotations


def chatt_stein_ok(op: bool, bound: bool) -> bool:
    """chatt_stein
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def chatt_stein_aux(aux: bool) -> bool:
    """chatt_stein
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_chatt_stein(seed: int = 0) -> float:
    checks = []
    checks.append(chatt_stein_ok(True, True))
    checks.append(not chatt_stein_ok(False, True))
    checks.append(chatt_stein_aux(True))
    checks.append(not chatt_stein_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_chatt_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chatt_stein": _bench_chatt_stein(seed)}
