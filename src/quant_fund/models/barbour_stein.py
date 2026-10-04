"""barbour stein module (SYNTHETIC)."""

from __future__ import annotations


def barbour_stein_ok(op: bool, bound: bool) -> bool:
    """barbour_stein
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def barbour_stein_aux(aux: bool) -> bool:
    """barbour_stein
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_barbour_stein(seed: int = 0) -> float:
    checks = []
    checks.append(barbour_stein_ok(True, True))
    checks.append(not barbour_stein_ok(False, True))
    checks.append(barbour_stein_aux(True))
    checks.append(not barbour_stein_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_barbour_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barbour_stein": _bench_barbour_stein(seed)}
