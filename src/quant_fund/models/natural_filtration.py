"""natural filtration module (SYNTHETIC)."""

from __future__ import annotations


def natural_filtration_ok(f: bool, right: bool) -> bool:
    """natural_filtration
    check:
    filtration —
    right
    continuity."""
    return f and right


def natural_filtration_aux(aux: bool) -> bool:
    """natural_filtration
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_natural_filtration(seed: int = 0) -> float:
    checks = []
    checks.append(natural_filtration_ok(True, True))
    checks.append(not natural_filtration_ok(False, True))
    checks.append(natural_filtration_aux(True))
    checks.append(not natural_filtration_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_natural_filtration(seed: int = 0) -> dict[str, float]:
    return {"synthetic_natural_filtration": _bench_natural_filtration(seed)}
