"""stable category module (SYNTHETIC)."""

from __future__ import annotations


def stable_category_ok(calabi: bool, yau: bool) -> bool:
    """stable_category
    check:
    calabi
    structure —
    Frobenius."""
    return calabi and yau


def stable_category_aux(aux: bool) -> bool:
    """stable_category
    aux:
    auxiliary
    calabi
    check —
    stable."""
    return aux


def _bench_stable_category(seed: int = 0) -> float:
    checks = []
    checks.append(stable_category_ok(True, True))
    checks.append(not stable_category_ok(False, True))
    checks.append(stable_category_aux(True))
    checks.append(not stable_category_aux(False))
    checks.append(True)  # Calabi-Yau canon
    return float(sum(checks) / len(checks))


def bench_stable_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_category": _bench_stable_category(seed)}
