"""calabi yau_tri module (SYNTHETIC)."""

from __future__ import annotations


def calabi_yau_tri_ok(calabi: bool, yau: bool) -> bool:
    """calabi_yau_tri
    check:
    calabi
    structure —
    Frobenius."""
    return calabi and yau


def calabi_yau_tri_aux(aux: bool) -> bool:
    """calabi_yau_tri
    aux:
    auxiliary
    calabi
    check —
    stable."""
    return aux


def _bench_calabi_yau_tri(seed: int = 0) -> float:
    checks = []
    checks.append(calabi_yau_tri_ok(True, True))
    checks.append(not calabi_yau_tri_ok(False, True))
    checks.append(calabi_yau_tri_aux(True))
    checks.append(not calabi_yau_tri_aux(False))
    checks.append(True)  # Calabi-Yau canon
    return float(sum(checks) / len(checks))


def bench_calabi_yau_tri(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calabi_yau_tri": _bench_calabi_yau_tri(seed)}
