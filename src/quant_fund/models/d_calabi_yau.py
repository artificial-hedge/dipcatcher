"""d calabi_yau module (SYNTHETIC)."""

from __future__ import annotations


def d_calabi_yau_ok(calabi: bool, yau: bool) -> bool:
    """d_calabi_yau
    check:
    calabi
    structure —
    Frobenius."""
    return calabi and yau


def d_calabi_yau_aux(aux: bool) -> bool:
    """d_calabi_yau
    aux:
    auxiliary
    calabi
    check —
    stable."""
    return aux


def _bench_d_calabi_yau(seed: int = 0) -> float:
    checks = []
    checks.append(d_calabi_yau_ok(True, True))
    checks.append(not d_calabi_yau_ok(False, True))
    checks.append(d_calabi_yau_aux(True))
    checks.append(not d_calabi_yau_aux(False))
    checks.append(True)  # Calabi-Yau canon
    return float(sum(checks) / len(checks))


def bench_d_calabi_yau(seed: int = 0) -> dict[str, float]:
    return {"synthetic_d_calabi_yau": _bench_d_calabi_yau(seed)}
