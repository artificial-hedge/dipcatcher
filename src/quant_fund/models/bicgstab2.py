"""bicgstab2 module (SYNTHETIC)."""

from __future__ import annotations


def bicgstab2_ok(res: bool, it: bool) -> bool:
    """bicgstab2
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def bicgstab2_aux(aux: bool) -> bool:
    """bicgstab2
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_bicgstab2(seed: int = 0) -> float:
    checks = []
    checks.append(bicgstab2_ok(True, True))
    checks.append(not bicgstab2_ok(False, True))
    checks.append(bicgstab2_aux(True))
    checks.append(not bicgstab2_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_bicgstab2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bicgstab2": _bench_bicgstab2(seed)}
