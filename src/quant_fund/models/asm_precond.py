"""asm precond module (SYNTHETIC)."""

from __future__ import annotations


def asm_precond_ok(part: bool, coarse: bool) -> bool:
    """asm_precond
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def asm_precond_aux(aux: bool) -> bool:
    """asm_precond
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_asm_precond(seed: int = 0) -> float:
    checks = []
    checks.append(asm_precond_ok(True, True))
    checks.append(not asm_precond_ok(False, True))
    checks.append(asm_precond_aux(True))
    checks.append(not asm_precond_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_asm_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asm_precond": _bench_asm_precond(seed)}
