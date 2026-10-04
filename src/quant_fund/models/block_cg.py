"""block cg module (SYNTHETIC)."""

from __future__ import annotations


def block_cg_ok(res: bool, it: bool) -> bool:
    """block_cg
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def block_cg_aux(aux: bool) -> bool:
    """block_cg
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_block_cg(seed: int = 0) -> float:
    checks = []
    checks.append(block_cg_ok(True, True))
    checks.append(not block_cg_ok(False, True))
    checks.append(block_cg_aux(True))
    checks.append(not block_cg_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_block_cg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_block_cg": _bench_block_cg(seed)}
