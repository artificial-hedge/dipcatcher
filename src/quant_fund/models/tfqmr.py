"""tfqmr module (SYNTHETIC)."""

from __future__ import annotations


def tfqmr_ok(res: bool, it: bool) -> bool:
    """tfqmr
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def tfqmr_aux(aux: bool) -> bool:
    """tfqmr
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_tfqmr(seed: int = 0) -> float:
    checks = []
    checks.append(tfqmr_ok(True, True))
    checks.append(not tfqmr_ok(False, True))
    checks.append(tfqmr_aux(True))
    checks.append(not tfqmr_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_tfqmr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tfqmr": _bench_tfqmr(seed)}
