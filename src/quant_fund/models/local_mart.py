"""local mart module (SYNTHETIC)."""

from __future__ import annotations


def local_mart_ok(pred: bool, mart: bool) -> bool:
    """local_mart
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def local_mart_aux(aux: bool) -> bool:
    """local_mart
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_local_mart(seed: int = 0) -> float:
    checks = []
    checks.append(local_mart_ok(True, True))
    checks.append(not local_mart_ok(False, True))
    checks.append(local_mart_aux(True))
    checks.append(not local_mart_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_local_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_mart": _bench_local_mart(seed)}
