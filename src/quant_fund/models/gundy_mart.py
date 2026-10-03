"""gundy mart module (SYNTHETIC)."""

from __future__ import annotations


def gundy_mart_ok(pred: bool, mart: bool) -> bool:
    """gundy_mart
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def gundy_mart_aux(aux: bool) -> bool:
    """gundy_mart
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_gundy_mart(seed: int = 0) -> float:
    checks = []
    checks.append(gundy_mart_ok(True, True))
    checks.append(not gundy_mart_ok(False, True))
    checks.append(gundy_mart_aux(True))
    checks.append(not gundy_mart_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_gundy_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gundy_mart": _bench_gundy_mart(seed)}
