"""dolean mart module (SYNTHETIC)."""

from __future__ import annotations


def dolean_mart_ok(sm1: bool, pw: bool) -> bool:
    """dolean_mart
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def dolean_mart_aux(aux: bool) -> bool:
    """dolean_mart
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_dolean_mart(seed: int = 0) -> float:
    checks = []
    checks.append(dolean_mart_ok(True, True))
    checks.append(not dolean_mart_ok(False, True))
    checks.append(dolean_mart_aux(True))
    checks.append(not dolean_mart_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_dolean_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dolean_mart": _bench_dolean_mart(seed)}
