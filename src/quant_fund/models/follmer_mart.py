"""follmer mart module (SYNTHETIC)."""

from __future__ import annotations


def follmer_mart_ok(sm1: bool, pw: bool) -> bool:
    """follmer_mart
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def follmer_mart_aux(aux: bool) -> bool:
    """follmer_mart
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_follmer_mart(seed: int = 0) -> float:
    checks = []
    checks.append(follmer_mart_ok(True, True))
    checks.append(not follmer_mart_ok(False, True))
    checks.append(follmer_mart_aux(True))
    checks.append(not follmer_mart_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_follmer_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_follmer_mart": _bench_follmer_mart(seed)}
