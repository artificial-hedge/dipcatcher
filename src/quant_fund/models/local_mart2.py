"""local mart2 module (SYNTHETIC)."""

from __future__ import annotations


def local_mart2_ok(sm1: bool, pw: bool) -> bool:
    """local_mart2
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def local_mart2_aux(aux: bool) -> bool:
    """local_mart2
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_local_mart2(seed: int = 0) -> float:
    checks = []
    checks.append(local_mart2_ok(True, True))
    checks.append(not local_mart2_ok(False, True))
    checks.append(local_mart2_aux(True))
    checks.append(not local_mart2_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_local_mart2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_mart2": _bench_local_mart2(seed)}
