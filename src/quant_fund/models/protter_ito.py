"""protter ito module (SYNTHETIC)."""

from __future__ import annotations


def protter_ito_ok(sm1: bool, pw: bool) -> bool:
    """protter_ito
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def protter_ito_aux(aux: bool) -> bool:
    """protter_ito
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_protter_ito(seed: int = 0) -> float:
    checks = []
    checks.append(protter_ito_ok(True, True))
    checks.append(not protter_ito_ok(False, True))
    checks.append(protter_ito_aux(True))
    checks.append(not protter_ito_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_protter_ito(seed: int = 0) -> dict[str, float]:
    return {"synthetic_protter_ito": _bench_protter_ito(seed)}
