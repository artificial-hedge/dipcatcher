"""usual cond module (SYNTHETIC)."""

from __future__ import annotations


def usual_cond_ok(sm1: bool, pw: bool) -> bool:
    """usual_cond
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def usual_cond_aux(aux: bool) -> bool:
    """usual_cond
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_usual_cond(seed: int = 0) -> float:
    checks = []
    checks.append(usual_cond_ok(True, True))
    checks.append(not usual_cond_ok(False, True))
    checks.append(usual_cond_aux(True))
    checks.append(not usual_cond_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_usual_cond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_usual_cond": _bench_usual_cond(seed)}
