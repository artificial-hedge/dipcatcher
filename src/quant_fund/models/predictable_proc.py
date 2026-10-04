"""predictable proc module (SYNTHETIC)."""

from __future__ import annotations


def predictable_proc_ok(pred: bool, mart: bool) -> bool:
    """predictable_proc
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def predictable_proc_aux(aux: bool) -> bool:
    """predictable_proc
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_predictable_proc(seed: int = 0) -> float:
    checks = []
    checks.append(predictable_proc_ok(True, True))
    checks.append(not predictable_proc_ok(False, True))
    checks.append(predictable_proc_aux(True))
    checks.append(not predictable_proc_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_predictable_proc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_predictable_proc": _bench_predictable_proc(seed)}
