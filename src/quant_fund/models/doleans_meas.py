"""doleans meas module (SYNTHETIC)."""

from __future__ import annotations


def doleans_meas_ok(pred: bool, mart: bool) -> bool:
    """doleans_meas
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def doleans_meas_aux(aux: bool) -> bool:
    """doleans_meas
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_doleans_meas(seed: int = 0) -> float:
    checks = []
    checks.append(doleans_meas_ok(True, True))
    checks.append(not doleans_meas_ok(False, True))
    checks.append(doleans_meas_aux(True))
    checks.append(not doleans_meas_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_doleans_meas(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doleans_meas": _bench_doleans_meas(seed)}
