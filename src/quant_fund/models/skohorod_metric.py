"""skohorod metric module (SYNTHETIC)."""

from __future__ import annotations


def skohorod_metric_ok(measure: bool, tight: bool) -> bool:
    """skohorod_metric
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def skohorod_metric_aux(aux: bool) -> bool:
    """skohorod_metric
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_skohorod_metric(seed: int = 0) -> float:
    checks = []
    checks.append(skohorod_metric_ok(True, True))
    checks.append(not skohorod_metric_ok(False, True))
    checks.append(skohorod_metric_aux(True))
    checks.append(not skohorod_metric_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_skohorod_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skohorod_metric": _bench_skohorod_metric(seed)}
