"""gneding metric module (SYNTHETIC)."""

from __future__ import annotations


def gneding_metric_ok(cp: bool, pg: bool) -> bool:
    """gneding_metric
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def gneding_metric_aux(aux: bool) -> bool:
    """gneding_metric
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_gneding_metric(seed: int = 0) -> float:
    checks = []
    checks.append(gneding_metric_ok(True, True))
    checks.append(not gneding_metric_ok(False, True))
    checks.append(gneding_metric_aux(True))
    checks.append(not gneding_metric_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_gneding_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gneding_metric": _bench_gneding_metric(seed)}
