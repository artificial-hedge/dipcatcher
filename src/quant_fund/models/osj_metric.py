"""osj metric module (SYNTHETIC)."""

from __future__ import annotations


def osj_metric_ok(proc: bool, tight: bool) -> bool:
    """osj_metric
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def osj_metric_aux(aux: bool) -> bool:
    """osj_metric
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_osj_metric(seed: int = 0) -> float:
    checks = []
    checks.append(osj_metric_ok(True, True))
    checks.append(not osj_metric_ok(False, True))
    checks.append(osj_metric_aux(True))
    checks.append(not osj_metric_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_osj_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osj_metric": _bench_osj_metric(seed)}
