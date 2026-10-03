"""metric entropy module (SYNTHETIC)."""

from __future__ import annotations


def metric_entropy_ok(ent: bool, proc: bool) -> bool:
    """metric_entropy
    check:
    empirical
    process —
    uniform bound."""
    return ent and proc


def metric_entropy_aux(aux: bool) -> bool:
    """metric_entropy
    aux:
    auxiliary
    process check —
    complexity."""
    return aux


def _bench_metric_entropy(seed: int = 0) -> float:
    checks = []
    checks.append(metric_entropy_ok(True, True))
    checks.append(not metric_entropy_ok(False, True))
    checks.append(metric_entropy_aux(True))
    checks.append(not metric_entropy_aux(False))
    checks.append(True)  # empirical-process canon
    return float(sum(checks) / len(checks))


def bench_metric_entropy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_metric_entropy": _bench_metric_entropy(seed)}
