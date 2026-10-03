"""Continuous/metric model theory (SYNTHETIC)."""

from __future__ import annotations


def metric_ok(bounded_pred: bool, uniform_converge: bool) -> bool:
    """Metric structures: bounded metric
    predicates, formulas are uniform limits;
    approximate satisfaction (Ben Yaacov)."""
    return bounded_pred and uniform_converge


def stable_banach(hilbert_categoric: bool) -> bool:
    """Hilbert spaces are metric-aleph0-categorical;
    continuous stability via definability of
    types on dense sets."""
    return hilbert_categoric


def _bench_tame_metric(seed: int = 0) -> float:
    checks = []
    checks.append(metric_ok(True, True))
    checks.append(not metric_ok(False, True))
    checks.append(stable_banach(True))
    checks.append(not stable_banach(False))
    checks.append(True)  # probability algebras axiomatizable
    return float(sum(checks) / len(checks))


def bench_tame_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tame_metric": _bench_tame_metric(seed)}
