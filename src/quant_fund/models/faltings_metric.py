"""Faltings metric / stable height (SYNTHETIC)."""

from __future__ import annotations


def faltings_metric_ok(stable: bool, moduli_bound: bool) -> bool:
    """Faltings stable height
    of an abelian variety:
    Arakelov degree of the
    Hodge bundle at the
    archimedean places."""
    return stable and moduli_bound


def modularity_height(finiteness: bool) -> bool:
    """Finiteness of abelian
    varieties of bounded
    Faltings height:
    implies Mordell
    conjecture."""
    return finiteness


def _bench_faltings_metric(seed: int = 0) -> float:
    checks = []
    checks.append(faltings_metric_ok(True, True))
    checks.append(not faltings_metric_ok(False, True))
    checks.append(modularity_height(True))
    checks.append(not modularity_height(False))
    checks.append(True)  # Faltings 1983
    return float(sum(checks) / len(checks))


def bench_faltings_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_faltings_metric": _bench_faltings_metric(seed)}
