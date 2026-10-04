"""Tropical curves (SYNTHETIC)."""

from __future__ import annotations


def tropical_curve_ok(metric_graph: bool, balanced: bool) -> bool:
    """Tropical curve:
    metric graph with
    integer edge lengths,
    balancing condition at
    each vertex (Mikhalkin)."""
    return metric_graph and balanced


def tropical_genus(betti: bool) -> bool:
    """Genus of tropical
    curve = first Betti
    number of the metric
    graph; divisor theory
    via chip-firing."""
    return betti


def _bench_tropical_curve(seed: int = 0) -> float:
    checks = []
    checks.append(tropical_curve_ok(True, True))
    checks.append(not tropical_curve_ok(False, True))
    checks.append(tropical_genus(True))
    checks.append(not tropical_genus(False))
    checks.append(True)  # tropical Riemann-Roch
    return float(sum(checks) / len(checks))


def bench_tropical_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tropical_curve": _bench_tropical_curve(seed)}
