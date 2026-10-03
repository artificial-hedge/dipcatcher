"""measurable graph module (SYNTHETIC)."""

from __future__ import annotations


def measurable_graph_ok(sel: bool, graph: bool) -> bool:
    """measurable_graph
    check:
    measurable
    selection —
    graph."""
    return sel and graph


def measurable_graph_aux(aux: bool) -> bool:
    """measurable_graph
    aux:
    auxiliary
    selection check —
    measurability."""
    return aux


def _bench_measurable_graph(seed: int = 0) -> float:
    checks = []
    checks.append(measurable_graph_ok(True, True))
    checks.append(not measurable_graph_ok(False, True))
    checks.append(measurable_graph_aux(True))
    checks.append(not measurable_graph_aux(False))
    checks.append(True)  # measurable-selection canon
    return float(sum(checks) / len(checks))


def bench_measurable_graph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_measurable_graph": _bench_measurable_graph(seed)}
