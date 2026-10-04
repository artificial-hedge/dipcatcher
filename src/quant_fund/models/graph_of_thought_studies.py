"""graph_of_thought_studies module (SYNTHETIC)."""

from __future__ import annotations


def graph_of_thought_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """graph_of_thought_studies

    check:
    graph_of_thought_studies: thought nodes and merge operations/states and edges
    """
    return fit_ok and sample_ok


def graph_of_thought_studies_aux(aux: bool) -> bool:
    """graph_of_thought_studies

    aux:
    graph_of_thought_studies: aggregation over reasoning subgraphs/paths and votes
    """
    return aux


def _bench_graph_of_thought_studies(seed: int = 0) -> float:
    checks = []
    checks.append(graph_of_thought_studies_ok(True, True))
    checks.append(not graph_of_thought_studies_ok(False, True))
    checks.append(graph_of_thought_studies_aux(True))
    checks.append(not graph_of_thought_studies_aux(False))
    checks.append(True)  # reasoning-prompt canon
    return float(sum(checks) / len(checks))


def bench_graph_of_thought_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_graph_of_thought_studies": _bench_graph_of_thought_studies(seed)}
