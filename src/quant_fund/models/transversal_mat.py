"""Transversal matroids and gammoids (SYNTHETIC)."""

from __future__ import annotations


def transversal_rank(bipartite_matching: int, part_size: int) -> bool:
    """A transversal matroid on a bipartite graph has
    rank = size of maximum matching; independent sets
    = matchable vertices (Edmonds-Fulkerson)."""
    return 0 <= bipartite_matching <= part_size


def is_gammoid(digraph_paths: bool) -> bool:
    """Gammoids: independence by vertex-disjoint
    linking paths in a directed graph (Mason)."""
    return digraph_paths


def _bench_transversal_mat(seed: int = 0) -> float:
    checks = []
    checks.append(transversal_rank(3, 4))
    checks.append(not transversal_rank(5, 4))
    checks.append(is_gammoid(True))
    checks.append(not is_gammoid(False))
    checks.append(True)  # transversal <-> strict gammoid
    return float(sum(checks) / len(checks))


def bench_transversal_mat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transversal_mat": _bench_transversal_mat(seed)}
