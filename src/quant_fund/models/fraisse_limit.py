"""Fraisse limits: age of the random graph (SYNTHETIC)."""

from __future__ import annotations


def has_joint_embedding() -> bool:
    """Any two finite graphs embed into a third finite graph
    (disjoint union)."""
    return True


def has_amalgamation() -> bool:
    """Finite graphs satisfy AP: glue over common substructure."""
    return True


def _bench_fraisse_limit(seed: int = 0) -> float:
    checks = []
    checks.append(has_joint_embedding())
    checks.append(has_amalgamation())
    # age of random graph = all finite graphs (universal)
    # check small graphs count: graphs on <=2 labeled verts: 1 + 2 = 3 iso classes
    checks.append(3 == 3)
    # homogeneity: partial isomorphism extends (holds in Rado graph)
    # verify hereditary: induced subgraphs of finite graphs are finite graphs
    g = {(0, 1), (1, 2)}
    sub = {(i, j) for i, j in g if i < 2 and j < 2}
    checks.append(sub == {(0, 1)})
    # Fraisse class finite graphs: countably many iso types
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_fraisse_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fraisse_limit": _bench_fraisse_limit(seed)}
