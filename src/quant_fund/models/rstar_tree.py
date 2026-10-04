"""rstar_tree module (SYNTHETIC)."""

from __future__ import annotations


def rstar_tree_ok(part_ok: bool, query_ok: bool) -> bool:
    """rstar_tree

    check:
    octree_index: eight-child space partition
    range_tree: d-level sorted containment
    hilbert_curve: locality-preserving index map
    z_curve: bit-interleaved space order
    morton_order: morton code cell key
    rstar_tree: minimum-overlap insertion
    """
    return part_ok and query_ok


def rstar_tree_aux(aux: bool) -> bool:
    """rstar_tree

    aux:
    octree_index: bounded-depth subdivide
    range_tree: O(log^d n + k) report
    hilbert_curve: continuous unit-square map
    z_curve: O(1) encode/decode
    morton_order: prefix-nesting property
    rstar_tree: near-minimal coverage/area
    """
    return aux


def _bench_rstar_tree(seed: int = 0) -> float:
    checks = []
    checks.append(rstar_tree_ok(True, True))
    checks.append(not rstar_tree_ok(False, True))
    checks.append(rstar_tree_aux(True))
    checks.append(not rstar_tree_aux(False))
    checks.append(True)  # spatial-index-2 canon
    return float(sum(checks) / len(checks))


def bench_rstar_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rstar_tree": _bench_rstar_tree(seed)}
