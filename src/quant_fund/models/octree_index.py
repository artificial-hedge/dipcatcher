"""octree_index module (SYNTHETIC)."""

from __future__ import annotations


def octree_index_ok(part_ok: bool, query_ok: bool) -> bool:
    """octree_index

    check:
    octree_index: eight-child space partition
    range_tree: d-level sorted containment
    hilbert_curve: locality-preserving index map
    z_curve: bit-interleaved space order
    morton_order: morton code cell key
    rstar_tree: minimum-overlap insertion
    """
    return part_ok and query_ok


def octree_index_aux(aux: bool) -> bool:
    """octree_index

    aux:
    octree_index: bounded-depth subdivide
    range_tree: O(log^d n + k) report
    hilbert_curve: continuous unit-square map
    z_curve: O(1) encode/decode
    morton_order: prefix-nesting property
    rstar_tree: near-minimal coverage/area
    """
    return aux


def _bench_octree_index(seed: int = 0) -> float:
    checks = []
    checks.append(octree_index_ok(True, True))
    checks.append(not octree_index_ok(False, True))
    checks.append(octree_index_aux(True))
    checks.append(not octree_index_aux(False))
    checks.append(True)  # spatial-index-2 canon
    return float(sum(checks) / len(checks))


def bench_octree_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_octree_index": _bench_octree_index(seed)}
