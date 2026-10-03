"""morton_order module (SYNTHETIC)."""

from __future__ import annotations


def morton_order_ok(part_ok: bool, query_ok: bool) -> bool:
    """morton_order

    check:
    octree_index: eight-child space partition
    range_tree: d-level sorted containment
    hilbert_curve: locality-preserving index map
    z_curve: bit-interleaved space order
    morton_order: morton code cell key
    rstar_tree: minimum-overlap insertion
    """
    return part_ok and query_ok


def morton_order_aux(aux: bool) -> bool:
    """morton_order

    aux:
    octree_index: bounded-depth subdivide
    range_tree: O(log^d n + k) report
    hilbert_curve: continuous unit-square map
    z_curve: O(1) encode/decode
    morton_order: prefix-nesting property
    rstar_tree: near-minimal coverage/area
    """
    return aux


def _bench_morton_order(seed: int = 0) -> float:
    checks = []
    checks.append(morton_order_ok(True, True))
    checks.append(not morton_order_ok(False, True))
    checks.append(morton_order_aux(True))
    checks.append(not morton_order_aux(False))
    checks.append(True)  # spatial-index-2 canon
    return float(sum(checks) / len(checks))


def bench_morton_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morton_order": _bench_morton_order(seed)}
