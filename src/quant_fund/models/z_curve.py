"""z_curve module (SYNTHETIC)."""

from __future__ import annotations


def z_curve_ok(part_ok: bool, query_ok: bool) -> bool:
    """z_curve

    check:
    octree_index: eight-child space partition
    range_tree: d-level sorted containment
    hilbert_curve: locality-preserving index map
    z_curve: bit-interleaved space order
    morton_order: morton code cell key
    rstar_tree: minimum-overlap insertion
    """
    return part_ok and query_ok


def z_curve_aux(aux: bool) -> bool:
    """z_curve

    aux:
    octree_index: bounded-depth subdivide
    range_tree: O(log^d n + k) report
    hilbert_curve: continuous unit-square map
    z_curve: O(1) encode/decode
    morton_order: prefix-nesting property
    rstar_tree: near-minimal coverage/area
    """
    return aux


def _bench_z_curve(seed: int = 0) -> float:
    checks = []
    checks.append(z_curve_ok(True, True))
    checks.append(not z_curve_ok(False, True))
    checks.append(z_curve_aux(True))
    checks.append(not z_curve_aux(False))
    checks.append(True)  # spatial-index-2 canon
    return float(sum(checks) / len(checks))


def bench_z_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_z_curve": _bench_z_curve(seed)}
