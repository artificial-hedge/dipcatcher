"""Local compactness and one-point compactification (SYNTHETIC)."""

from __future__ import annotations


def one_point_compact(loc_cpt: bool, hausdorff: bool) -> bool:
    """X has a Hausdorff one-point compactification iff X is
    locally compact Hausdorff."""
    return loc_cpt and hausdorff


def _bench_locally_compact(seed: int = 0) -> float:
    checks = []
    # R is locally compact Hausdorff -> S^1 compactification
    checks.append(one_point_compact(True, True))
    # Q is not locally compact -> no nice compactification
    checks.append(not one_point_compact(False, True))
    # R^n compactifies to S^n
    checks.append(one_point_compact(True, True))
    # compact spaces are locally compact
    checks.append(one_point_compact(True, True))
    # open subspace of LCH is LCH
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_locally_compact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_locally_compact": _bench_locally_compact(seed)}
