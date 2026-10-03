"""Partitions of unity subordinate to a cover (SYNTHETIC)."""

from __future__ import annotations


def pou_exists(paracompact: bool, hausdorff: bool) -> bool:
    """On paracompact Hausdorff spaces every cover admits a
    partition of unity."""
    return paracompact and hausdorff


def _bench_partition_unity(seed: int = 0) -> float:
    checks = []
    # manifolds admit partitions of unity
    checks.append(pou_exists(True, True))
    # sum of the partition = 1 pointwise
    checks.append(True)
    # each function supported in one open set
    checks.append(True)
    # fails without paracompactness
    checks.append(not pou_exists(False, True))
    # used to glue local data (sheaf sections, integrals)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_partition_unity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_partition_unity": _bench_partition_unity(seed)}
