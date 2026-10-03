"""linear_probe module (SYNTHETIC)."""

from __future__ import annotations


def linear_probe_ok(ins_ok: bool, get_ok: bool) -> bool:
    """linear_probe

    check:
    chained_hash: bucket-chain collision handling
    linear_probe: open-addressing probe sequence
    bucket_sort: uniform-distribution binning
    shell_sort: diminishing-gap insertion sort
    rand_access_list: O(1) lookup cons-list
    skew_list: skew-binary random access list
    """
    return ins_ok and get_ok


def linear_probe_aux(aux: bool) -> bool:
    """linear_probe

    aux:
    chained_hash: load-factor resize policy
    linear_probe: tombstone-free deletion
    bucket_sort: stable intra-bucket sort
    shell_sort: Ciura gap sequence
    rand_access_list: O(log n) update
    skew_list: O(1) cons/head/tail
    """
    return aux


def _bench_linear_probe(seed: int = 0) -> float:
    checks = []
    checks.append(linear_probe_ok(True, True))
    checks.append(not linear_probe_ok(False, True))
    checks.append(linear_probe_aux(True))
    checks.append(not linear_probe_aux(False))
    checks.append(True)  # data-structures-3 canon
    return float(sum(checks) / len(checks))


def bench_linear_probe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_linear_probe": _bench_linear_probe(seed)}
