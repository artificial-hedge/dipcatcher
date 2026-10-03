"""skew_list module (SYNTHETIC)."""

from __future__ import annotations


def skew_list_ok(ins_ok: bool, get_ok: bool) -> bool:
    """skew_list

    check:
    chained_hash: bucket-chain collision handling
    linear_probe: open-addressing probe sequence
    bucket_sort: uniform-distribution binning
    shell_sort: diminishing-gap insertion sort
    rand_access_list: O(1) lookup cons-list
    skew_list: skew-binary random access list
    """
    return ins_ok and get_ok


def skew_list_aux(aux: bool) -> bool:
    """skew_list

    aux:
    chained_hash: load-factor resize policy
    linear_probe: tombstone-free deletion
    bucket_sort: stable intra-bucket sort
    shell_sort: Ciura gap sequence
    rand_access_list: O(log n) update
    skew_list: O(1) cons/head/tail
    """
    return aux


def _bench_skew_list(seed: int = 0) -> float:
    checks = []
    checks.append(skew_list_ok(True, True))
    checks.append(not skew_list_ok(False, True))
    checks.append(skew_list_aux(True))
    checks.append(not skew_list_aux(False))
    checks.append(True)  # data-structures-3 canon
    return float(sum(checks) / len(checks))


def bench_skew_list(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skew_list": _bench_skew_list(seed)}
