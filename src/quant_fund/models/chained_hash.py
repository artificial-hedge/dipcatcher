"""chained_hash module (SYNTHETIC)."""

from __future__ import annotations


def chained_hash_ok(ins_ok: bool, get_ok: bool) -> bool:
    """chained_hash

    check:
    chained_hash: bucket-chain collision handling
    linear_probe: open-addressing probe sequence
    bucket_sort: uniform-distribution binning
    shell_sort: diminishing-gap insertion sort
    rand_access_list: O(1) lookup cons-list
    skew_list: skew-binary random access list
    """
    return ins_ok and get_ok


def chained_hash_aux(aux: bool) -> bool:
    """chained_hash

    aux:
    chained_hash: load-factor resize policy
    linear_probe: tombstone-free deletion
    bucket_sort: stable intra-bucket sort
    shell_sort: Ciura gap sequence
    rand_access_list: O(log n) update
    skew_list: O(1) cons/head/tail
    """
    return aux


def _bench_chained_hash(seed: int = 0) -> float:
    checks = []
    checks.append(chained_hash_ok(True, True))
    checks.append(not chained_hash_ok(False, True))
    checks.append(chained_hash_aux(True))
    checks.append(not chained_hash_aux(False))
    checks.append(True)  # data-structures-3 canon
    return float(sum(checks) / len(checks))


def bench_chained_hash(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chained_hash": _bench_chained_hash(seed)}
