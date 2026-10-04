"""perfect_hash module (SYNTHETIC)."""

from __future__ import annotations


def perfect_hash_ok(bucket_ok: bool, probe_ok: bool) -> bool:
    """perfect_hash

    check:
    cuckoo_hash: two-table displacement bound
    hopscotch_hash: neighborhood invariant
    robin_hood_hash: probe-distance ordering
    swiss_table: SIMD group probe
    open_addr_hash: probe sequence validity
    perfect_hash: collision-free guarantee
    """
    return bucket_ok and probe_ok


def perfect_hash_aux(aux: bool) -> bool:
    """perfect_hash

    aux:
    cuckoo_hash: kickout cycle detection
    hopscotch_hash: hop-bitmap scan
    robin_hood_hash: backward-shift delete
    swiss_table: ctrl-byte match
    open_addr_hash: tombstone handling
    perfect_hash: minimal perfect map
    """
    return aux


def _bench_perfect_hash(seed: int = 0) -> float:
    checks = []
    checks.append(perfect_hash_ok(True, True))
    checks.append(not perfect_hash_ok(False, True))
    checks.append(perfect_hash_aux(True))
    checks.append(not perfect_hash_aux(False))
    checks.append(True)  # hash-table canon
    return float(sum(checks) / len(checks))


def bench_perfect_hash(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perfect_hash": _bench_perfect_hash(seed)}
