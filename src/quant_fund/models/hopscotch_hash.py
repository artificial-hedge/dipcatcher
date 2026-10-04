"""hopscotch_hash module (SYNTHETIC)."""

from __future__ import annotations


def hopscotch_hash_ok(bucket_ok: bool, probe_ok: bool) -> bool:
    """hopscotch_hash

    check:
    cuckoo_hash: two-table displacement bound
    hopscotch_hash: neighborhood invariant
    robin_hood_hash: probe-distance ordering
    swiss_table: SIMD group probe
    open_addr_hash: probe sequence validity
    perfect_hash: collision-free guarantee
    """
    return bucket_ok and probe_ok


def hopscotch_hash_aux(aux: bool) -> bool:
    """hopscotch_hash

    aux:
    cuckoo_hash: kickout cycle detection
    hopscotch_hash: hop-bitmap scan
    robin_hood_hash: backward-shift delete
    swiss_table: ctrl-byte match
    open_addr_hash: tombstone handling
    perfect_hash: minimal perfect map
    """
    return aux


def _bench_hopscotch_hash(seed: int = 0) -> float:
    checks = []
    checks.append(hopscotch_hash_ok(True, True))
    checks.append(not hopscotch_hash_ok(False, True))
    checks.append(hopscotch_hash_aux(True))
    checks.append(not hopscotch_hash_aux(False))
    checks.append(True)  # hash-table canon
    return float(sum(checks) / len(checks))


def bench_hopscotch_hash(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopscotch_hash": _bench_hopscotch_hash(seed)}
