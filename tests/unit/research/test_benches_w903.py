"""Wave-903 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w903 import (
    bench_cuckoo_hash_family,
    bench_hopscotch_hash_family,
    bench_open_addr_hash_family,
    bench_perfect_hash_family,
    bench_robin_hood_hash_family,
    bench_swiss_table_family,
)

_FAMILY_BENCHES = [
    bench_cuckoo_hash_family,
    bench_hopscotch_hash_family,
    bench_robin_hood_hash_family,
    bench_swiss_table_family,
    bench_open_addr_hash_family,
    bench_perfect_hash_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
