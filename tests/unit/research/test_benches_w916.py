"""Wave-916 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w916 import (
    bench_da_trie_family,
    bench_fst_index_family,
    bench_hollow_dsu_family,
    bench_hollow_heap_family,
    bench_rank_pairing_family,
    bench_soft_heap_family,
)

_FAMILY_BENCHES = [
    bench_soft_heap_family,
    bench_hollow_heap_family,
    bench_rank_pairing_family,
    bench_hollow_dsu_family,
    bench_da_trie_family,
    bench_fst_index_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
