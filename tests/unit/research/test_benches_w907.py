"""Wave-907 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w907 import (
    bench_dsu_rollback_family,
    bench_interval_heap_family,
    bench_potential_dsu_family,
    bench_union_find_family,
    bench_van_emde_boas_family,
    bench_weak_heap_family,
)

_FAMILY_BENCHES = [
    bench_union_find_family,
    bench_dsu_rollback_family,
    bench_potential_dsu_family,
    bench_van_emde_boas_family,
    bench_interval_heap_family,
    bench_weak_heap_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
