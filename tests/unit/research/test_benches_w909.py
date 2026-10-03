"""Wave-909 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w909 import (
    bench_deque_array_family,
    bench_doubly_linked_list_family,
    bench_gap_buffer_family,
    bench_piece_table_family,
    bench_unrolled_list_family,
    bench_xor_linked_list_family,
)

_FAMILY_BENCHES = [
    bench_doubly_linked_list_family,
    bench_unrolled_list_family,
    bench_gap_buffer_family,
    bench_piece_table_family,
    bench_deque_array_family,
    bench_xor_linked_list_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
