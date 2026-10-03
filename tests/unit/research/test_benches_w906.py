"""Wave-906 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w906 import (
    bench_finger_tree_family,
    bench_persistent_array_family,
    bench_pure_queue_family,
    bench_rope_string_family,
    bench_skip_list_family,
    bench_vlist_family,
)

_FAMILY_BENCHES = [
    bench_skip_list_family,
    bench_persistent_array_family,
    bench_finger_tree_family,
    bench_rope_string_family,
    bench_vlist_family,
    bench_pure_queue_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
