"""Wave-965 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w965 import (
    bench_compact_normal_family,
    bench_hilbert_schmidt_op_family,
    bench_polar_operator_family,
    bench_schmidt_decomp_family,
    bench_singular_value_op_family,
    bench_trace_class_op_family,
)

_FAMILY_BENCHES = [
    bench_hilbert_schmidt_op_family,
    bench_trace_class_op_family,
    bench_singular_value_op_family,
    bench_schmidt_decomp_family,
    bench_compact_normal_family,
    bench_polar_operator_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
