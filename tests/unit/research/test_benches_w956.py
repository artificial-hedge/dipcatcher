"""Wave-956 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w956 import (
    bench_cp_rank_family,
    bench_mode_n_product_family,
    bench_tensor_norm_family,
    bench_tensor_symmetry_family,
    bench_tensor_trace_family,
    bench_tucker_rank_family,
)

_FAMILY_BENCHES = [
    bench_tucker_rank_family,
    bench_cp_rank_family,
    bench_tensor_norm_family,
    bench_tensor_trace_family,
    bench_mode_n_product_family,
    bench_tensor_symmetry_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
