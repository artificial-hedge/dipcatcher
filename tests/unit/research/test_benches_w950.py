"""Wave-950 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w950 import (
    bench_hadamard_product_family,
    bench_khatri_rao_family,
    bench_kron_product_family,
    bench_outer_product_family,
    bench_tensor_contraction_family,
    bench_tensor_unfold_family,
)

_FAMILY_BENCHES = [
    bench_tensor_contraction_family,
    bench_khatri_rao_family,
    bench_kron_product_family,
    bench_hadamard_product_family,
    bench_tensor_unfold_family,
    bench_outer_product_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
