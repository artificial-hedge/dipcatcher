"""Wave-959 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w959 import (
    bench_accretive_op_family,
    bench_contraction_op_family,
    bench_differential_op_family,
    bench_integral_op_family,
    bench_sectorial_op_family,
    bench_toeplitz_op_family,
)

_FAMILY_BENCHES = [
    bench_toeplitz_op_family,
    bench_integral_op_family,
    bench_differential_op_family,
    bench_contraction_op_family,
    bench_accretive_op_family,
    bench_sectorial_op_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
