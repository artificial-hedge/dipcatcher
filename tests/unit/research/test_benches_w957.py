"""Wave-957 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w957 import (
    bench_fredholm_op_family,
    bench_multiplication_op_family,
    bench_normal_operator_family,
    bench_selfadjoint_op_family,
    bench_shift_operator_family,
    bench_unitary_operator_family,
)

_FAMILY_BENCHES = [
    bench_selfadjoint_op_family,
    bench_unitary_operator_family,
    bench_shift_operator_family,
    bench_fredholm_op_family,
    bench_normal_operator_family,
    bench_multiplication_op_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
