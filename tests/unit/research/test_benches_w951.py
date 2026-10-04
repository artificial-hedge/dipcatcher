"""Wave-951 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w951 import (
    bench_determinant_cofactor_family,
    bench_frechet_derivative_family,
    bench_kronecker_sum_family,
    bench_matrix_exponential_family,
    bench_permanent_matrix_family,
    bench_vec_operator_family,
)

_FAMILY_BENCHES = [
    bench_determinant_cofactor_family,
    bench_permanent_matrix_family,
    bench_matrix_exponential_family,
    bench_frechet_derivative_family,
    bench_vec_operator_family,
    bench_kronecker_sum_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
