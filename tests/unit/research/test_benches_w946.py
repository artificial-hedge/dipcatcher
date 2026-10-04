"""Wave-946 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w946 import (
    bench_cholesky_piv_family,
    bench_douglas_factor_family,
    bench_matrix_square_root_family,
    bench_perron_frobenius_family,
    bench_polar_decomp_family,
    bench_sylvester_matrix_family,
)

_FAMILY_BENCHES = [
    bench_perron_frobenius_family,
    bench_douglas_factor_family,
    bench_cholesky_piv_family,
    bench_matrix_square_root_family,
    bench_polar_decomp_family,
    bench_sylvester_matrix_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
