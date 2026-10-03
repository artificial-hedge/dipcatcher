"""Wave-948 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w948 import (
    bench_circulant_matrix_family,
    bench_companion_matrix_family,
    bench_hankel_matrix_family,
    bench_hessenberg_form_family,
    bench_krylov_matrix_family,
    bench_vandermonde_matrix_family,
)

_FAMILY_BENCHES = [
    bench_circulant_matrix_family,
    bench_companion_matrix_family,
    bench_vandermonde_matrix_family,
    bench_krylov_matrix_family,
    bench_hessenberg_form_family,
    bench_hankel_matrix_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
