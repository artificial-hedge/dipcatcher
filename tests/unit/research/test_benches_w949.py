"""Wave-949 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w949 import (
    bench_deflating_subspace_family,
    bench_invariant_subspace_family,
    bench_jordan_form_family,
    bench_kronecker_canonical_family,
    bench_matrix_pencil_family,
    bench_rational_canonical_family,
)

_FAMILY_BENCHES = [
    bench_matrix_pencil_family,
    bench_kronecker_canonical_family,
    bench_invariant_subspace_family,
    bench_deflating_subspace_family,
    bench_jordan_form_family,
    bench_rational_canonical_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
