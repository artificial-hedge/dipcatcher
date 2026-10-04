"""Wave-955 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w955 import (
    bench_back_substitution_family,
    bench_forward_substitution_family,
    bench_givens_rotation_family,
    bench_gram_determinant_family,
    bench_gram_matrix_family,
    bench_householder_reflect_family,
)

_FAMILY_BENCHES = [
    bench_gram_matrix_family,
    bench_gram_determinant_family,
    bench_householder_reflect_family,
    bench_givens_rotation_family,
    bench_back_substitution_family,
    bench_forward_substitution_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
