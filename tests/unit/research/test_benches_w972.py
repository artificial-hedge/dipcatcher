"""Wave-972 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w972 import (
    bench_free_berg_family,
    bench_free_cumulant_family,
    bench_free_entropy_family,
    bench_free_fisher_info_family,
    bench_freeness_check_family,
    bench_matrix_model_free_family,
)

_FAMILY_BENCHES = [
    bench_free_entropy_family,
    bench_free_fisher_info_family,
    bench_free_cumulant_family,
    bench_freeness_check_family,
    bench_matrix_model_free_family,
    bench_free_berg_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
