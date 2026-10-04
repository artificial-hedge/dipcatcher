"""Wave-954 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w954 import (
    bench_condition_number_family,
    bench_low_rank_approx_family,
    bench_matrix_truncate_family,
    bench_nuclear_norm_family,
    bench_rank_estimate_family,
    bench_spectral_threshold_family,
)

_FAMILY_BENCHES = [
    bench_low_rank_approx_family,
    bench_nuclear_norm_family,
    bench_spectral_threshold_family,
    bench_matrix_truncate_family,
    bench_rank_estimate_family,
    bench_condition_number_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
