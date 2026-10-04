"""Wave-952 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w952 import (
    bench_eigval_bounds_family,
    bench_power_deflation_family,
    bench_qr_iteration_family,
    bench_schur_decomp_family,
    bench_spectral_gap_family,
    bench_spectral_radius_family,
)

_FAMILY_BENCHES = [
    bench_qr_iteration_family,
    bench_power_deflation_family,
    bench_schur_decomp_family,
    bench_eigval_bounds_family,
    bench_spectral_radius_family,
    bench_spectral_gap_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
