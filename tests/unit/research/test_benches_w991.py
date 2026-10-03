"""Wave-991 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w991 import (
    bench_bloch_decomp_family,
    bench_gamma_convergence_family,
    bench_h_convergence_family,
    bench_homogenization_family,
    bench_mosco_conv_family,
    bench_two_scale_conv_family,
)

_FAMILY_BENCHES = [
    bench_homogenization_family,
    bench_two_scale_conv_family,
    bench_gamma_convergence_family,
    bench_mosco_conv_family,
    bench_bloch_decomp_family,
    bench_h_convergence_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
