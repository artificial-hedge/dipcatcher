"""Wave-997 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w997 import (
    bench_bilinear_estimates_family,
    bench_i_method_family,
    bench_kdv_dispersion_family,
    bench_local_smoothing_family,
    bench_nls_dispersion_family,
    bench_strichartz_estimates_family,
)

_FAMILY_BENCHES = [
    bench_nls_dispersion_family,
    bench_kdv_dispersion_family,
    bench_strichartz_estimates_family,
    bench_local_smoothing_family,
    bench_bilinear_estimates_family,
    bench_i_method_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
