"""Wave-1018 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1018 import (
    bench_four_vectors_family,
    bench_geodesic_motion_family,
    bench_gravitational_lensing_family,
    bench_gravitational_waves_family,
    bench_lorentz_transformation_family,
    bench_spacetime_interval_family,
)

_FAMILY_BENCHES = [
    bench_lorentz_transformation_family,
    bench_spacetime_interval_family,
    bench_four_vectors_family,
    bench_geodesic_motion_family,
    bench_gravitational_lensing_family,
    bench_gravitational_waves_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
