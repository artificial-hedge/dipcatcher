"""Adapter tests for wave-287 differential-geometry canon benches."""

from quant_fund.research.benches_w287 import (
    bench_christoffel_family,
    bench_first_ff_family,
    bench_frenet_frame_family,
    bench_gauss_curve_family,
    bench_geodesic_sphere_family,
    bench_surf_area_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_first_ff_family,
        bench_gauss_curve_family,
        bench_frenet_frame_family,
        bench_christoffel_family,
        bench_geodesic_sphere_family,
        bench_surf_area_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
