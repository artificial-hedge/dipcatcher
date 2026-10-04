"""Adapter tests for wave-295 astronomy/orbital-mechanics canon benches."""

from quant_fund.research.benches_w295 import (
    bench_gauss_iod_family,
    bench_kepler_solve_family,
    bench_lambert_problem_family,
    bench_orbit_maneuver_family,
    bench_orbital_elements_family,
    bench_tle_propagate_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_orbital_elements_family,
        bench_kepler_solve_family,
        bench_lambert_problem_family,
        bench_tle_propagate_family,
        bench_orbit_maneuver_family,
        bench_gauss_iod_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
