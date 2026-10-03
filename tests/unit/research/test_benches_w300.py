"""Adapter tests for wave-300 astronomy-2 canon benches."""

from quant_fund.research.benches_w300 import (
    bench_delta_t_family,
    bench_eclipse_circ_family,
    bench_equinox_prec_family,
    bench_nutation_lite_family,
    bench_planet_vsop_family,
    bench_rise_set_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_equinox_prec_family,
        bench_nutation_lite_family,
        bench_rise_set_family,
        bench_eclipse_circ_family,
        bench_delta_t_family,
        bench_planet_vsop_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
