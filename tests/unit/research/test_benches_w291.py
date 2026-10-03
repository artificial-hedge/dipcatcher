"""Adapter tests for wave-291 VLSI/EDA canon benches."""

from quant_fund.research.benches_w291 import (
    bench_a_star_route_family,
    bench_drc_check_family,
    bench_levelize_family,
    bench_netlist_parse_family,
    bench_place_quadratic_family,
    bench_sta_timing_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_netlist_parse_family,
        bench_sta_timing_family,
        bench_a_star_route_family,
        bench_drc_check_family,
        bench_place_quadratic_family,
        bench_levelize_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
