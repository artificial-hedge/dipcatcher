"""Adapter tests for wave-296 robotics-4 canon benches."""

from quant_fund.research.benches_w296 import (
    bench_chomp_family,
    bench_gjk_epa_family,
    bench_ilqr_family,
    bench_lqr_funnel_family,
    bench_rts_smoother_family,
    bench_se3_spline_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_lqr_funnel_family,
        bench_chomp_family,
        bench_gjk_epa_family,
        bench_ilqr_family,
        bench_rts_smoother_family,
        bench_se3_spline_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
