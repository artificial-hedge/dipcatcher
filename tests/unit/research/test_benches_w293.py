"""Adapter tests for wave-293 graphics-3 canon benches."""

from quant_fund.research.benches_w293 import (
    bench_deferred_shade_family,
    bench_env_map_family,
    bench_frustum_cull_family,
    bench_lod_select_family,
    bench_sdf_raymarch_family,
    bench_shadow_pcf_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_deferred_shade_family,
        bench_sdf_raymarch_family,
        bench_frustum_cull_family,
        bench_lod_select_family,
        bench_env_map_family,
        bench_shadow_pcf_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
