"""Adapter tests for wave-301 geophysics/seismic canon benches."""

from quant_fund.research.benches_w301 import (
    bench_avo_shuey_family,
    bench_eikonal_fmm_family,
    bench_kirchhoff_mig_family,
    bench_nmo_dix_family,
    bench_taup_transform_family,
    bench_vibroseis_sweep_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_nmo_dix_family,
        bench_taup_transform_family,
        bench_kirchhoff_mig_family,
        bench_avo_shuey_family,
        bench_vibroseis_sweep_family,
        bench_eikonal_fmm_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
