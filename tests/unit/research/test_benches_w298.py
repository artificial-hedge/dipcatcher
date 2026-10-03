"""Adapter tests for wave-298 medical-imaging canon benches."""

from quant_fund.research.benches_w298 import (
    bench_art_sirt_family,
    bench_chan_vese_family,
    bench_cs_mri_family,
    bench_hu_moments_family,
    bench_mi_register_family,
    bench_radon_fbp_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_radon_fbp_family,
        bench_art_sirt_family,
        bench_cs_mri_family,
        bench_hu_moments_family,
        bench_chan_vese_family,
        bench_mi_register_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
