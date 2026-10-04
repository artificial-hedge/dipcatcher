"""Wave-855 wavelet-Galerkin adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w855 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wavelet_galerkin": b.bench_wavelet_galerkin_family(),
        "daubechies_basis": b.bench_daubechies_basis_family(),
        "coiflet_basis": b.bench_coiflet_basis_family(),
        "spline_wavelet": b.bench_spline_wavelet_family(),
        "wavelet_collocation": b.bench_wavelet_collocation_family(),
        "adapt_wavelet": b.bench_adapt_wavelet_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
