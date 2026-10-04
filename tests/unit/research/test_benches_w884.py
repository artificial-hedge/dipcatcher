"""Wave-884 wavelet/spectral adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w884 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wavelet_matrix": b.bench_wavelet_matrix_family(),
        "second_gen_wavelet": b.bench_second_gen_wavelet_family(),
        "nodal_dg": b.bench_nodal_dg_family(),
        "multidomain_sem": b.bench_multidomain_sem_family(),
        "boundary_element": b.bench_boundary_element_family(),
        "marquina_flux": b.bench_marquina_flux_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
