"""Wave-887 solver/transport adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w887 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "epi_rk": b.bench_epi_rk_family(),
        "gauss_rk": b.bench_gauss_rk_family(),
        "adjoint_sparse": b.bench_adjoint_sparse_family(),
        "element_free": b.bench_element_free_family(),
        "diffusion_approx_sp": b.bench_diffusion_approx_sp_family(),
        "importance_rel": b.bench_importance_rel_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
