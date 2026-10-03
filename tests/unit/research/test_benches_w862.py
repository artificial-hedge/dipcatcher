"""Wave-862 sparse-grid/dimension-adaptive adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w862 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "smolyak_grid": b.bench_smolyak_grid_family(),
        "sparse_tensor": b.bench_sparse_tensor_family(),
        "anisotropic_quad": b.bench_anisotropic_quad_family(),
        "gerstner_griebel": b.bench_gerstner_griebel_family(),
        "combination_technique": b.bench_combination_technique_family(),
        "dimension_adaptive": b.bench_dimension_adaptive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
