"""Wave-592 tensor-category adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w592 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tensor_cat": b.bench_tensor_cat_family(),
        "braided_cat": b.bench_braided_cat_family(),
        "rigid_cat": b.bench_rigid_cat_family(),
        "fusion_cat": b.bench_fusion_cat_family(),
        "spherical_cat": b.bench_spherical_cat_family(),
        "premodular": b.bench_premodular_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
