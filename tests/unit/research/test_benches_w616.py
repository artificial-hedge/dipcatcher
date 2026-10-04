"""Wave-616 tensor-category-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w616 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "multifusion": b.bench_multifusion_family(),
        "premodular2": b.bench_premodular2_family(),
        "braided_functor": b.bench_braided_functor_family(),
        "center_cat": b.bench_center_cat_family(),
        "fusion_ring": b.bench_fusion_ring_family(),
        "ds_category": b.bench_ds_category_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
