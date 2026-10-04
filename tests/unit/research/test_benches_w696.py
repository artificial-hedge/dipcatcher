"""Wave-696 category-19 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w696 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_rank": b.bench_cat_rank_family(),
        "cat_index": b.bench_cat_index_family(),
        "cat_monotone": b.bench_cat_monotone_family(),
        "cat_kernel": b.bench_cat_kernel_family(),
        "cat_image": b.bench_cat_image_family(),
        "cat_pullback": b.bench_cat_pullback_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
