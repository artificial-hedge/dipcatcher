"""Wave-709 category-22 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w709 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_univariant2": b.bench_cat_univariant2_family(),
        "cat_ab2": b.bench_cat_ab2_family(),
        "cat_exact3": b.bench_cat_exact3_family(),
        "cat_freyd": b.bench_cat_freyd_family(),
        "cat_ab_loc": b.bench_cat_ab_loc_family(),
        "cat_pro_object2": b.bench_cat_pro_object2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
