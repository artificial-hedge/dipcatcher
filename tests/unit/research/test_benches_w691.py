"""Wave-691 category-18 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w691 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_pretopos": b.bench_cat_pretopos_family(),
        "cat_semisimple": b.bench_cat_semisimple_family(),
        "cat_fusion": b.bench_cat_fusion_family(),
        "cat_tannakian2": b.bench_cat_tannakian2_family(),
        "cat_ribbon": b.bench_cat_ribbon_family(),
        "cat_semiadd": b.bench_cat_semiadd_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
