"""Wave-687 category-17 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w687 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_fibrant_obj": b.bench_cat_fibrant_obj_family(),
        "cat_cofibrant": b.bench_cat_cofibrant_family(),
        "cat_bicomplete": b.bench_cat_bicomplete_family(),
        "cat_univariant": b.bench_cat_univariant_family(),
        "cat_descent": b.bench_cat_descent_family(),
        "cat_glueable": b.bench_cat_glueable_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
