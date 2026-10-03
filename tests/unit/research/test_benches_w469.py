"""Wave-469 category-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w469 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "enriched_cat": b.bench_enriched_cat_family(),
        "weight_lim": b.bench_weight_lim_family(),
        "fibered_cat": b.bench_fibered_cat_family(),
        "derivator2": b.bench_derivator2_family(),
        "accessible_cat": b.bench_accessible_cat_family(),
        "day_conv": b.bench_day_conv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
