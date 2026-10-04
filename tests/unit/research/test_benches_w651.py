"""Wave-651 category-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w651 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "flat_functor": b.bench_flat_functor_family(),
        "filtered_cat": b.bench_filtered_cat_family(),
        "sifted_cat2": b.bench_sifted_cat2_family(),
        "regular_cat": b.bench_regular_cat_family(),
        "abelian_cat": b.bench_abelian_cat_family(),
        "malcev_cat": b.bench_malcev_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
