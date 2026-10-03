"""Wave-667 category-13 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w667 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stable_cat2": b.bench_stable_cat2_family(),
        "exact_cat2": b.bench_exact_cat2_family(),
        "ab_cat": b.bench_ab_cat_family(),
        "grothendieck_cat": b.bench_grothendieck_cat_family(),
        "coniveau_fil": b.bench_coniveau_fil_family(),
        "special_cat": b.bench_special_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
