"""Wave-678 category-16 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w678 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "simplicial_cat": b.bench_simplicial_cat_family(),
        "homotopical_cat": b.bench_homotopical_cat_family(),
        "relative_cat": b.bench_relative_cat_family(),
        "equipment_cat": b.bench_equipment_cat_family(),
        "fibrant_cat": b.bench_fibrant_cat_family(),
        "pointed_cat": b.bench_pointed_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
