"""Wave-677 category-15 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w677 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derivator_cat": b.bench_derivator_cat_family(),
        "quillen_cat": b.bench_quillen_cat_family(),
        "combinatorial_mc": b.bench_combinatorial_mc_family(),
        "cat_dg": b.bench_cat_dg_family(),
        "univalent_cat": b.bench_univalent_cat_family(),
        "cat_structure": b.bench_cat_structure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
