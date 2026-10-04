"""Wave-656 category-12 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w656 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "essentially_small": b.bench_essentially_small_family(),
        "finitely_accessible": b.bench_finitely_accessible_family(),
        "admissible_cat": b.bench_admissible_cat_family(),
        "definable_cat": b.bench_definable_cat_family(),
        "cocomplete_cat": b.bench_cocomplete_cat_family(),
        "cartesian_cat2": b.bench_cartesian_cat2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
