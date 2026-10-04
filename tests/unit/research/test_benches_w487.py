"""Wave-487 category-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w487 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "compact_obj": b.bench_compact_obj_family(),
        "dualizable_cat": b.bench_dualizable_cat_family(),
        "comma_cat": b.bench_comma_cat_family(),
        "prestack": b.bench_prestack_family(),
        "endo_prof": b.bench_endo_prof_family(),
        "exact_cat": b.bench_exact_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
