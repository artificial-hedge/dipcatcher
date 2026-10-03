"""Wave-708 category-21 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w708 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_pseudo_limit": b.bench_cat_pseudo_limit_family(),
        "cat_weak_eq": b.bench_cat_weak_eq_family(),
        "cat_reedy_cat": b.bench_cat_reedy_cat_family(),
        "cat_dold_kan": b.bench_cat_dold_kan_family(),
        "cat_hoc": b.bench_cat_hoc_family(),
        "cat_enriched_lim": b.bench_cat_enriched_lim_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
