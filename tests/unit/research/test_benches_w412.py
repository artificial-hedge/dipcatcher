"""Wave-412 2-category adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w412 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "two_cat": b.bench_two_cat_family(),
        "bicat_comp": b.bench_bicat_comp_family(),
        "mate_calc": b.bench_mate_calc_family(),
        "double_cat": b.bench_double_cat_family(),
        "lax_functor": b.bench_lax_functor_family(),
        "cat_enriched": b.bench_cat_enriched_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
