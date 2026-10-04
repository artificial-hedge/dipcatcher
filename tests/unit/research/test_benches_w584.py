"""Wave-584 condensed-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w584 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "clausen_scholze2": b.bench_clausen_scholze2_family(),
        "solid_cohom": b.bench_solid_cohom_family(),
        "nuclear_space": b.bench_nuclear_space_family(),
        "analytic_sheaf": b.bench_analytic_sheaf_family(),
        "solid_tensor2": b.bench_solid_tensor2_family(),
        "proetale_site2": b.bench_proetale_site2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
