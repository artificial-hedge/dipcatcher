"""Wave-637 p-adic-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w637 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fargues_scholze2": b.bench_fargues_scholze2_family(),
        "curve_padic": b.bench_curve_padic_family(),
        "diamond_mod": b.bench_diamond_mod_family(),
        "etale_phiphi": b.bench_etale_phiphi_family(),
        "cocartesian_diamond": b.bench_cocartesian_diamond_family(),
        "scholze_bc": b.bench_scholze_bc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
