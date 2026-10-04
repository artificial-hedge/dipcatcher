"""Wave-712 motivic-24 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w712 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_additive": b.bench_motivic_additive_family(),
        "motivic_additive_cat": b.bench_motivic_additive_cat_family(),
        "motivic_cover": b.bench_motivic_cover_family(),
        "motivic_gysin2": b.bench_motivic_gysin2_family(),
        "motivic_chern2": b.bench_motivic_chern2_family(),
        "motivic_filtration2": b.bench_motivic_filtration2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
