"""Wave-717 noncommutative-motives adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w717 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "nc_motive": b.bench_nc_motive_family(),
        "dg_enhancement": b.bench_dg_enhancement_family(),
        "bondal_kapranov": b.bench_bondal_kapranov_family(),
        "enhanced_triangulated": b.bench_enhanced_triangulated_family(),
        "tabuada_motive": b.bench_tabuada_motive_family(),
        "nc_k_theory": b.bench_nc_k_theory_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
