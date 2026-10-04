"""Wave-646 p-adic-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w646 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fargues_cat": b.bench_fargues_cat_family(),
        "v_stack": b.bench_v_stack_family(),
        "untilt2": b.bench_untilt2_family(),
        "spatial_diamond": b.bench_spatial_diamond_family(),
        "diamond_sheaf": b.bench_diamond_sheaf_family(),
        "bdr_plus": b.bench_bdr_plus_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
