"""Wave-607 infinity-categories-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w607 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kan_complex": b.bench_kan_complex_family(),
        "horn_filler": b.bench_horn_filler_family(),
        "nerve_cat": b.bench_nerve_cat_family(),
        "mapping_space": b.bench_mapping_space_family(),
        "homotopy_cat": b.bench_homotopy_cat_family(),
        "cocartesian": b.bench_cocartesian_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
