"""Wave-705 derived-geometry-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w705 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_geometry7": b.bench_derived_geometry7_family(),
        "derived_abelian2": b.bench_derived_abelian2_family(),
        "derived_stack3": b.bench_derived_stack3_family(),
        "derived_morph": b.bench_derived_morph_family(),
        "derived_cover": b.bench_derived_cover_family(),
        "derived_topos": b.bench_derived_topos_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
