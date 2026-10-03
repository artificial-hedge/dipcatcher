"""Wave-601 derived-geometry-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w601 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dg_algebra": b.bench_dg_algebra_family(),
        "derived_loop": b.bench_derived_loop_family(),
        "derived_tangent": b.bench_derived_tangent_family(),
        "virtual_fund": b.bench_virtual_fund_family(),
        "structured_space": b.bench_structured_space_family(),
        "e_infinity_ring": b.bench_e_infinity_ring_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
