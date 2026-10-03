"""Wave-488 derived-geometry-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w488 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "shifted_tangent": b.bench_shifted_tangent_family(),
        "derived_quot": b.bench_derived_quot_family(),
        "virtual_pull": b.bench_virtual_pull_family(),
        "intrinsic_be": b.bench_intrinsic_be_family(),
        "d_critical": b.bench_d_critical_family(),
        "perfect_obstruction": b.bench_perfect_obstruction_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
