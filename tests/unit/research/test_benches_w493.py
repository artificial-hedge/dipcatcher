"""Wave-493 tropical-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w493 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tropical_poly": b.bench_tropical_poly_family(),
        "berkovich_an": b.bench_berkovich_an_family(),
        "skeleton_trop": b.bench_skeleton_trop_family(),
        "tropical_curve": b.bench_tropical_curve_family(),
        "mikhalkin": b.bench_mikhalkin_family(),
        "tropical_cycle": b.bench_tropical_cycle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
