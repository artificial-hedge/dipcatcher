"""Wave-898 BVP adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w898 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "shooting_bvp": b.bench_shooting_bvp_family(),
        "multiple_shooting": b.bench_multiple_shooting_family(),
        "collocation_bvp": b.bench_collocation_bvp_family(),
        "finite_diff_bvp": b.bench_finite_diff_bvp_family(),
        "relaxation_bvp": b.bench_relaxation_bvp_family(),
        "riccati_bvp": b.bench_riccati_bvp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
