"""Wave-797 2BSDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w797 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "second_bsde": b.bench_second_bsde_family(),
        "doubly_bsde": b.bench_doubly_bsde_family(),
        "reflected_bsde2": b.bench_reflected_bsde2_family(),
        "obstacle_bsde": b.bench_obstacle_bsde_family(),
        "quadratic_bsde": b.bench_quadratic_bsde_family(),
        "super_linear": b.bench_super_linear_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
