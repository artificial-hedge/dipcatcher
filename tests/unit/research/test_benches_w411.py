"""Wave-411 algebraic-geometry-9 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w411 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "blow_up": b.bench_blow_up_family(),
        "intersection_mult": b.bench_intersection_mult_family(),
        "tangent_cone": b.bench_tangent_cone_family(),
        "normalization": b.bench_normalization_family(),
        "divisor_class": b.bench_divisor_class_family(),
        "dualizing": b.bench_dualizing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
