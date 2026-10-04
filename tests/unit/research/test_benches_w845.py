"""Wave-845 integral-transforms adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w845 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "laplace_transform": b.bench_laplace_transform_family(),
        "mellin_transform": b.bench_mellin_transform_family(),
        "hankel_transform": b.bench_hankel_transform_family(),
        "z_transform": b.bench_z_transform_family(),
        "hilbert_transform": b.bench_hilbert_transform_family(),
        "abel_transform": b.bench_abel_transform_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
